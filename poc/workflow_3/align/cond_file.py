"""align 이미지에 딸린 ``cond.txt`` 조건 파일을 읽어 좌표를 뽑아내는 파서.

오피스 다운로더가 각 이미지 옆에 숨김 폴더로 cond.txt 를 떨군다:
``<image>.jpeg`` → ``.<image>.jpeg/cond.txt`` (파일명 그대로, 앞에 점).

cond.txt 한 줄 형식: ``key  값,값,...`` (key 와 값 사이는 공백/탭, 값끼리는 콤마).
우리가 쓰는 키 ([[project_align_cond_files_and_coords]]):
  - ``Scope``        : OM / SEM (modality — fail 멈춘 step 의 종류)
  - ``Pixel``        : 이미지 크기 (예: 512,512 / 1024,1024)
  - ``Magnification``: 등록 배율 (OM 은 104/210, SEM 은 5000 처럼 큰 값).
  - ``Image_rotation``: 등록 당시 화면 회전(도) — SEM key 가 OM 과 다른 회전으로
        등록되는 경우가 있어 보정 전에 장비 회전을 맞춰야 한다.
  - ``!OM_Brightness``: OM 등록 당시 밝기. 값이 ``OM_DARK_BRIGHTNESS_MAX`` 미만이면
        recipe 가 **OM-D**(암시야, OM 의 명암 반전)로 등록된 것이다 → live 화면도
        OM-D 여야 매칭이 된다. ``required_image_mode`` 참고.
  - ``!Cursor_info`` : crosshair / white box 좌표가 한 줄에 들어 있다.
        elements[4],[5]      = crosshair (cx, cy)        — 둘 다 -1 이 아니면 존재
        elements[6],[7],[8],[9] = white box (left, top, right, bottom)
                              — [8],[9] 가 -1 이 아니면 존재
    cursor 좌표는 Pixel 의 10배 oversample 프레임이다(이미지 px = cursor/10).
    실제 이미지 위 좌표 변환은 본 파서가 아니라 그리기/inpaint 단계에서 적용한다.
"""

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# !Cursor_info 요소 인덱스 (0-base). 좌표는 cursor oversample 프레임 기준 raw 값.
_CROSSHAIR_IDX = (4, 5)
_BOX_IDX = (6, 7, 8, 9)

# Scope 가 **없을 때만** 쓰는 폴백 임계: !OM_Brightness 가 이 값 미만이면 OM-D 로 본다.
# 출처: 사용자 2026-09-22, 본인도 "I assume" 라고 밝힌 **가정**이다. 같은 날 Scope 가
# msr cond 에도 있고 OM/OMDF 를 정확히 가른다는 것이 확인되어 1순위에서 내려왔다 -
# 이제 Scope 없는 cond 를 위한 최후 수단이다. 등호 없음(< 만).
OM_DARK_BRIGHTNESS_MAX = 35000.0


@dataclass(frozen=True)
class CondInfo:
    """cond.txt 한 장에서 뽑은 조건 (없는 항목은 None)."""

    scope: str | None = None                       # "OM" / "SEM" (원문 토큰)
    pixel: tuple[int, int] | None = None           # (width, height)
    box_ltrb: tuple[int, int, int, int] | None = None   # cursor 프레임 raw 좌표
    crosshair_xy: tuple[int, int] | None = None         # cursor 프레임 raw 좌표
    raw: dict[str, list[str]] = field(default_factory=dict)  # key → 값 토큰 (디버그용)

    @property
    def is_sem(self) -> bool:
        return bool(self.scope) and "SEM" in self.scope.upper()

    @property
    def is_om(self) -> bool:
        return bool(self.scope) and "OM" in self.scope.upper()

    @property
    def is_om_dark(self) -> bool:
        """Scope 가 OMDF(암시야 OM)인가. OM 계열이면서 dark - is_om 도 참이다."""
        return self.is_om and "DF" in self.scope.upper()

    @property
    def magnification(self) -> float | None:
        tokens = self.raw.get("magnification") or []
        value = _to_int(tokens[0]) if tokens else None
        return float(value) if value is not None and value > 0 else None

    @property
    def image_rotation(self) -> float | None:
        """``Image_rotation`` (도). 등록 당시 화면 회전 - 없으면 None.

        SEM align key 는 OM 과 다른 회전으로 등록되는 경우가 있다(사용자 2026-09-16).
        회전이 어긋난 채로 매칭하면 template 이 회전만큼 안 맞아 점수가 떨어지므로,
        보정 전에 장비 회전을 이 값으로 맞춰야 한다(SEM Monitor 의 'Rot..' 버튼).
        0 과 None 은 다르다 - 0 은 '회전 없음'(유효한 값), None 은 '모른다'.

        키 철자는 실데이터에서 흔들리므로(``!Cursor_inf`` 선례) ``image_rot`` 접두 →
        ``rotat`` 포함 순으로 찾는다. 값이 숫자가 아니면 None.
        """
        tokens = next((v for k, v in self.raw.items() if k.startswith("image_rot")), None)
        if tokens is None:
            tokens = next((v for k, v in self.raw.items() if "rotat" in k), None)
        value = _to_float(tokens[0]) if tokens else None
        # float("nan")/float("inf") 는 예외 없이 통과한다 - 이 값은 장비 회전을 실제로
        # 돌리는 데 쓰이므로 파서에서 막는다(소비처마다 막으면 하나는 빼먹는다).
        return value if value is not None and math.isfinite(value) else None

    @property
    def om_brightness(self) -> float | None:
        """``!OM_Brightness`` 값 (없으면 None). 0 은 유효한 값이라 None 과 다르다."""
        tokens = self.raw.get("om_brightness") or []
        value = _to_float(tokens[0]) if tokens else None
        return value if value is not None and math.isfinite(value) else None

    @property
    def required_image_mode(self) -> str | None:
        """이 key 가 등록된 **live 화면 모드** "OM" | "OM-D" | "SEM" (모르면 None).

        화면 모드는 recipe 가 자동으로 맞춰 주지 않고 **수동 설정**이라
        ([[project_om_d_polarity_and_image_mode]]) 화면이 recipe 와 다른 모드일 수 있고
        그 자체가 align fail 의 원인이 된다. OM-D 는 OM 의 명암 **반전**이라 모드가
        어긋나면 NCC 가 ≈ −1 이 되고 ``max(0, ncc)`` 가 0 으로 눌러 완벽한 key 라도
        sel <= 0.5 에 갇힌다(match 임계 0.6053 을 구조적으로 못 넘는다).

        판별 순서 - **Scope 가 1순위**다(2026-09-22 사용자 확인: rcp 뿐 아니라 msr cond 도
        Scope 를 갖고 OM / OMDF / SEM 을 정확히 구분한다). Scope 는 등록기가 적은 사실이고
        밝기 임계는 우리 가정이므로, 둘이 갈리면 Scope 가 이긴다.

          SEM        -> "SEM"   (dark 모드 없음)
          OMDF       -> "OM-D"  (암시야로 등록됨 - 명암 반전)
          OM         -> "OM"
          Scope 없음 -> ``!OM_Brightness`` 폴백 (< OM_DARK_BRIGHTNESS_MAX 이면 OM-D)

        Scope 도 밝기도 없으면 **None** 이다 - "모르면 OM" 으로 추측하면 반전 화면을
        정상이라 부르게 된다.

        반환값 문자열은 ``monitor/manual_image_mode_change.IMAGE_MODES`` 의 key 와 같다.
        """
        if self.is_sem:
            return "SEM"
        if self.is_om:
            return "OM-D" if self.is_om_dark else "OM"
        if self.scope:
            return None  # 아는 Scope 가 아니다 - 밝기로 덮어쓰지 않는다.
        # Scope 없는 cond (구형 msr 등) 폴백. 밝기 키가 없으면 OM 계열인지조차 모른다.
        brightness = self.om_brightness
        if brightness is None:
            return None
        return "OM-D" if brightness < OM_DARK_BRIGHTNESS_MAX else "OM"


def _norm_key(key: str) -> str:
    """비교용 키 정규화: 앞의 '!' 제거 + 소문자."""
    return key.lstrip("!").strip().lower()


def _to_int(token: str) -> int | None:
    """토큰을 int 로. 실패하면 None."""
    try:
        return int(token.strip())
    except (ValueError, AttributeError):
        return None


def _to_float(token: str) -> float | None:
    """토큰을 float 로. 실패하면 None (회전각은 '0.00' 처럼 소수로 온다)."""
    try:
        return float(token.strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _present(tokens: list[str], idx: tuple[int, ...]) -> bool:
    """주어진 인덱스 값들이 모두 존재하고 -1 이 아니면 True."""
    if max(idx) >= len(tokens):
        return False
    return all(_to_int(tokens[i]) not in (None, -1) for i in idx)


def parse_cond(text: str) -> CondInfo:
    """cond.txt 본문 문자열을 CondInfo 로 파싱한다."""
    raw: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # key 는 첫 공백/탭 이전 토큰, 값은 나머지를 콤마로 분해.
        parts = line.split(None, 1)
        if not parts:
            continue
        key = parts[0]
        value_str = parts[1].strip() if len(parts) > 1 else ""
        raw[_norm_key(key)] = [t.strip() for t in value_str.split(",")] if value_str else []

    scope = raw.get("scope", [None])[0]

    pixel = None
    px = raw.get("pixel", [])
    if len(px) >= 2 and _to_int(px[0]) is not None and _to_int(px[1]) is not None:
        pixel = (_to_int(px[0]), _to_int(px[1]))

    # 실데이터 키는 "!Cursor_inf"(끝 o 없음)·"!Cursor_info" 등 흔들린다 → 접두 매칭.
    cur = next((v for k, v in raw.items() if k.startswith("cursor_inf")), [])
    box_ltrb = None
    if _present(cur, _BOX_IDX):
        box_ltrb = tuple(_to_int(cur[i]) for i in _BOX_IDX)
    crosshair_xy = None
    if _present(cur, _CROSSHAIR_IDX):
        crosshair_xy = tuple(_to_int(cur[i]) for i in _CROSSHAIR_IDX)

    return CondInfo(
        scope=scope,
        pixel=pixel,
        box_ltrb=box_ltrb,
        crosshair_xy=crosshair_xy,
        raw=raw,
    )


def cond_for_image(cond: "CondInfo | None", shape_hw) -> "CondInfo | None":
    """cursor 좌표를 *로드된 이미지 크기* 에 맞춘 CondInfo 사본을 돌려준다.

    cursor 프레임은 ``Pixel × 10`` 기준이므로, 로드된 이미지가 cond.pixel 과 다른
    해상도면(리사이즈 저장 등) 고정 /10 변환이 좌표를 어긋나게 한다 — 그 오차는
    모든 프레임에 동일하게 걸려 blur 게이트로도 못 잡는 계통 오차가 된다. 여기서
    box/crosshair 를 loaded/pixel 비율로 축별 보정하고 pixel 을 로드 크기로 갱신해
    **멱등**으로 만든다(여러 레이어에서 겹쳐 불러도 이중 보정 없음). pixel 이 없거나
    0 이하, 또는 이미 로드 크기와 같으면 원본을 그대로 반환한다(기존 동작 불변).
    """
    if cond is None or cond.pixel is None:
        return cond
    pw, ph = cond.pixel
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if pw <= 0 or ph <= 0 or w <= 0 or h <= 0 or (pw == w and ph == h):
        return cond
    sx, sy = w / pw, h / ph
    print(
        f"[WARNING] cond.Pixel({pw}x{ph}) != 로드 이미지({w}x{h}) - "
        f"cursor 좌표를 x{sx:.3f}/x{sy:.3f} 보정"
    )
    box = cond.box_ltrb
    if box is not None:
        box = (
            int(round(box[0] * sx)), int(round(box[1] * sy)),
            int(round(box[2] * sx)), int(round(box[3] * sy)),
        )
    xh = cond.crosshair_xy
    if xh is not None:
        xh = (int(round(xh[0] * sx)), int(round(xh[1] * sy)))
    return replace(cond, pixel=(w, h), box_ltrb=box, crosshair_xy=xh)


def cond_path_for(image_path) -> Path:
    """이미지 경로 → 짝이 되는 cond.txt 경로 (.<파일명>/cond.txt)."""
    image_path = Path(image_path)
    return image_path.parent / f".{image_path.name}" / "cond.txt"


def load_cond(image_path) -> CondInfo | None:
    """이미지에 딸린 cond.txt 를 읽어 파싱한다. 없으면 None."""
    path = cond_path_for(image_path)
    if not path.is_file():
        return None
    return parse_cond(path.read_text(encoding="utf-8", errors="replace"))


# --- modality 추론 (공유) ---------------------------------------------------
# **2026-09-22 정정**: msr cond 에도 Scope 가 있고 OM/OMDF/SEM 을 정확히 가른다(사용자
# 확인). 종전 주석의 "msr 에는 Scope 가 없다"(2026-06-08)는 틀렸다. 그래서 Scope 가
# 1순위이고, 아래 키/배율 휴리스틱은 Scope 없는 cond 를 위한 폴백으로 남는다:
# OM = !OM_Brightness 키 + Magnification<200, SEM = Accelerating_voltage 키 + Magnification>500.
# 키 존재가 배율보다 우선([[project_align_cond_files_and_coords]]).
# 반환은 계속 'om' | 'sem' 2값이다 - OMDF 는 OM 과 같은 key 계열(IMAP0001)이라 라우팅·풀링이
# 같다. "어느 화면 모드로 찍혔나"는 CondInfo.required_image_mode 의 일이다.
# 두 eval(consensus·localization)이 같은 추론을 써야 해서 여기(공유 모듈)에 둔다 —
# consensus eval 이 localization eval 을 import 하므로 역방향 import 는 순환이 된다.
MSR_OM_MAG_MAX = 200     # Magnification < 이값 → OM (보조 신호).
MSR_SEM_MAG_MIN = 500    # Magnification > 이값 → SEM (보조 신호).


def msr_modality(cond: "CondInfo | None") -> str | None:
    """msr cond 의 modality 추론 'om' | 'sem' | None (Scope 없음 → 키/배율).

    Scope 가 있으면 그것이 1순위다(OM/OMDF → om, SEM → sem). 없을 때만 키 폴백:
    ``!OM_Brightness`` 키 → om, ``Accelerating_voltage`` 키 → sem (키 존재가 확정).
    키도 없으면 Magnification 보조: <MSR_OM_MAG_MAX → om, >MSR_SEM_MAG_MIN →
    sem, 그 사이(또는 미상)는 None(모호). raw 키는 parse 시 '!'·소문자화됨.
    """
    if cond is None:
        return None
    if cond.is_sem:
        return "sem"
    if cond.is_om:
        return "om"
    raw = cond.raw or {}
    if "accelerating_voltage" in raw:
        return "sem"
    if "om_brightness" in raw:
        return "om"
    mag_tokens = raw.get("magnification") or []
    mag = _to_int(mag_tokens[0]) if mag_tokens else None
    if mag is not None:
        if mag < MSR_OM_MAG_MAX:
            return "om"
        if mag > MSR_SEM_MAG_MIN:
            return "sem"
    return None
