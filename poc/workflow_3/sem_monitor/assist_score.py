"""Recipe Monitor Assist Window 의 score 색상을 읽어 측정 성부를 판정한다.

Assist Window 는 tool 창 내부 패널이다(별도 최상위 창이 아님). 3열(Addressing1 /
Addressing2 / Measurement) x 7행으로 최신 7회 측정의 썸네일과 score 가 실시간으로 쌓이고,
**score 숫자의 색이 곧 성부다** - 검정이면 정상 측정, 빨강이면 측정 실패. 측정이 진행
중인 행은 빈칸이다. 최신 행은 맨 아래에 쌓인다.

설계 경계(프로젝트 규칙): VLM 은 패널 *영역*만 1회 식별하고, 색 판정은 전부 CV 가 한다.
우리가 필요한 건 값이 아니라 색이므로 폴링마다 OCR 을 돌리지 않는다 - 읽지 않는 정보는
틀릴 수 없다.
"""

import numpy as np
from dataclasses import dataclass, field

# 표 형태.
ASSIST_ROWS = 7
ASSIST_NEWEST_ROW_AT = "bottom"  # tool 버전이 다르면 "top".
ASSIST_COLUMNS = ("Addressing1", "Addressing2", "Measurement")

# 색 분류 임계. 배경은 밝고 잉크(글자)는 어둡다는 전제.
INK_MEAN_MAX = 200      # 채널 평균이 이보다 어두우면 잉크로 본다.
INK_MIN_PIXELS = 6      # 잉크가 이보다 적으면 빈칸(안티에일리어싱 무시).
RED_CHROMA_MIN = 60     # max-min 이 이 이상이면 유채색.
RED_DOMINANCE_MIN = 40  # R - max(G,B) 가 이 이상이면 붉은 계열.
RED_RATIO_MIN = 0.30    # 잉크 중 빨강 비율이 이 이상이면 red.
BLACK_RATIO_MAX = 0.10  # 이 이하면 black. 사이는 unknown.


def classify_ink(cell_rgb: np.ndarray) -> str:
    """셀 하나의 잉크 색을 판정한다. "black"|"red"|"blank"|"unknown".

    입력은 RGB numpy 배열이다(PIL Image 를 np.array 로 바꾼 형태). 흑/적 비율이 어느
    쪽으로도 확실하지 않으면 "unknown" 을 돌려준다 - 호출부가 streak 을 끊게 해서
    애매함이 done 판정으로 새지 않게 한다.
    """
    if cell_rgb is None or cell_rgb.size == 0:
        return "blank"
    arr = cell_rgb.astype(np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return "blank"

    red_c = arr[:, :, 0]
    green_c = arr[:, :, 1]
    blue_c = arr[:, :, 2]
    mean = arr[:, :, :3].mean(axis=2)
    ink = mean < INK_MEAN_MAX
    ink_count = int(ink.sum())
    if ink_count < INK_MIN_PIXELS:
        return "blank"

    chroma = arr[:, :, :3].max(axis=2) - arr[:, :, :3].min(axis=2)
    dominance = red_c - np.maximum(green_c, blue_c)
    is_red = ink & (chroma >= RED_CHROMA_MIN) & (dominance >= RED_DOMINANCE_MIN)
    red_ratio = float(is_red.sum()) / float(ink_count)

    if red_ratio >= RED_RATIO_MIN:
        return "red"
    if red_ratio <= BLACK_RATIO_MAX:
        return "black"
    return "unknown"


# Addressing2 는 대개 비어 있어 판정에 쓰지 않는다.
ASSIST_CRITICAL_COLUMNS = ("Addressing1", "Measurement")


def row_verdict(cells: dict) -> str:
    """측정 1회(행 하나)의 성부. "ok"|"fail"|"pending"|"unknown".

    pending 을 **Measurement 기준으로만** 판정하는 이유: Addressing2 는 대개 비어 있고
    Addressing1 도 레시피에 따라 없을 수 있다. 없는 칸을 '진행 중' 으로 읽으면 그
    레시피는 영영 done 이 되지 않는다. Measurement 가 최종 결과이므로 그것으로 완료를
    판정하고, Addressing1 은 값이 있을 때만 실패 신호로 쓴다.
    """
    critical = [cells.get(name, "blank") for name in ASSIST_CRITICAL_COLUMNS]
    if any(state == "red" for state in critical):
        return "fail"
    if any(state == "unknown" for state in critical):
        return "unknown"
    if cells.get("Measurement", "blank") == "blank":
        return "pending"
    return "ok"


@dataclass
class RowState:
    """Assist Window 의 행 하나 - 열별 잉크 색과 그로부터 나온 성부."""

    cells: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return row_verdict(self.cells)


def ok_streak(rows: list) -> int:
    """최신 행부터 세어 연속 정상(ok) 개수. fail/unknown 을 만나면 멈춘다.

    최신 쪽의 pending(측정 진행 중)은 건너뛴다 - 아직 결과가 안 나온 행이 그 앞의 연속
    정상 기록을 지우면 안 된다. 목록은 index 0 이 가장 오래된 행이다.
    """
    idx = len(rows) - 1
    while idx >= 0 and rows[idx].verdict == "pending":
        idx -= 1
    streak = 0
    while idx >= 0 and rows[idx].verdict == "ok":
        streak += 1
        idx -= 1
    return streak


# 헤더 텍스트 매칭용 - 영숫자만 남기고 소문자 비교(OCR 공백/기호 흔들림 흡수).
def _normalize(text: str) -> str:
    """열 이름 비교용 정규화."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _is_score_text(text: str) -> bool:
    """score 로 볼 텍스트인지(숫자만)."""
    stripped = (text or "").strip()
    return bool(stripped) and all(ch.isdigit() for ch in stripped)


@dataclass
class AssistLayout:
    """Assist 패널의 score 셀 격자. 1회 만들어 캐시한다."""

    panel_box: dict
    grid: list        # grid[row][col] = {"left","top","right","bottom"} (패널 crop 좌표계)
    columns: tuple


def build_score_grid(items: list, panel_size: tuple, *, rows: int = ASSIST_ROWS):
    """OCR spotting 항목에서 score 셀 격자를 만든다. 실패 시 None.

    열은 헤더 텍스트로 잡는다(순서로 추정하지 않는다 - Addressing2 가 비어 있으면 숫자
    덩어리가 2개뿐이라 어느 것이 Measurement 인지 알 수 없다). 행은 숫자 항목의 y 중심을
    모아 띠(band) 간 간격의 중앙값으로 pitch 를 구한 뒤 rows 개로 외삽한다 - 중간 띠 하나가
    누락돼도 다수결로 버틴다.

    items 는 패널 crop 좌표계여야 한다. panel_size 는 (width, height).
    """
    if not items:
        return None

    # --- 열: 헤더 텍스트로 x 범위 확정 ---
    header_boxes = {}
    for item in items:
        name = _normalize(item.get("text", ""))
        for column in ASSIST_COLUMNS:
            if name == _normalize(column) and column not in header_boxes:
                header_boxes[column] = item.get("bbox") or {}
    if len(header_boxes) != len(ASSIST_COLUMNS):
        print(f"[WARNING] Assist 헤더 인식 부족({sorted(header_boxes)}) - 격자 생성 실패")
        return None

    # --- 행: 숫자 항목의 y 중심 -> pitch -> 외삽 ---
    header_bottom = max(int(box.get("bottom", 0)) for box in header_boxes.values())
    centers = []
    heights = []
    for item in items:
        if not _is_score_text(item.get("text", "")):
            continue
        box = item.get("bbox") or {}
        top = int(box.get("top", 0))
        bottom = int(box.get("bottom", 0))
        if bottom <= header_bottom:
            continue
        centers.append((top + bottom) / 2.0)
        heights.append(max(1, bottom - top))
    if not centers:
        print("[WARNING] Assist 숫자 항목 없음 - 격자 생성 실패")
        return None

    cell_h = int(round(sorted(heights)[len(heights) // 2]))
    band_centers = _cluster_1d(sorted(centers), tolerance=cell_h)
    if len(band_centers) < 2:
        print("[WARNING] Assist 행이 2개 미만 - pitch 를 알 수 없어 격자 생성 실패")
        return None

    gaps = sorted(band_centers[i + 1] - band_centers[i] for i in range(len(band_centers) - 1))
    pitch = gaps[len(gaps) // 2]
    if pitch <= 0:
        return None

    # 최신 행이 맨 아래이므로 가장 아래 띠를 마지막 행에 맞추고 위로 채운다.
    last_center = band_centers[-1]
    grid = []
    for row_idx in range(rows):
        center = last_center - pitch * (rows - 1 - row_idx)
        top = int(round(center - cell_h / 2.0))
        bottom = int(round(center + cell_h / 2.0))
        row_boxes = []
        for column in ASSIST_COLUMNS:
            box = header_boxes[column]
            row_boxes.append({
                "left": int(box.get("left", 0)),
                "right": int(box.get("right", 0)),
                "top": top,
                "bottom": bottom,
            })
        grid.append(row_boxes)

    return AssistLayout(
        panel_box={"left": 0, "top": 0, "right": panel_size[0], "bottom": panel_size[1]},
        grid=grid,
        columns=tuple(ASSIST_COLUMNS),
    )


def _cluster_1d(values: list, *, tolerance: int) -> list:
    """정렬된 1D 값들을 tolerance 이내로 묶어 각 묶음의 평균을 돌려준다."""
    if not values:
        return []
    clusters = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / float(len(group)) for group in clusters]


__all__ = [
    "ASSIST_COLUMNS",
    "ASSIST_CRITICAL_COLUMNS",
    "ASSIST_NEWEST_ROW_AT",
    "ASSIST_ROWS",
    "AssistLayout",
    "RowState",
    "build_score_grid",
    "classify_ink",
    "ok_streak",
    "row_verdict",
]
