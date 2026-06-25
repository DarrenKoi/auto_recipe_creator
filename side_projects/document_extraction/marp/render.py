"""Stage 6: Marp deck(.md) 렌더 (marp-cli) (marp_roundtrip_design.md Stage 6).

deck.md 를 marp-cli 로 PNG(슬라이드별 이미지)/PPTX/PDF/HTML 로 렌더한다. PNG 는
Stage 7 의 SSIM 검증 입력(원본 캡처 vs 재렌더 이미지)으로 쓴다.

설계: 인자 빌더(build_render_args)는 순수라 집에서 검증되고, 실제 렌더(render_deck)
는 marp-cli 가 필요하다. marp 바이너리가 없으면 graceful degrade(available=False,
예외 없음) — office 에서 `marp` 설치 또는 `npx @marp-team/marp-cli` 로 돈다.
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# fmt -> (marp-cli 플래그, 출력 확장자). png 는 --images 로 슬라이드별 이미지.
_FORMAT_FLAGS = {
    "png": ("--images", ".png"),
    "pptx": ("--pptx", ".pptx"),
    "pdf": ("--pdf", ".pdf"),
    "html": ("--html", ".html"),
}


def build_render_args(deck_path, out_dir, fmt):
    """marp-cli 인자 리스트(순수). base 명령(marp / npx ...) 뒤에 붙일 부분만.

    png 은 `--images png` 로 슬라이드별 PNG(<stem>.001.png ...) 를 out_dir 에 쓴다.
    pptx/pdf/html 은 단일 파일. 알 수 없는 fmt 는 ValueError.
    """
    if fmt not in _FORMAT_FLAGS:
        raise ValueError(f"지원하지 않는 렌더 포맷: {fmt} (지원: {sorted(_FORMAT_FLAGS)})")
    deck_path = Path(deck_path)
    out_dir = Path(out_dir)
    flag, ext = _FORMAT_FLAGS[fmt]
    out_path = out_dir / (deck_path.stem + ext)
    args = [str(deck_path)]
    if fmt == "png":
        args += [flag, "png"]   # --images png
    else:
        args += [flag]
    args += ["-o", str(out_path)]
    return args


def resolve_marp_command():
    """marp-cli 호출 base 명령을 해석한다(I/O). PATH 의 `marp` 우선, 없으면
    `npx --yes @marp-team/marp-cli`, 둘 다 없으면 None(graceful degrade)."""
    if shutil.which("marp"):
        return ["marp"]
    if shutil.which("npx"):
        return ["npx", "--yes", "@marp-team/marp-cli"]
    return None


@dataclass
class RenderResult:
    """렌더 결과. available=marp 사용 가능 여부, ok=렌더 성공, outputs=생성 파일들."""

    available: bool
    ok: bool = False
    fmt: str = "png"
    outputs: list = field(default_factory=list)
    stderr: str = ""


def render_deck(deck_path, out_dir, *, fmt="png", marp_cmd=None, timeout=180):
    """deck.md 를 marp-cli 로 렌더한다(I/O). marp 부재 시 available=False 로 graceful.

    marp_cmd: base 명령 override(테스트/오프라인용). None 이면 resolve_marp_command.
    성공 시 out_dir 에서 stem.* 산출물을 수집해 outputs 에 담는다.
    """
    deck_path = Path(deck_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = marp_cmd or resolve_marp_command()
    if not base:
        print("[WARNING] marp-cli 를 찾을 수 없습니다 (marp/npx 부재). 렌더 건너뜀.")
        return RenderResult(available=False, ok=False, fmt=fmt)

    cmd = list(base) + build_render_args(deck_path, out_dir, fmt)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:   # base 명령 자체가 실행 불가.
        print(f"[WARNING] marp 실행 실패(FileNotFoundError): {exc}. 렌더 건너뜀.")
        return RenderResult(available=False, ok=False, fmt=fmt)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] marp 렌더 timeout({timeout}s).")
        return RenderResult(available=True, ok=False, fmt=fmt, stderr="timeout")

    ok = proc.returncode == 0
    # 생성 산출물만 수집(소스 deck.md 등 비-산출물 제외): 포맷 확장자로 필터.
    _ext = _FORMAT_FLAGS[fmt][1]   # 예: ".png"
    outputs = (
        sorted(str(p) for p in out_dir.glob(deck_path.stem + ".*")
               if p.suffix.lower() == _ext)
        if ok else []
    )
    if ok:
        print(f"[INFO] marp 렌더 완료: {len(outputs)} 산출물 -> {out_dir}")
    else:
        print(f"[ERROR] marp 렌더 실패(rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    return RenderResult(available=True, ok=ok, fmt=fmt, outputs=outputs,
                        stderr=proc.stderr)


__all__ = [
    "RenderResult",
    "build_render_args",
    "render_deck",
    "resolve_marp_command",
]
