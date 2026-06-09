"""
Workflow 3 package — 실시간 Align Fail 모니터링 시스템.

workflow_1(RCS GUI 자동화)과 workflow_2(CV align-key 보정)의 production 경로를
전면 이전해 하나의 end-to-end 루프로 통합한 패키지:

  알람 감지(ALID=9006) → RCS 장비 접속 → CV align fail 처리(보정)
  → 실패 시 cube rich notification → 상시 screenshot 녹화 → tool 닫기 → 다음 장비 대기

서브패키지 의존 방향(역방향 금지):
  monitor → {rcs, vision, runner, vlm, util}
  vision  → {vlm, util}
  rcs     → {vlm, util, runner}

workflow_3 는 poc.workflow_1 / poc.workflow_2 를 import 하지 않는다
(legacy 가 workflow_3 를 import 하는 방향만 허용).
"""

import os
import re
import time
from pathlib import Path


def _enable_dpi_awareness() -> None:
    """프로세스를 Per-Monitor DPI aware 로 설정한다(Windows 전용, 실패 시 무시).

    pywinauto(rectangle)·mss(capture)·pynput(click)이 모두 물리 픽셀 좌표계로
    일치하도록 만든다. 설정하지 않으면 DPI 배율(125/150%) 화면에서 캡처(물리)와
    창 좌표(논리)가 어긋나 클릭이 빗나간다(검출 overlay 는 멀쩡해 보여도 클릭은
    엉뚱한 위치). pywinauto 가 import 시 자체로 awareness 를 설정하므로, 그 전에
    (=패키지 최초 import 시점) 호출해야 우리 설정이 적용된다.
    """
    import ctypes

    try:  # PER_MONITOR_AWARE_V2 (Win10 1703+) — 가장 정확.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:  # PROCESS_PER_MONITOR_DPI_AWARE (Win8.1+).
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:  # System DPI aware (구형 폴백).
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_enable_dpi_awareness()

WORKFLOW_3_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW_3_DIR / "debug_images"
LOG_DIR = WORKFLOW_3_DIR / "logs"
TEMPLATES_DIR = WORKFLOW_3_DIR / "templates"

# 오피스 MES 가 align fail 시 생성하는 이미지 루트. align fail 핸들러는 여기에
# captured_img_from_rcs 를 함께 적재하고, vision.align_fail_assets 가 읽는다.
#   align_images/<eqp_id>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr, captured_img_from_rcs}
# 물리 경로는 오피스 MES 도구가 직접 타겟하므로 workflow_1 시절 위치를 그대로 쓴다.
# 옮겨야 할 때는 env ALIGN_IMAGES_DIR 한 줄로 전환한다.
_align_images_env = os.environ.get("ALIGN_IMAGES_DIR", "").strip()
ALIGN_IMAGES_DIR = (
    Path(_align_images_env)
    if _align_images_env
    else WORKFLOW_3_DIR.parent / "workflow_1" / "align_images"
)

# consensus S-image gather 캐시 루트. MES 산출물(align_images)이 아니라 우리가 만드는
# 파생 캐시라 위치 자유 — workflow_3 아래 둔다. env 로 override 가능.
_consensus_cache_env = os.environ.get("ALIGN_CONSENSUS_CACHE_DIR", "").strip()
ALIGN_CONSENSUS_CACHE_DIR = (
    Path(_consensus_cache_env)
    if _consensus_cache_env
    else WORKFLOW_3_DIR / "align_consensus_cache"
)

_TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{6}_\d{6}_")


def _slugify_model_name(model_name: str) -> str:
    """모델명을 폴더명에 안전한 slug 로 변환한다."""
    safe_chars = []
    for char in model_name.strip().lower():
        if char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")

    slug = "".join(safe_chars).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default-model"


def resolve_debug_model_name(model_name: str | None = None) -> str:
    """디버그 이미지 저장에 사용할 모델명을 결정한다."""
    from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_MODEL_NAME

    resolved = (
        (model_name or "").strip()
        or os.environ.get("VLM_MODEL_NAME", "").strip()
        or DEFAULT_SCREEN_ANALYSIS_MODEL_NAME
    )
    return _slugify_model_name(resolved or "default-model")


def debug_image_dir(debug_root: Path, model_name: str | None = None) -> Path:
    """모델명 기준 디버그 이미지 하위 디렉터리를 반환한다."""
    out_dir = debug_root / resolve_debug_model_name(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def debug_image_path(
    debug_root: Path,
    filename: str,
    model_name: str | None = None,
    timestamp_tag: str | None = None,
    now: float | None = None,
) -> Path:
    """모델명 하위 폴더를 포함한 디버그 이미지 파일 경로를 반환한다."""
    relative_path = Path(filename)
    resolved_name = relative_path.name
    if not _TIMESTAMP_PREFIX_PATTERN.match(resolved_name):
        if timestamp_tag:
            resolved_tag = str(timestamp_tag).strip()
        else:
            resolved_now = time.time() if now is None else now
            resolved_tag = time.strftime("%y%m%d_%H%M%S", time.localtime(resolved_now))
        resolved_name = f"{resolved_tag}_{resolved_name}"

    return (
        debug_image_dir(debug_root, model_name=model_name)
        / relative_path.with_name(resolved_name)
    )


__all__ = [
    "ALIGN_CONSENSUS_CACHE_DIR",
    "ALIGN_IMAGES_DIR",
    "DEBUG_IMAGE_DIR",
    "LOG_DIR",
    "TEMPLATES_DIR",
    "WORKFLOW_3_DIR",
    "debug_image_dir",
    "debug_image_path",
    "resolve_debug_model_name",
]
