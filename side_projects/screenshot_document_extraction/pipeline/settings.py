"""파이프라인 공통 경로와 튜닝 상수.

CLAUDE.md 규약에 따라 `argparse` 또는 CLI 플래그를 쓰지 않는다.
변경이 필요하면 이 파일의 상수를 직접 수정하거나
환경변수(`SCREENSHOT_EXTRACTION_*`)로 덮어쓴다.
"""

import os
from pathlib import Path


PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_DIR / "data"
INPUTS_DIR: Path = DATA_DIR / "inputs"
CAPTURES_DIR: Path = DATA_DIR / "captures"
EXTRACTED_DIR: Path = DATA_DIR / "extracted"
ORGANIZED_DIR: Path = DATA_DIR / "organized"
LOGS_DIR: Path = PROJECT_DIR / "logs"
STATE_LEDGER_PATH: Path = LOGS_DIR / "pipeline_state.json"
PIPELINE_LOG_NAME: str = "screenshot_extraction"

# 캡처 단계 설정
JPEG_QUALITY: int = int(os.environ.get("SCREENSHOT_EXTRACTION_JPEG_QUALITY", "92"))
NATIVE_RENDER_DPI: int = int(os.environ.get("SCREENSHOT_EXTRACTION_RENDER_DPI", "200"))
# PowerPoint Slide.Export 의 픽셀 폭/높이. 1280x720 정도면 16:9 슬라이드에 충분.
POWERPOINT_EXPORT_WIDTH: int = 1280
POWERPOINT_EXPORT_HEIGHT: int = 720

# 추출 단계 설정
WEBP_QUALITY: int = 90
PADDLEOCR_SERVICE_SLUG: str = "paddleocr-vl-1.5"
UI_VENUS_SERVICE_SLUG: str = "ui-venus"
VLM_TIMEOUT_SEC: float = 180.0

# 안전 모드: true 면 실제 Office 프로세스 실행이나 VLM 호출을 막고 dry-run 한다.
SAFE_MODE: bool = (
    os.environ.get("SCREENSHOT_EXTRACTION_SAFE_MODE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# 실패한 doc 을 재시도할지 여부. false 면 startup 시 [WARNING] 만 찍고 skip.
RETRY_FAILED: bool = (
    os.environ.get("SCREENSHOT_EXTRACTION_RETRY_FAILED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# 입력 확장자 → source_type 매핑
EXTENSION_TO_SOURCE_TYPE: dict[str, str] = {
    ".pptx": "powerpoint",
    ".ppt": "powerpoint",
    ".docx": "word",
    ".doc": "word",
    ".xlsx": "excel",
    ".xls": "excel",
    ".pdf": "pdf",
}


def ensure_directories() -> None:
    """파이프라인이 사용하는 모든 출력 디렉터리를 미리 만든다."""
    for path in (INPUTS_DIR, CAPTURES_DIR, EXTRACTED_DIR, ORGANIZED_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
