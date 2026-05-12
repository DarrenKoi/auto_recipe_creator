"""캡처 핸들러 공용 유틸리티."""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol

from pipeline.settings import CAPTURES_DIR, JPEG_QUALITY


@dataclass(frozen=True)
class PageArtifact:
    """캡처된 페이지 하나의 메타데이터."""

    page_index: int  # 1-based
    width: int
    height: int
    source_path: str
    source_type: str
    capture_method: str  # "native_export" | "screenshot"
    captured_at: str


class PageIterator(Protocol):
    """핸들러가 반환하는 페이지 iterator 형태.

    구현체는 다음을 yield 한다:
        (page_index, jpeg_bytes, width, height)
    """

    def __iter__(self) -> Iterator[tuple[int, bytes, int, int]]: ...


def is_windows() -> bool:
    """Windows 플랫폼 여부."""
    return sys.platform.startswith("win")


def doc_capture_dir(doc_id: str) -> Path:
    """doc 의 캡처 출력 디렉터리. 없으면 만든다."""
    out = CAPTURES_DIR / doc_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def page_jpeg_path(doc_id: str, page_index: int) -> Path:
    """페이지 JPEG 경로 (zero-padded 3자리)."""
    return doc_capture_dir(doc_id) / f"page_{page_index:03d}.jpg"


def page_meta_path(doc_id: str, page_index: int) -> Path:
    """페이지 메타 JSON 경로."""
    return doc_capture_dir(doc_id) / f"page_{page_index:03d}.meta.json"


def save_page_jpeg(jpeg_bytes: bytes, jpeg_path: Path) -> None:
    """JPEG bytes 를 파일로 저장한다.

    bytes 가 이미 JPEG 헤더로 시작하지 않으면 [WARNING] 만 찍고 그대로 쓴다
    (예: 핸들러가 PNG bytes 를 잘못 넘긴 경우 디버깅 단서가 된다).
    """
    if not jpeg_bytes:
        raise ValueError("빈 JPEG bytes 가 전달되었다.")
    if jpeg_bytes[:2] != b"\xff\xd8":
        print(f"[WARNING] JPEG 헤더가 아닌 bytes: path={jpeg_path}")
    jpeg_path.write_bytes(jpeg_bytes)


def save_page_meta(meta: PageArtifact, meta_path: Path) -> None:
    """페이지 메타 JSON 을 저장한다."""
    meta_path.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def now_iso() -> str:
    """타임존 없는 ISO-8601 시각."""
    return datetime.now().replace(microsecond=0).isoformat()


def render_pdf_to_jpeg_pages(
    pdf_path: Path,
    *,
    dpi: int,
    jpeg_quality: int = JPEG_QUALITY,
) -> Iterator[tuple[int, bytes, int, int]]:
    """PyMuPDF 로 PDF 를 페이지별 JPEG bytes 로 렌더한다.

    Word/Excel 핸들러는 먼저 PDF 로 export 한 뒤 이 함수를 통해 JPEG 를 얻는다.
    PDF 핸들러도 동일한 함수를 재사용한다.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 외부 의존성
        raise ImportError(
            "PyMuPDF (pymupdf) 가 필요합니다. `uv pip install pymupdf` 로 설치하세요."
        ) from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(str(pdf_path))
    try:
        for index, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            jpeg_bytes = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
            yield index, jpeg_bytes, pixmap.width, pixmap.height
    finally:
        doc.close()
