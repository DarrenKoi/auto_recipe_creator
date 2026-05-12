"""PDF 캡처 핸들러 — PyMuPDF 만 사용한다 (Acrobat 미사용)."""

from pathlib import Path
from typing import Iterator

from pipeline.settings import NATIVE_RENDER_DPI, JPEG_QUALITY
from pipeline.capture.common import render_pdf_to_jpeg_pages


def iter_pages(source_path: Path) -> Iterator[tuple[int, bytes, int, int]]:
    """PDF 페이지마다 (page_index, jpeg_bytes, width, height) 를 yield."""
    if not source_path.exists():
        raise FileNotFoundError(f"PDF 를 찾을 수 없다: {source_path}")
    yield from render_pdf_to_jpeg_pages(
        source_path,
        dpi=NATIVE_RENDER_DPI,
        jpeg_quality=JPEG_QUALITY,
    )
