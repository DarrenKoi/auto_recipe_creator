"""PDF 추출 핸들러 — PyMuPDF로 페이지별 JPEG 렌더.

Acrobat/Edge 같은 외부 뷰어를 띄우지 않고 fitz로 직접 비트맵을 만든다.
크로스 플랫폼(Windows/macOS/Linux) 모두 동작.
"""

from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_jpegs


RENDER_DPI = 200
JPEG_QUALITY = 92


def extract(source: Path, out_dir: Path) -> int:
    """PDF 페이지를 JPEG로 추출한다. 저장된 페이지 수를 반환."""
    if not source.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없다: {source}")

    print(f"[INFO] PDF 추출 시작: {source.name}")
    next_index = render_pdf_to_jpegs(
        source,
        out_dir,
        start_index=1,
        dpi=RENDER_DPI,
        quality=JPEG_QUALITY,
    )
    page_count = next_index - 1
    print(f"[INFO] PDF 추출 완료: {source.name} ({page_count}페이지)")
    return page_count
