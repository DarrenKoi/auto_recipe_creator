"""출력 폴더/파일 경로 헬퍼."""

from pathlib import Path


def doc_output_dir(output_root: Path, source: Path) -> Path:
    """파일명(확장자 제외)을 딴 출력 폴더 경로를 반환한다."""
    return output_root / source.stem


def page_image_path(out_dir: Path, page_index: int) -> Path:
    """page_001.jpg 형식의 출력 경로를 반환한다."""
    return out_dir / f"page_{page_index:03d}.jpg"


def render_pdf_to_jpegs(pdf_path: Path, out_dir: Path, *, start_index: int = 1, dpi: int = 200, quality: int = 92) -> int:
    """PyMuPDF로 PDF를 페이지별 JPEG로 저장하고 마지막 인덱스 다음 값을 반환한다.

    Word/Excel 핸들러가 시트/문서를 PDF로 export 한 뒤 호출.
    PDF 핸들러도 동일 함수를 재사용.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (pymupdf) 가 필요합니다. `uv pip install pymupdf` 로 설치하세요."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(str(pdf_path))
    page_index = start_index
    try:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            jpeg_bytes = pixmap.tobytes("jpeg", jpg_quality=quality)
            out_path = page_image_path(out_dir, page_index)
            out_path.write_bytes(jpeg_bytes)
            page_index += 1
    finally:
        doc.close()
    return page_index
