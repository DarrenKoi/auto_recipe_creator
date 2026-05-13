"""출력 폴더/파일 경로 헬퍼."""

from pathlib import Path

from PIL import Image

from side_projects.document_extraction.util.screen_capture import save_webp_capped


def doc_output_dir(output_root: Path, source: Path) -> Path:
    """파일명(확장자 제외)을 딴 출력 폴더 경로를 반환한다."""
    return output_root / source.stem


def page_image_path(out_dir: Path, page_index: int) -> Path:
    """page_001.webp 형식의 출력 경로를 반환한다."""
    return out_dir / f"page_{page_index:03d}.webp"


def render_pdf_to_webps(
    pdf_path: Path,
    out_dir: Path,
    *,
    start_index: int = 1,
    dpi: int = 200,
) -> int:
    """PyMuPDF로 PDF를 페이지별 WebP(1MB 이하)로 저장하고 마지막 인덱스 다음 값을 반환한다.

    Word/Excel 핸들러가 시트/문서를 PDF로 export 한 뒤 호출.
    PDF 핸들러도 동일 함수를 재사용.

    pixmap → PIL.Image 로 한 번 변환한 뒤 `save_webp_capped` 으로 위임하여
    PPT 스크린 캡처 경로와 동일한 1MB 캡 로직을 공유한다.
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
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            out_path = page_image_path(out_dir, page_index)
            save_webp_capped(image, out_path)
            page_index += 1
    finally:
        doc.close()
    return page_index
