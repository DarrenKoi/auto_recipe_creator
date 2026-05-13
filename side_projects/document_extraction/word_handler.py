"""Word 추출 핸들러 — COM으로 PDF export 후 PyMuPDF로 페이지별 JPEG.

Word에는 슬라이드쇼 같은 개념이 없으므로 PDF 변환 경유가 자연스럽다.
ExportAsFixedFormat(ExportFormat=17 = wdExportFormatPDF) 사용.
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_jpegs


RENDER_DPI = 200
JPEG_QUALITY = 92

WD_EXPORT_FORMAT_PDF = 17


def _import_com():
    try:
        import pythoncom  # noqa: F401
        from win32com import client
    except ImportError as exc:
        raise ImportError(
            "pywin32가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client


def extract(source: Path, out_dir: Path) -> int:
    """Word 문서를 PDF로 export 한 뒤 페이지별 JPEG로 저장."""
    if not source.exists():
        raise FileNotFoundError(f"Word 파일을 찾을 수 없다: {source}")

    client = _import_com()
    import pythoncom

    print(f"[INFO] Word 추출 시작: {source.name}")

    pythoncom.CoInitialize()
    tmp_dir = Path(tempfile.mkdtemp(prefix="word_export_"))
    tmp_pdf = tmp_dir / "exported.pdf"
    try:
        app = client.Dispatch("Word.Application")
        app.Visible = False
        document = app.Documents.Open(
            str(source.resolve()),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
        )
        try:
            document.ExportAsFixedFormat(
                OutputFileName=str(tmp_pdf),
                ExportFormat=WD_EXPORT_FORMAT_PDF,
            )
        finally:
            document.Close(SaveChanges=False)
        app.Quit()

        if not tmp_pdf.exists():
            raise RuntimeError(f"Word PDF export 실패: {tmp_pdf}")

        next_index = render_pdf_to_jpegs(
            tmp_pdf,
            out_dir,
            start_index=1,
            dpi=RENDER_DPI,
            quality=JPEG_QUALITY,
        )
        page_count = next_index - 1
    finally:
        try:
            tmp_pdf.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
        pythoncom.CoUninitialize()

    print(f"[INFO] Word 추출 완료: {source.name} ({page_count}페이지)")
    return page_count
