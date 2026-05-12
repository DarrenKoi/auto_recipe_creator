"""Word 캡처 핸들러 — COM 으로 PDF export 후 PyMuPDF 로 페이지별 JPEG."""

import tempfile
from pathlib import Path
from typing import Iterator

from pipeline.settings import JPEG_QUALITY, NATIVE_RENDER_DPI
from pipeline.capture.common import is_windows, render_pdf_to_jpeg_pages


# Word ExportAsFixedFormat 의 ExportFormat 상수: 17 = wdExportFormatPDF
WD_EXPORT_FORMAT_PDF: int = 17


def _import_com():
    """pywin32 lazy import."""
    if not is_windows():
        raise ImportError("Word 캡처는 Windows 에서만 동작한다.")
    try:
        import pythoncom  # noqa: F401
        from win32com import client  # type: ignore
    except ImportError as exc:  # pragma: no cover - Windows 전용
        raise ImportError(
            "pywin32 가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client


def iter_pages(source_path: Path) -> Iterator[tuple[int, bytes, int, int]]:
    """Word 문서를 PDF 로 export 한 뒤 페이지별 JPEG 를 yield."""
    if not source_path.exists():
        raise FileNotFoundError(f"Word 파일을 찾을 수 없다: {source_path}")

    client = _import_com()
    import pythoncom

    pythoncom.CoInitialize()
    tmp_dir = Path(tempfile.mkdtemp(prefix="word_export_"))
    tmp_pdf = tmp_dir / "exported.pdf"
    try:
        app = client.Dispatch("Word.Application")
        app.Visible = False
        document = app.Documents.Open(
            str(source_path.resolve()),
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
        yield from render_pdf_to_jpeg_pages(
            tmp_pdf,
            dpi=NATIVE_RENDER_DPI,
            jpeg_quality=JPEG_QUALITY,
        )
    finally:
        try:
            tmp_pdf.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
        pythoncom.CoUninitialize()
