"""Excel 캡처 핸들러 — 시트별로 PDF export 후 PyMuPDF 로 JPEG.

시트 하나가 가로/세로로 넓어 여러 페이지에 걸쳐 인쇄되면 그대로 보존한다
(예: 시트 3개 × 시트별 2페이지 → 총 6개 JPEG, page_001..page_006).
PageSetup 의 Zoom 을 False, FitToPagesWide=1 로 설정해 가로 정렬을 우선한다.
"""

import tempfile
from pathlib import Path
from typing import Iterator

from pipeline.settings import JPEG_QUALITY, NATIVE_RENDER_DPI
from pipeline.capture.common import is_windows, render_pdf_to_jpeg_pages


# Excel ExportAsFixedFormat 의 Type 상수: 0 = xlTypePDF
XL_TYPE_PDF: int = 0


def _import_com():
    """pywin32 lazy import."""
    if not is_windows():
        raise ImportError("Excel 캡처는 Windows 에서만 동작한다.")
    try:
        import pythoncom  # noqa: F401
        from win32com import client  # type: ignore
    except ImportError as exc:  # pragma: no cover - Windows 전용
        raise ImportError(
            "pywin32 가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client


def iter_pages(source_path: Path) -> Iterator[tuple[int, bytes, int, int]]:
    """Excel 워크북의 visible 시트를 PDF 로 export 한 뒤 페이지별 JPEG 를 yield."""
    if not source_path.exists():
        raise FileNotFoundError(f"Excel 파일을 찾을 수 없다: {source_path}")

    client = _import_com()
    import pythoncom

    pythoncom.CoInitialize()
    tmp_dir = Path(tempfile.mkdtemp(prefix="excel_export_"))
    try:
        app = client.Dispatch("Excel.Application")
        app.Visible = False
        app.DisplayAlerts = False
        workbook = app.Workbooks.Open(
            str(source_path.resolve()),
            UpdateLinks=0,
            ReadOnly=True,
            AddToMru=False,
        )

        global_page_index = 0
        try:
            for sheet in workbook.Worksheets:
                # Visible 상태 (-1 = xlSheetVisible) 가 아닌 시트는 건너뛴다.
                try:
                    if int(sheet.Visible) != -1:
                        continue
                except Exception:
                    pass

                tmp_pdf = tmp_dir / f"{sheet.Name}.pdf"
                try:
                    sheet.PageSetup.Zoom = False
                    sheet.PageSetup.FitToPagesWide = 1
                    sheet.PageSetup.FitToPagesTall = False
                except Exception:
                    # PageSetup 실패해도 export 자체는 시도해 본다.
                    print(
                        f"[WARNING] PageSetup 조정 실패 (sheet={sheet.Name}); "
                        "원본 설정으로 export 한다."
                    )

                sheet.ExportAsFixedFormat(XL_TYPE_PDF, str(tmp_pdf))
                if not tmp_pdf.exists():
                    print(
                        f"[WARNING] 시트 PDF export 실패: sheet={sheet.Name}; 건너뜀"
                    )
                    continue

                for _local_index, jpeg_bytes, width, height in render_pdf_to_jpeg_pages(
                    tmp_pdf,
                    dpi=NATIVE_RENDER_DPI,
                    jpeg_quality=JPEG_QUALITY,
                ):
                    global_page_index += 1
                    yield global_page_index, jpeg_bytes, width, height
                try:
                    tmp_pdf.unlink()
                except OSError:
                    pass
        finally:
            workbook.Close(SaveChanges=False)
        app.Quit()
    finally:
        try:
            for leftover in tmp_dir.iterdir():
                leftover.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
        pythoncom.CoUninitialize()
