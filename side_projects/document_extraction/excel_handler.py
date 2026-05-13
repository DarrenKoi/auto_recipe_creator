"""Excel 추출 핸들러 — xlwings로 시트별 인쇄 페이지 단위 추출.

각 시트를 Excel의 ExportAsFixedFormat으로 PDF로 export 한 뒤,
PyMuPDF로 페이지별 JPEG 렌더한다. 한 시트가 인쇄 시 여러 페이지로
나뉘면 그대로 여러 JPEG가 생성된다.
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_jpegs


RENDER_DPI = 200
JPEG_QUALITY = 92


def _import_xlwings():
    try:
        import xlwings as xw
    except ImportError as exc:
        raise ImportError(
            "xlwings가 필요합니다. `uv pip install xlwings` 로 설치하세요."
        ) from exc
    return xw


def extract(source: Path, out_dir: Path) -> int:
    """Excel 워크북을 열고 시트별로 PDF export → JPEG 분할."""
    if not source.exists():
        raise FileNotFoundError(f"Excel 파일을 찾을 수 없다: {source}")

    xw = _import_xlwings()
    print(f"[INFO] Excel 추출 시작: {source.name}")

    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False
    try:
        # 프린터 통신 비활성화로 ExportAsFixedFormat 속도 향상
        try:
            app.api.PrintCommunication = False
        except Exception:
            pass

        wb = app.books.open(
            str(source.resolve()),
            read_only=True,
            update_links=False,
        )
        page_index = 1
        try:
            for sheet in wb.sheets:
                try:
                    used = sheet.used_range
                    if used is None or used.shape == (1, 1) and used.value in (None, ""):
                        print(f"[INFO]   - 시트 스킵(빈 시트): {sheet.name}")
                        continue
                except Exception:
                    pass

                with tempfile.TemporaryDirectory(prefix="xlw_pdf_") as tmpdir:
                    tmp_pdf = Path(tmpdir) / f"{sheet.name}.pdf"
                    # 0 = xlTypePDF
                    sheet.api.ExportAsFixedFormat(0, str(tmp_pdf))
                    if not tmp_pdf.exists():
                        print(f"[WARNING]   - 시트 export 실패: {sheet.name}")
                        continue
                    before = page_index
                    page_index = render_pdf_to_jpegs(
                        tmp_pdf,
                        out_dir,
                        start_index=page_index,
                        dpi=RENDER_DPI,
                        quality=JPEG_QUALITY,
                    )
                    print(
                        f"[INFO]   - 시트 처리: {sheet.name} "
                        f"({page_index - before}페이지)"
                    )
        finally:
            wb.close()
    finally:
        try:
            app.api.PrintCommunication = True
        except Exception:
            pass
        app.quit()

    page_count = page_index - 1
    print(f"[INFO] Excel 추출 완료: {source.name} ({page_count}페이지)")
    return page_count
