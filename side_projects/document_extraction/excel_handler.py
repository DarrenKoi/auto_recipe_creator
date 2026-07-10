"""Excel 추출 핸들러 — xlwings로 시트별 인쇄 페이지 단위 추출.

각 시트를 Excel의 ExportAsFixedFormat으로 PDF로 export 한 뒤,
PyMuPDF로 페이지별 WebP(1MB 이하) 렌더한다. 한 시트가 인쇄 시 여러 페이지로
나뉘면 그대로 여러 WebP가 생성된다.
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_webps


RENDER_DPI = 200


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
        failed_sheets = 0
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
                    # 0 = xlTypePDF. DRM 이 export 를 차단하면 여기서 COM 예외가 난다
                    # -> 해당 시트만 스킵하고 계속(워크북 전체 중단 방지).
                    try:
                        sheet.api.ExportAsFixedFormat(0, str(tmp_pdf))
                    except Exception as exc:
                        print(
                            f"[WARNING]   - 시트 export 예외(DRM 차단 가능): "
                            f"{sheet.name}: {exc}"
                        )
                        failed_sheets += 1
                        continue
                    if not tmp_pdf.exists():
                        print(f"[WARNING]   - 시트 export 실패: {sheet.name}")
                        failed_sheets += 1
                        continue
                    before = page_index
                    page_index = render_pdf_to_webps(
                        tmp_pdf,
                        out_dir,
                        start_index=page_index,
                        dpi=RENDER_DPI,
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
    if page_count == 0 and failed_sheets > 0:
        # 스크롤형 시트는 PDF/Word 처럼 페이지 개념이 없어 범용 뷰어 캡처 폴백이
        # 불안정하다 -> 폴백 없이 명확히 안내(수동으로 인쇄 미리보기 캡처 권장).
        print(
            f"[WARNING] Excel 전 시트 export 실패({failed_sheets}건) - DRM 차단 가능. "
            "DRM Excel 은 수동 캡처(인쇄 미리보기)로 처리하세요."
        )
    print(f"[INFO] Excel 추출 완료: {source.name} ({page_count}페이지)")
    return page_count
