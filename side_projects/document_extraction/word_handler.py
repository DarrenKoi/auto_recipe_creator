"""Word 추출 핸들러 — COM으로 PDF export 후 PyMuPDF로 페이지별 WebP.

Word에는 슬라이드쇼 같은 개념이 없으므로 PDF 변환 경유가 자연스럽다.
ExportAsFixedFormat(ExportFormat=17 = wdExportFormatPDF) 사용.

DRM 보호 문서는 Word 가 열어서 "표시"는 해 주지만 export/변환을 차단하는 경우가
많다. 이때는 util/viewer_capture 폴백으로 문서를 기본 프로그램(Word)에서 열어
읽기 모드 전체화면으로 페이지별 화면 캡처한다.
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_webps


RENDER_DPI = 200

WD_EXPORT_FORMAT_PDF = 17

# DRM 뷰어 캡처 폴백 키(office 보정): Alt -> W -> F = 보기 리본 keytip 으로 읽기
# 모드 진입(화면 단위 페이지 넘김이 가능해짐). 한국어 Office 도 keytip 은 동일.
WORD_VIEWER_FULLSCREEN_KEYS = "%wf"
WORD_VIEWER_NEXT_KEY = "{PGDN}"


def _import_com():
    try:
        import pythoncom  # noqa: F401
        from win32com import client
    except ImportError as exc:
        raise ImportError(
            "pywin32가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client


def _extract_via_viewer(source: Path, out_dir: Path) -> int:
    """DRM Word 문서를 읽기 모드 화면 캡처로 추출한다(Windows 전용)."""
    from side_projects.document_extraction.util.viewer_capture import (
        capture_paged_viewer,
    )

    return capture_paged_viewer(
        source,
        out_dir,
        fullscreen_keys=WORD_VIEWER_FULLSCREEN_KEYS,
        next_key=WORD_VIEWER_NEXT_KEY,
    )


def extract(source: Path, out_dir: Path) -> int:
    """Word 문서를 PDF로 export 한 뒤 페이지별 WebP(1MB 이하)로 저장.

    export 가 실패하면(DRM 차단이 대표 케이스) 뷰어 화면 캡처로 폴백한다.
    """
    if not source.exists():
        raise FileNotFoundError(f"Word 파일을 찾을 수 없다: {source}")

    client = _import_com()
    import pythoncom

    print(f"[INFO] Word 추출 시작: {source.name}")

    pythoncom.CoInitialize()
    tmp_dir = Path(tempfile.mkdtemp(prefix="word_export_"))
    tmp_pdf = tmp_dir / "exported.pdf"
    export_error: Exception | None = None
    try:
        app = None
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
        except Exception as exc:
            export_error = exc
        finally:
            if app is not None:
                try:
                    app.Quit()
                except Exception:
                    pass

        if export_error is not None or not tmp_pdf.exists():
            reason = export_error if export_error is not None else "export 파일 미생성"
            print(f"[WARNING] Word PDF export 실패(DRM 차단 가능): {reason}")
            print(f"[WARNING] 뷰어 화면 캡처 폴백으로 전환: {source.name}")
            page_count = _extract_via_viewer(source, out_dir)
            print(
                f"[INFO] Word 추출 완료(뷰어 캡처): {source.name} ({page_count}페이지)"
            )
            return page_count

        next_index = render_pdf_to_webps(
            tmp_pdf,
            out_dir,
            start_index=1,
            dpi=RENDER_DPI,
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
