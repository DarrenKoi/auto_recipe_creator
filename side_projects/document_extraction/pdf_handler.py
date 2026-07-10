"""PDF 추출 핸들러 — PyMuPDF로 페이지별 WebP 렌더 (+ DRM 뷰어 캡처 폴백).

기본 경로는 Acrobat/Edge 같은 외부 뷰어를 띄우지 않고 fitz로 직접 비트맵을 만든다.
크로스 플랫폼(Windows/macOS/Linux) 모두 동작.

DRM/암호화 PDF 는 fitz 가 열지 못하거나(needs_pass/열기 실패) 열어도 렌더가 불가한데,
DRM 은 허가된 뷰어 안에서의 "표시"는 항상 허용하므로 Windows 에서는
util/viewer_capture 로 기본 뷰어를 열어 페이지별 화면 캡처 폴백을 시도한다.
"""

import os
from pathlib import Path

from side_projects.document_extraction.util.output_paths import render_pdf_to_webps


RENDER_DPI = 200

# DRM 뷰어 캡처 폴백 키(office 에서 기본 뷰어에 맞게 보정).
# Acrobat (Reader) 전체화면 = Ctrl+L. Edge PDF 뷰어가 기본이면 F11 등으로 조정.
PDF_VIEWER_FULLSCREEN_KEYS = "^l"
PDF_VIEWER_NEXT_KEY = "{PGDN}"


def _probe_unreadable(source: Path) -> str:
    """fitz 로 직접 렌더 불가한 PDF 인지 조사한다. 가능하면 "", 불가면 사유 문자열.

    DRM 래핑 파일은 대개 fitz.open 자체가 실패하고, 암호화 PDF 는 needs_pass 로 잡힌다.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (pymupdf) 가 필요합니다. `uv pip install pymupdf` 로 설치하세요."
        ) from exc

    try:
        doc = fitz.open(str(source))
    except Exception as exc:
        return f"PyMuPDF 열기 실패(DRM/손상 가능): {exc}"
    try:
        if bool(getattr(doc, "needs_pass", False)):
            return "암호화 PDF(needs_pass=True)"
    finally:
        doc.close()
    return ""


def _extract_via_viewer(source: Path, out_dir: Path) -> int:
    """DRM/암호화 PDF 를 기본 뷰어 화면 캡처로 추출한다(Windows 전용)."""
    from side_projects.document_extraction.util.viewer_capture import (
        capture_paged_viewer,
    )

    return capture_paged_viewer(
        source,
        out_dir,
        fullscreen_keys=PDF_VIEWER_FULLSCREEN_KEYS,
        next_key=PDF_VIEWER_NEXT_KEY,
    )


def extract(source: Path, out_dir: Path) -> int:
    """PDF 페이지를 WebP(1MB 이하)로 추출한다. 저장된 페이지 수를 반환."""
    if not source.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없다: {source}")

    print(f"[INFO] PDF 추출 시작: {source.name}")

    reason = _probe_unreadable(source)
    if reason:
        print(f"[WARNING] 직접 렌더 불가: {reason}")
        if os.name != "nt":
            raise RuntimeError(
                f"DRM/암호화 PDF 는 Windows 뷰어 캡처 폴백이 필요하다: {source.name}"
            )
        page_count = _extract_via_viewer(source, out_dir)
        print(f"[INFO] PDF 추출 완료(뷰어 캡처): {source.name} ({page_count}페이지)")
        return page_count

    next_index = render_pdf_to_webps(
        source,
        out_dir,
        start_index=1,
        dpi=RENDER_DPI,
    )
    page_count = next_index - 1
    print(f"[INFO] PDF 추출 완료: {source.name} ({page_count}페이지)")
    return page_count
