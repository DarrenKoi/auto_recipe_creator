"""PowerPoint 추출 핸들러 — 슬라이드쇼 모드 진입 후 화면 캡처.

기존 screenshot_document_extraction의 Slide.Export 방식과 달리,
실제 사용자가 발표 시 보게 될 픽셀과 동일한 결과를 얻기 위해
SlideShowSettings.Run()으로 슬라이드쇼를 띄우고 화면을 캡처한다.

듀얼 모니터 환경에서는 발표자 보기가 활성화될 수 있으므로
ShowPresenterView=False / ShowType=ppShowTypeSpeaker 로 단일 모니터를 강제한다.
캡처는 항상 primary 모니터(mss monitors[1])를 대상으로 한다.
"""

import time
from pathlib import Path

from side_projects.document_extraction.util.output_paths import page_image_path
from side_projects.document_extraction.util.screen_capture import (
    capture_primary_monitor,
    save_webp_capped,
)


# PowerPoint COM 상수
PP_SHOW_TYPE_SPEAKER = 1            # ppShowTypeSpeaker
PP_SLIDE_SHOW_MANUAL_ADVANCE = 1    # ppSlideShowManualAdvance
MSO_FALSE = 0

# 슬라이드쇼 진입/전환 대기 시간(초)
INITIAL_RENDER_WAIT = 1.5
SLIDE_ADVANCE_WAIT = 0.6


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
    """PowerPoint를 슬라이드쇼로 띄우고 슬라이드별 화면 캡처."""
    if not source.exists():
        raise FileNotFoundError(f"PowerPoint 파일을 찾을 수 없다: {source}")

    client = _import_com()
    import pythoncom

    print(f"[INFO] PPT 추출 시작: {source.name}")

    pythoncom.CoInitialize()
    app = None
    presentation = None
    slideshow_window = None
    try:
        app = client.Dispatch("PowerPoint.Application")
        # PowerPoint는 Visible=True 가 아니면 슬라이드쇼를 띄울 수 없는 경우가 있다.
        try:
            app.Visible = True
        except Exception:
            pass

        presentation = app.Presentations.Open(
            str(source.resolve()),
            ReadOnly=True,
            Untitled=False,
            WithWindow=True,
        )

        settings = presentation.SlideShowSettings
        # 단일 모니터 강제: 발표자 보기 비활성화 + 발표자 타입으로 고정
        try:
            settings.ShowPresenterView = MSO_FALSE
        except Exception as e:
            print(f"[WARNING]   - ShowPresenterView 설정 실패(무시): {e}")
        try:
            settings.ShowType = PP_SHOW_TYPE_SPEAKER
        except Exception:
            pass
        try:
            settings.AdvanceMode = PP_SLIDE_SHOW_MANUAL_ADVANCE
        except Exception:
            pass

        slide_count = presentation.Slides.Count
        if slide_count <= 0:
            print(f"[WARNING] 슬라이드가 없습니다: {source.name}")
            return 0

        slideshow_window = settings.Run()
        time.sleep(INITIAL_RENDER_WAIT)

        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, slide_count + 1):
            image = capture_primary_monitor()
            out_path = page_image_path(out_dir, i)
            save_webp_capped(image, out_path)
            print(f"[INFO]   - 슬라이드 {i}/{slide_count} 캡처 → {out_path.name}")

            if i < slide_count:
                try:
                    slideshow_window.View.Next()
                except Exception as e:
                    print(f"[WARNING]   - View.Next 실패({i}): {e}")
                    break
                time.sleep(SLIDE_ADVANCE_WAIT)

        return slide_count
    finally:
        # 슬라이드쇼 종료
        if slideshow_window is not None:
            try:
                slideshow_window.View.Exit()
            except Exception:
                pass
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()
        print(f"[INFO] PPT 추출 완료: {source.name}")
