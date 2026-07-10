"""DRM 보호 문서용 뷰어 화면 캡처 폴백 (Windows 전용 I/O + 순수 판정 로직).

PyMuPDF 직접 렌더(PDF)나 COM PDF export(Word/Excel)는 DRM 파일에서 실패한다.
그러나 DRM 은 "허가된 앱 안에서의 표시"는 항상 허용하므로, 기본 연결 프로그램으로
문서를 열고(os.startfile) 페이지 넘김 키를 보내며 primary 모니터를 캡처하면
사용자가 화면에서 보는 픽셀 그대로를 얻을 수 있다 — PPT 슬라이드쇼 캡처와 동일 원리.

총 페이지 수를 미리 알 수 없으므로 "페이지를 넘겼는데 화면이 그대로면 마지막"
이라는 frame-diff 종료 조건을 쓴다. 판정 함수(frames_look_identical)는 순수라
Mac 에서 테스트되고, 캡처 루프(capture_paged_viewer)는 office Windows 에서 돈다.

키 시퀀스는 WScript.Shell SendKeys 문법: "^"=Ctrl, "%"=Alt, "{PGDN}"/"{ESC}" 등.
뷰어별 전체화면/닫기 키는 핸들러 쪽 상수로 두고 office 에서 보정한다.
"""

import os
import time
from pathlib import Path

from PIL import Image

from side_projects.document_extraction.util.output_paths import page_image_path
from side_projects.document_extraction.util.screen_capture import (
    capture_primary_monitor,
    save_webp_capped,
)


# 캡처 루프 타이밍/한도 (office 보정 가능)
INITIAL_VIEWER_WAIT = 4.0     # 뷰어 첫 렌더 대기(초). DRM 복호화로 느릴 수 있음
FULLSCREEN_SETTLE_WAIT = 1.5  # 전체화면 전환 후 재렌더 대기(초)
PAGE_ADVANCE_WAIT = 0.8       # 페이지 넘김 키 이후 렌더 대기(초)
MAX_VIEWER_PAGES = 300        # 무한 루프 방지 상한

# frame-diff 판정 파라미터
DIFF_DOWNSCALE_WIDTH = 256    # 판정용 축소 폭(px). 커서 깜빡임 등 미세 노이즈 흡수
DIFF_MEAN_THRESHOLD = 1.5     # 평균 절대차(0~255)가 이 미만이면 "같은 화면"


def _to_diff_gray(image: Image.Image) -> Image.Image:
    """frame-diff 용으로 이미지를 축소 그레이스케일로 정규화한다(순수)."""
    w, h = image.size
    if w > DIFF_DOWNSCALE_WIDTH:
        new_h = max(1, int(h * DIFF_DOWNSCALE_WIDTH / w))
        image = image.resize((DIFF_DOWNSCALE_WIDTH, new_h), Image.BILINEAR)
    return image.convert("L")


def frames_look_identical(
    a: Image.Image,
    b: Image.Image,
    *,
    mean_diff_threshold: float = DIFF_MEAN_THRESHOLD,
) -> bool:
    """두 캡처 프레임이 사실상 같은 화면인지 판정한다(순수).

    축소 그레이스케일 후 픽셀 평균 절대차가 threshold 미만이면 True.
    페이지 넘김 키를 보냈는데도 True 면 "마지막 페이지 도달"로 해석한다.
    크기가 다르면(모니터 변경 등) 항상 False.
    """
    ga = _to_diff_gray(a)
    gb = _to_diff_gray(b)
    if ga.size != gb.size:
        return False
    pa = list(ga.getdata())
    pb = list(gb.getdata())
    total = 0
    for va, vb in zip(pa, pb):
        total += abs(va - vb)
    mean_diff = total / max(1, len(pa))
    return mean_diff < mean_diff_threshold


def _import_wshell():
    """WScript.Shell(SendKeys/AppActivate)을 lazy import 한다(Windows 전용)."""
    try:
        from win32com import client
    except ImportError as exc:
        raise ImportError(
            "pywin32가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client.Dispatch("WScript.Shell")


def capture_paged_viewer(
    source: Path,
    out_dir: Path,
    *,
    fullscreen_keys: str = "",
    next_key: str = "{PGDN}",
    close_keys: tuple[str, ...] = ("{ESC}", "%{F4}"),
    activate_title: str = "",
    max_pages: int = MAX_VIEWER_PAGES,
) -> int:
    """문서를 기본 뷰어로 열어 페이지 단위 화면 캡처한다(Windows 전용 I/O).

    1) os.startfile 로 기본 연결 프로그램에서 열기(DRM 은 허가 앱에서 복호화됨)
    2) (선택) fullscreen_keys 로 전체화면 진입
    3) 캡처 -> next_key -> 캡처 ... 화면이 안 변하면 종료(frame-diff)
    4) close_keys 로 뷰어 정리(전체화면 해제 + 창 닫기)

    캡처 중 키보드/마우스를 건드리면 안 된다(PPT 캡처와 동일 주의).
    저장된 페이지 수를 반환한다.
    """
    if os.name != "nt":
        raise RuntimeError(
            "viewer capture 폴백은 Windows 전용입니다 (office PC 에서 실행)."
        )

    shell = _import_wshell()
    print(f"[INFO] 뷰어 캡처 폴백 시작: {source.name} (입력 금지, 자리 비우지 말 것)")
    os.startfile(str(source.resolve()))  # noqa: S606 - 의도된 뷰어 실행
    time.sleep(INITIAL_VIEWER_WAIT)

    if activate_title:
        try:
            shell.AppActivate(activate_title)
            time.sleep(0.3)
        except Exception as exc:
            print(f"[WARNING] AppActivate 실패(무시): {exc}")

    if fullscreen_keys:
        try:
            shell.SendKeys(fullscreen_keys)
        except Exception as exc:
            print(f"[WARNING] 전체화면 키 전송 실패(무시): {exc}")
        time.sleep(FULLSCREEN_SETTLE_WAIT)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    prev: Image.Image | None = None
    try:
        for _ in range(max_pages):
            image = capture_primary_monitor()
            if prev is not None and frames_look_identical(prev, image):
                print("[INFO] 화면 변화 없음 -> 마지막 페이지로 판단, 캡처 종료")
                break
            saved += 1
            out_path = page_image_path(out_dir, saved)
            save_webp_capped(image, out_path)
            print(f"[INFO]   - 페이지 {saved} 캡처 -> {out_path.name}")
            prev = image
            shell.SendKeys(next_key)
            time.sleep(PAGE_ADVANCE_WAIT)
        else:
            print(f"[WARNING] max_pages({max_pages}) 도달 - 캡처를 강제 종료")
    finally:
        for keys in close_keys:
            try:
                shell.SendKeys(keys)
                time.sleep(0.4)
            except Exception:
                pass
        print(f"[INFO] 뷰어 캡처 폴백 완료: {source.name} ({saved}페이지)")
    return saved


__all__ = [
    "DIFF_MEAN_THRESHOLD",
    "MAX_VIEWER_PAGES",
    "capture_paged_viewer",
    "frames_look_identical",
]
