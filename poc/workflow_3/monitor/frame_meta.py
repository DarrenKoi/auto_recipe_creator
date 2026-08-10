"""수동 녹화 프레임의 사이드카 메타 기록 - 창 rect / 전면 창 / 가림 / 커서 좌표.

capture_window 는 창 핸들이 아니라 **창 rect 의 스크린 그랩**이라(util/image_utils.py),
다른 앱이 위에 뜨면 그 앱이 찍힌다. 수십 분 도는 수동 세션에서는 실제로 발생하므로,
프레임마다 "그때 이 창이 실제로 보이고 있었는가"를 같이 남겨 분석 단계에서 걸러낸다.

가림 판정은 픽셀이 아니라 기하다 - 창 영역 5개 지점에서 WindowFromPoint 를 찍어
그 지점의 최상위 창이 우리 창인지 본다. 포커스를 뺏지 않은 부분 겹침까지 잡힌다.

커서 좌표는 GetCursorPos 폴링이며 **입력 후킹이 아니다.** 키 입력은 기록하지 않는다.
"""

import ctypes
import json
import os
from pathlib import Path

# 사이드카 파일명 - 분석 단계가 frame 키로 프레임과 조인한다.
FRAME_META_FILENAME = "frame_meta.jsonl"
# 가림 표본점 수 (중앙 + 사분면).
_PROBE_COUNT = 5


def classify_occlusion(hit_handles, our_handles) -> str:
    """표본점의 최상위 창 핸들 목록으로 가림 정도를 판정한다.

    반환: "none"(전부 우리 창) | "partial"(일부) | "full"(하나도 아님) |
          "unknown"(표본 없음 - 조회 실패라 판정하지 않는다).
    """
    hits = [h for h in (hit_handles or []) if h is not None]
    if not hits:
        return "unknown"
    ours = sum(1 for h in hits if h in (our_handles or set()))
    if ours == len(hits):
        return "none"
    if ours == 0:
        return "full"
    return "partial"


def probe_points(rect) -> list:
    """창 rect 내부의 표본점 5개(중앙 + 사분면 중심)를 화면 좌표로 만든다.

    경계에 붙으면 이웃 창이 잡힐 수 있어 안쪽으로 넣는다. 아주 작은 창에서도
    좌표가 rect 밖으로 나가지 않도록 클램프한다.
    """
    left, top = int(rect["left"]), int(rect["top"])
    right, bottom = int(rect["right"]), int(rect["bottom"])
    width = max(1, right - left)
    height = max(1, bottom - top)

    def _clamp(value, low, high):
        return max(low, min(high, value))

    cx = left + width // 2
    cy = top + height // 2
    qx, qy = width // 4, height // 4
    raw = [
        (cx, cy),
        (cx - qx, cy - qy),
        (cx + qx, cy - qy),
        (cx - qx, cy + qy),
        (cx + qx, cy + qy),
    ]
    points = [
        (_clamp(x, left + 1, right - 1), _clamp(y, top + 1, bottom - 1))
        for x, y in raw
    ]
    return points[:_PROBE_COUNT]


def _win_point_type():
    """ctypes.wintypes.POINT 를 지연 import 로 얻는다(Mac 에서 import 실패 회피).

    `import ctypes` 만으로는 ctypes.wintypes 가 로드되지 않으며, 비 Windows 에서는
    wintypes import 자체가 실패한다. 그래서 Windows 분기 안에서만 가져온다.
    """
    from ctypes import wintypes

    return wintypes.POINT


def read_cursor_screen_xy():
    """현재 커서의 화면 좌표를 읽는다(입력 후킹 아님). 실패/비 Windows 는 None."""
    if os.name != "nt":
        return None
    try:
        point = _win_point_type()()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return (int(point.x), int(point.y))
    except Exception:
        return None
    return None


def probe_occlusion(rect, our_handles) -> str:
    """rect 표본점에서 WindowFromPoint 를 찍어 가림 정도를 판정한다."""
    if os.name != "nt" or not rect:
        return "unknown"
    try:
        user32 = ctypes.windll.user32
        point_type = _win_point_type()
        hits = []
        for x, y in probe_points(rect):
            hits.append(int(user32.WindowFromPoint(point_type(x, y))))
    except Exception as exc:
        print(f"[WARNING] 가림 판정 실패(unknown 으로 기록): {exc}")
        return "unknown"
    return classify_occlusion(hits, our_handles)


def build_meta_record(
    *, frame_name, t_sec, rect, foreground_title, occlusion, cursor_xy
) -> dict:
    """사이드카 1줄 레코드를 만든다(커서가 창 안인지도 함께 계산)."""
    cursor_list = [int(cursor_xy[0]), int(cursor_xy[1])] if cursor_xy else None
    in_window = False
    if cursor_list and rect:
        in_window = (
            int(rect["left"]) <= cursor_list[0] <= int(rect["right"])
            and int(rect["top"]) <= cursor_list[1] <= int(rect["bottom"])
        )
    return {
        "frame": frame_name,
        "t_sec": round(float(t_sec), 3),
        "window_rect": rect,
        "foreground_title": foreground_title or "",
        "occlusion": occlusion,
        "cursor_screen_xy": cursor_list,
        "cursor_in_window": bool(in_window),
    }


class FrameMetaWriter:
    """frame_meta.jsonl 에 프레임당 1줄을 append 한다(실패는 삼킨다)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / FRAME_META_FILENAME
        self._handle = None
        self._failed = False

    def _ensure_open(self):
        if self._handle is not None or self._failed:
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        except Exception as exc:
            self._failed = True
            print(f"[WARNING] frame_meta 기록 비활성화(열기 실패): {exc}")

    def append(self, record) -> None:
        """레코드 1건을 기록한다. 어떤 실패도 밖으로 던지지 않는다.

        열기 실패든 쓰기 실패든 한 번 _failed 가 서면 이후 append 는 즉시
        반환한다 - 재시도도, 재경고도 하지 않는다. 5초 단위로 몇 분씩 도는
        루프에서 지속 장애가 나면 프레임마다 경고를 찍어 콘솔을 도배하는 것을
        막기 위함이다(경고는 최초 1회만).
        """
        if self._failed:
            return
        self._ensure_open()
        if self._handle is None:
            return
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        except Exception as exc:
            self._failed = True
            print(f"[WARNING] frame_meta 기록 실패(이후 생략): {exc}")

    def close(self) -> None:
        """파일 핸들을 닫는다(실패 무시)."""
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None
