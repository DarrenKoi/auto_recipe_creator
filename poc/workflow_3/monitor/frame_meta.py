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
import time
from pathlib import Path

# 사이드카 파일명 - 분석 단계가 frame 키로 프레임과 조인한다.
FRAME_META_FILENAME = "frame_meta.jsonl"
# 가림 표본점 수 (중앙 + 사분면).
_PROBE_COUNT = 5
# GetAncestor(hwnd, GA_ROOT) - 자식 컨트롤 핸들을 그 창의 최상위 창으로 올린다.
GA_ROOT = 2


def classify_occlusion(hit_handles, our_handles) -> str:
    """표본점의 최상위 창 핸들 목록으로 가림 정도를 판정한다.

    반환: "none"(전부 우리 창) | "partial"(일부) | "full"(하나도 아님) |
          "unknown"(표본 없음 - 조회 실패라 판정하지 않는다).

    (2026-08-10 최종 리뷰 FINDING 1) None 뿐 아니라 0 도 "정보 없음"으로 버린다.
    WindowFromPoint 는 실패 시 NULL(=0) 을 돌려주는데, 이를 "남의 창"으로 세면
    조회가 전부 실패한 프레임이 "full" 로 확정되어 분석 단계에서 통째로
    버려진다 - 재현 불가능한 녹화에서 가장 나쁜 실패 형태다. 전부 0 이면
    "unknown" 이어야 한다.
    """
    hits = [h for h in (hit_handles or []) if h]
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


def normalize_hits_to_root(raw_hits, resolve_root) -> list:
    """WindowFromPoint 결과를 최상위(root) 창 핸들로 정규화한다.

    (2026-08-10 최종 리뷰 FINDING 1) WindowFromPoint 는 그 지점의 **자식 컨트롤**
    핸들을 돌려준다. RCS Remote Monitoring 은 MFC 계열이라 창 안이 온통 자식
    컨트롤이고, 그래서 원시 핸들을 top-level 핸들 집합과 비교하면 5점 모두
    불일치 -> 모든 프레임이 "full" 로 찍혀 분석에서 전량 폐기됐다.

    resolve_root 는 핸들 하나를 받아 root 핸들을 돌려주는 콜러블이다
    (Windows 에서는 GetAncestor(hwnd, GA_ROOT)). GetAncestor 를 Mac 에서 부를 수
    없어 순수 함수로 분리해 두었다 - 정규화 규칙 자체는 여기서 테스트한다.

    0/None(정보 없음)은 그대로 None 으로 남기고, resolve_root 가 실패하거나
    0 을 돌려줘도 원시 핸들로 되돌리지 않고 None 으로 둔다 - "판정 불가" 가
    "남의 창" 으로 둔갑해 프레임을 잃는 것보다 낫다.
    """
    normalized = []
    for hit in (raw_hits or []):
        if not hit:
            normalized.append(None)
            continue
        try:
            root = resolve_root(int(hit))
        except Exception:
            root = None
        normalized.append(int(root) if root else None)
    return normalized


def probe_occlusion(rect, our_handles) -> str:
    """rect 표본점에서 WindowFromPoint 를 찍어 가림 정도를 판정한다.

    원시 결과는 자식 컨트롤 핸들이므로 GetAncestor(GA_ROOT) 로 최상위 창까지
    올린 뒤에 우리 창인지 비교한다(FINDING 1).
    """
    if os.name != "nt" or not rect:
        return "unknown"
    try:
        user32 = ctypes.windll.user32
        point_type = _win_point_type()
        raw_hits = []
        for x, y in probe_points(rect):
            raw_hits.append(int(user32.WindowFromPoint(point_type(x, y))))
        hits = normalize_hits_to_root(
            raw_hits, lambda handle: int(user32.GetAncestor(handle, GA_ROOT))
        )
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


class FrameMetaRecorder:
    """`capture_fn` 을 감싸 프레임마다 사이드카 1줄을 남기는 공용 래퍼.

    수동 녹화 런처와 알람 사이클이 **같은** 래퍼를 쓴다. `RecordingSession` 은
    건드리지 않는다 - 캡처 함수를 감싸는 것만으로 프레임과 같은 시각의 창 rect /
    전면 창 / 가림 / 커서를 남길 수 있기 때문이다(주입점이 이미 계약이다).

    커서는 `GetCursorPos` 폴링 결과를 **기록만** 한다. 어떤 Action 이나 의도 claim 도
    만들지 않는다 - 그 해석은 오프라인 분석 단계(recording_filter)의 몫이다.

    수집 실패는 **1회 경고 후 영구 비활성**이다. 20fps 루프에서 프레임마다 경고를
    찍으면 콘솔이 도배되어 정작 중요한 로그가 묻힌다. 대신 사유를 남겨
    `completeness()` 가 manifest 로 내보낸다 - 사이드카가 왜 비었는지는 사후에
    알 수 있어야 한다.
    """

    def __init__(self, tool_window, out_dir, *, our_handles=None, started_at=None):
        self.tool_window = tool_window
        self.writer = FrameMetaWriter(out_dir)
        self.started_at = time.time() if started_at is None else float(started_at)
        self.our_handles = (
            set(our_handles) if our_handles is not None
            else _handles_of(tool_window)
        )
        self.records = 0
        self.disabled_reason = ""
        # 마지막으로 기록한 레코드와 그 시각 - Guard 판독이 "그때 창이 보였는가" 를
        # 파일을 다시 파싱하지 않고 물어볼 수 있게 한다(그리고 stale 판정의 기준).
        self.last_record = None
        self.last_at = 0.0
        self._seq = 0

    def wrap(self, capture_fn):
        """캡처 함수를 감싼 새 함수를 돌려준다(원본은 그대로 호출된다)."""
        def _capture():
            image = capture_fn()
            if not self.disabled_reason:
                self._append_for(image)
            return image

        return _capture

    def _append_for(self, _image) -> None:
        try:
            rect_obj = self.tool_window.rectangle()
            rect = {
                "left": int(rect_obj.left), "top": int(rect_obj.top),
                "right": int(rect_obj.right), "bottom": int(rect_obj.bottom),
            }
            _fg_handle, fg_title = _read_foreground()
            record = build_meta_record(
                frame_name=f"seq_{self._seq:04d}",
                t_sec=time.time() - self.started_at,
                rect=rect,
                foreground_title=fg_title,
                occlusion=probe_occlusion(rect, self.our_handles),
                cursor_xy=read_cursor_screen_xy(),
            )
            self.writer.append(record)
            self.last_record = record
            self.last_at = time.time()
            self._seq += 1
            self.records += 1
        except Exception as exc:
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            print(f"[WARNING] frame_meta 수집 비활성화(녹화는 계속): {exc}")

    def completeness(self, frames: int = 0) -> dict:
        """manifest 에 실을 수집 완전성 요약 - 사이드카가 비면 사유가 남는다."""
        return {
            "meta_enabled": True,
            "frames": int(frames),
            "meta_records": self.records,
            "meta_disabled_reason": self.disabled_reason,
        }

    def close(self) -> None:
        self.writer.close()


def _handles_of(tool_window) -> set:
    """가림 판정 기준 핸들 - 창에서 직접 뽑는다. 못 뽑으면 빈 집합(=unknown 판정)."""
    try:
        from poc.workflow_3.util.window_utils import _extract_window_handle

        handle = _extract_window_handle(tool_window)
    except Exception:
        return set()
    return {int(handle)} if handle else set()


def _read_foreground():
    """전면 창 정보를 읽는다. 유틸이 없는 환경(비 Windows)에서는 (None, "")."""
    try:
        from poc.workflow_3.util import read_foreground_window_info
    except Exception:
        return (None, "")
    if read_foreground_window_info is None:
        return (None, "")
    return read_foreground_window_info()
