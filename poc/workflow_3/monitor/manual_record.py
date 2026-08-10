"""엔지니어 수동 조작 녹화 런처 - 알람 없이 이미 열린 tool 창을 녹화한다.

알람 사이클(`monitor/cycle.py`)의 녹화는 align fail 이 떠야만 시작된다. 이 모듈은
엔지니어와 "지금부터 녹화하겠다"고 약속한 뒤, 이미 열려 있는 Remote Monitoring 창을
그 자리에서 녹화하기 위한 독립 진입점이다. 접속(tool 더블클릭)은 하지 않는다.

수집한 프레임은 모방 학습/절차 분석의 원천 데이터가 되며, 분석은 별도 실행이다
(`recording_filter/filter_recording.py`).

실행:
    uv run python poc/workflow_3/monitor/manual_record.py
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.rcs.login_rcs_common import REMOTE_MONITORING_WINDOW_TITLE_PREFIX
from poc.workflow_3.util import env_flag, env_float, env_int, make_timestamp_tag

# 폴더명으로 쓸 수 없는 문자(Windows 예약 문자 + 공백/괄호)를 밑줄로 바꾼다.
# \w 는 유니코드 단어 문자(한글 포함)까지 허용한다 - ASCII 로 한정하면 "장비1" 같은
# 한글 EQP 명이 전부 "1" 처럼 깎여나가 서로 다른 장비가 같은 폴더로 충돌한다
# (2026-08-10 코디네이터 리뷰 FINDING 2). "." 는 여기서 허용 문자라 정규식만으로는
# ".."(부모 디렉터리 이동) 를 걸러내지 못한다 - 그건 sanitize_eqp_for_path 의
# 후처리(양끝 "._- " 트림 + 빈 결과 폴백)가 담당한다 (FINDING 1).
_PATH_HOSTILE_RE = re.compile(r"[^\w.-]+", re.UNICODE)
# EQP 를 못 읽었을 때의 대체 폴더명 - 프레임을 잃는 것보다 낫다.
UNKNOWN_EQP = "unknown_eqp"
# 수동 세션 전용 하위 폴더명 (알람 캡처의 captured_img_from_rcs 와 구분).
MANUAL_DIRNAME = "_manual"


def parse_eqp_from_title(title) -> str:
    """창 제목에서 EQP 문자열을 추출한다(접두어 제거). 실패하면 빈 문자열.

    제목은 "Remote Monitoring System - <EQP>" 형태다. 접두어 매칭은 대소문자를
    무시하고, EQP 뒤에 부가 정보가 붙어 있으면 통째로 보존한다(폴더명 정규화는
    sanitize_eqp_for_path 의 몫이라 여기서는 자르지 않는다).
    """
    normalized = (title or "").strip()
    prefix = REMOTE_MONITORING_WINDOW_TITLE_PREFIX
    if len(normalized) < len(prefix):
        return ""
    if normalized[: len(prefix)].lower() != prefix.lower():
        return ""
    return normalized[len(prefix):].strip()


def sanitize_eqp_for_path(eqp) -> str:
    """EQP 문자열을 폴더명으로 안전한 형태로 바꾼다. 비면 UNKNOWN_EQP.

    양끝의 "." / "-" / "_" / 공백은 잘라낸다 - Windows 는 이름 끝의 "." 를
    잘못 처리하고("MCD916." 같은 폴더), 입력이 온통 "."/".."로만 되어 있으면
    (".", "..", "...") 트림 후 빈 문자열이 되어 자동으로 UNKNOWN_EQP 로 폴백한다.
    이 폴백이 없으면 manual_recording_dir 가 ALIGN_IMAGES_DIR / ".." 를 만들어
    의도한 루트 밖에 쓰게 된다.
    """
    cleaned = _PATH_HOSTILE_RE.sub("_", (eqp or "").strip())
    cleaned = cleaned.strip("._- \t")
    return cleaned or UNKNOWN_EQP


def manual_recording_dir(eqp_id, tag):
    """수동 세션 프레임 저장 폴더 - <root>/<eqp>/_manual/<tag>/recording."""
    return ALIGN_IMAGES_DIR / sanitize_eqp_for_path(eqp_id) / MANUAL_DIRNAME / str(tag) / "recording"


@dataclass
class ManualRecordSettings:
    """수동 녹화 세션 파라미터 (CLI 인자 없음 - 모듈 상수 + env 오버라이드)."""

    max_sec: float = 600.0        # 실질 상한. 0 이면 무제한.
    max_frames: int = 4000        # 백스톱. 정상이면 안 걸린다. 0 이면 무제한.
    max_disk_mb: float = 2000.0   # 백스톱. 0 이면 무제한.
    poll_sec: float = 0.2         # 샘플링 요청 간격(실제 주기는 처리시간 + 이 값).
    heartbeat_sec: float = 5.0    # 변화가 없어도 이 간격마다 1장.
    change_min_px: int = 4        # 변화 판정 임계(알람 녹화와 동일).
    jpeg_quality: int = 85        # q95 대비 용량 약 절반.
    eqp_id: str = ""              # 모니터링 창이 여럿일 때만 필요.
    meta_enabled: bool = True     # 사이드카 기록 on/off.
    watch_interval_sec: float = 5.0   # 예산 감시 주기.


def load_manual_record_settings() -> ManualRecordSettings:
    """env 오버라이드를 적용한 설정을 만든다."""
    import os

    return ManualRecordSettings(
        max_sec=env_float("MANUAL_RECORD_MAX_SEC", 600.0),
        max_frames=env_int("MANUAL_RECORD_MAX_FRAMES", 4000),
        max_disk_mb=env_float("MANUAL_RECORD_MAX_DISK_MB", 2000.0),
        poll_sec=env_float("MANUAL_RECORD_POLL_SEC", 0.2),
        heartbeat_sec=env_float("MANUAL_RECORD_HEARTBEAT_SEC", 5.0),
        change_min_px=env_int("MANUAL_RECORD_CHANGE_MIN_PX", 4),
        jpeg_quality=env_int("MANUAL_RECORD_JPEG_QUALITY", 85),
        eqp_id=os.getenv("MANUAL_RECORD_EQP_ID", "").strip(),
        meta_enabled=env_flag("MANUAL_RECORD_META", True),
        watch_interval_sec=env_float("MANUAL_RECORD_WATCH_INTERVAL_SEC", 5.0),
    )


def budget_stop_reason(frame_count, disk_mb, settings) -> str:
    """프레임/디스크 예산 초과 사유를 돌려준다. 여유가 있으면 빈 문자열.

    사유는 하나만 돌려준다 - manifest 만 보고 원인을 구분할 수 있어야 하기 때문이다.
    0 은 무제한을 뜻하며 max_sec 규약과 같다.
    """
    if settings.max_frames > 0 and frame_count >= settings.max_frames:
        return "frame_budget"
    if settings.max_disk_mb > 0 and disk_mb >= settings.max_disk_mb:
        return "disk_budget"
    return ""


def dir_size_mb(path) -> float:
    """폴더 안 파일 용량 합계를 MB 로 돌려준다(실패는 0.0)."""
    total = 0
    try:
        for item in Path(path).rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        return 0.0
    return total / (1024.0 * 1024.0)


def pick_window_row(rows, wanted_eqp):
    """모니터링 창 후보 중 하나를 고른다.

    rows 는 (title, handle) 튜플 목록이다. 후보가 여럿인데 EQP 지정이 없으면
    None 을 돌려준다 - 엉뚱한 장비를 녹화하느니 다시 실행하는 편이 낫다.
    EQP 를 줬어도 그 부분 문자열에 매칭되는 창이 둘 이상이면(예: "MCD91" 이
    MCD916 과 MCD917 모두에 매칭) 첫 번째를 임의로 채택하지 않고 마찬가지로
    None 을 돌려준다 - 그렇지 않으면 엉뚱한 장비가 조용히 선택된다
    (2026-08-10 리뷰 FINDING 1).
    """
    if not rows:
        return None
    wanted = (wanted_eqp or "").strip().lower()
    if wanted:
        matches = [row for row in rows if wanted in (row[0] or "").lower()]
        if len(matches) == 1:
            return matches[0]
        return None
    if len(rows) == 1:
        return rows[0]
    return None


def resolve_capture_handles(
    tool_window, picked_handle, resolved_title="", picked_title="", *, extract_fn=None
):
    """가림 판정에 쓸 핸들 집합을 **실제 캡처 대상 창**에서 뽑는다.

    (2026-08-10 최종 리뷰 FINDING 5) 창 선택(pick_window_row)과 캡처 대상 해석
    (find_remote_monitoring_window)은 각자 독립적으로 제목을 매칭한다. 둘이
    다른 창으로 갈리면 캡처는 A, 가림 판정은 B 가 되어 모든 프레임이 경고 없이
    "full" 로 찍히고 분석에서 전량 폐기된다.

    resolved 핸들을 못 얻으면 고른 핸들로 폴백하고, 두 핸들이 다르면 둘 다
    이름을 찍어 경고한 뒤 **resolved 핸들**(실제 캡처하는 창)로 진행한다.
    extract_fn 은 테스트 주입용이다(window_utils 는 Windows 전용이라 Mac 에서
    import 자체가 실패한다).
    """
    extractor = extract_fn
    if extractor is None:
        try:
            from poc.workflow_3.util.window_utils import _extract_window_handle

            extractor = _extract_window_handle
        except Exception as exc:
            print(f"[WARNING] 창 핸들 추출 유틸을 쓸 수 없습니다(고른 핸들 사용): {exc}")

    resolved_handle = None
    if extractor is not None:
        try:
            resolved_handle = extractor(tool_window)
        except Exception as exc:
            print(f"[WARNING] 캡처 대상 창의 핸들 추출 실패(고른 핸들 사용): {exc}")

    if not resolved_handle:
        print(
            "[WARNING] 캡처 대상 창의 핸들을 얻지 못했습니다 - 목록에서 고른 핸들로 "
            f"가림을 판정합니다(handle={picked_handle})."
        )
        return {int(picked_handle)} if picked_handle else set()

    if picked_handle and int(resolved_handle) != int(picked_handle):
        print(
            "[WARNING] 선택한 창과 실제 캡처 창이 다릅니다 - 캡처 창 기준으로 "
            f"가림을 판정합니다. 선택={picked_title!r}(handle={picked_handle}), "
            f"캡처={resolved_title!r}(handle={int(resolved_handle)})"
        )
    return {int(resolved_handle)}


from poc.workflow_3.monitor.frame_meta import (
    FrameMetaWriter,
    build_meta_record,
    probe_occlusion,
    read_cursor_screen_xy,
)
from poc.workflow_3.monitor.recording import RecordingSession


def _collect_monitoring_rows():
    """열려 있는 Remote Monitoring 창 목록을 (title, handle) 로 모은다."""
    from poc.workflow_3.util import collect_window_rows

    if collect_window_rows is None:
        print("[ERROR] window_utils 를 쓸 수 없습니다(Windows 에서 실행하세요).")
        return []
    prefix = REMOTE_MONITORING_WINDOW_TITLE_PREFIX.lower()
    rows = []
    for row in collect_window_rows(visible_only=True):
        if (row.title or "").strip().lower().startswith(prefix):
            rows.append((row.title, row.handle))
    return rows


def _make_capture_fn(tool_window, meta_writer, started_at, our_handles):
    """RecordingSession 에 주입할 capture_fn 을 만든다(캡처 + 사이드카 기록).

    RecordingSession 은 수정하지 않는다. 캡처 함수를 감싸는 것만으로 프레임과
    같은 시각의 창 rect/가림/커서를 남길 수 있다. 사이드카 기록 실패는 삼켜
    캡처 자체를 방해하지 않는다.
    """
    from poc.workflow_3.util import capture_window

    state = {"seq": 0}

    def _capture():
        image = capture_window(tool_window)
        if meta_writer is None:
            return image
        try:
            rect_obj = tool_window.rectangle()
            rect = {
                "left": int(rect_obj.left), "top": int(rect_obj.top),
                "right": int(rect_obj.right), "bottom": int(rect_obj.bottom),
            }
            from poc.workflow_3.util import read_foreground_window_info

            _fg_handle, fg_title = (
                read_foreground_window_info() if read_foreground_window_info else (None, "")
            )
            meta_writer.append(build_meta_record(
                frame_name=f"seq_{state['seq']:04d}",
                t_sec=time.time() - started_at,
                rect=rect,
                foreground_title=fg_title,
                occlusion=probe_occlusion(rect, our_handles),
                cursor_xy=read_cursor_screen_xy(),
            ))
            state["seq"] += 1
        except Exception as exc:
            print(f"[WARNING] frame_meta 수집 실패(계속 진행): {exc}")
        return image

    return _capture


def _watch_until_stop(session, out_dir, settings) -> str:
    """세션이 끝나거나 예산/중단 신호가 올 때까지 감시하고 중지 사유를 돌려준다.

    이 함수는 session 을 직접 멈추지 않는다 - 관찰만 하고 사유 문자열을
    리턴한다. 그래야 호출부(main)가 반환값과 무관하게 항상
    session.stop()/meta_writer.close() 를 실행할 수 있다("teardown 은 항상
    완료된다"는 저장소 규칙, 2026-08-10 리뷰 FINDING 2). KeyboardInterrupt 는
    기존과 동일하게 "user_interrupt" 다. 그 외의 예기치 못한 예외
    (is_alive/sleep/budget_stop_reason/dir_size_mb 어디서든 날 수 있다) 는
    "watch_error" 로 잡아 삼킨다 - 그러지 않으면 예외가 main() 밖으로 새어나가
    녹화 스레드가 계속 돌고 manifest 도 안 써지는 채로 프로세스가 끝난다.
    """
    try:
        while session.is_alive():
            time.sleep(settings.watch_interval_sec)
            reason = budget_stop_reason(len(session.frames), dir_size_mb(out_dir), settings)
            if reason:
                print(f"[WARNING] ===== 예산 상한 도달({reason}) - 녹화를 종료합니다 =====")
                return reason
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 감지 - 녹화를 종료합니다.")
        return "user_interrupt"
    except Exception as exc:
        print(f"[WARNING] 감시 루프에서 예기치 못한 오류(녹화는 종료합니다): {exc}")
        return "watch_error"
    return "stopped"


def main() -> int:
    """수동 녹화 세션을 실행한다. 종료 코드 0=정상, 1=시작 실패."""
    settings = load_manual_record_settings()
    rows = _collect_monitoring_rows()
    if not rows:
        print("[ERROR] 열려 있는 Remote Monitoring 창이 없습니다. RCS 에서 tool 을 먼저 열어주세요.")
        return 1

    chosen = pick_window_row(rows, settings.eqp_id)
    if chosen is None:
        if settings.eqp_id:
            print(
                f"[ERROR] MANUAL_RECORD_EQP_ID={settings.eqp_id!r} 로 특정 창 하나를 "
                f"고를 수 없습니다(매칭 0개 또는 여러 개). 후보:"
            )
        else:
            print(f"[ERROR] 모니터링 창이 {len(rows)}개 있습니다. MANUAL_RECORD_EQP_ID 로 지정하세요:")
        for title, handle in rows:
            print(f"        - {title}  (handle={handle})")
        return 1

    title, handle = chosen
    eqp_id = parse_eqp_from_title(title) or UNKNOWN_EQP
    tag = make_timestamp_tag()
    out_dir = manual_recording_dir(eqp_id, tag)
    print(f"[INFO] 수동 녹화 대상: eqp={eqp_id!r}, title={title!r}")
    print(f"[INFO] 저장 경로: {out_dir}")
    print(
        f"[INFO] 상한: max_sec={settings.max_sec}s, max_frames={settings.max_frames}, "
        f"max_disk_mb={settings.max_disk_mb}, poll={settings.poll_sec}s"
    )

    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window

    tool_window, resolved_title, _backend = find_remote_monitoring_window(eqp_id)
    if tool_window is None:
        print(f"[ERROR] 창 핸들을 얻지 못했습니다: {title!r}")
        return 1

    # (2026-08-10 최종 리뷰 FINDING 5) 가림 판정 기준 핸들은 **실제로 캡처하는 창**
    # 에서 뽑는다. pick_window_row 가 고른 handle 과 find_remote_monitoring_window
    # 가 독립적으로 찾은 창이 어긋나면, 캡처는 A 를 하면서 가림은 B 로 재는 셈이라
    # 모든 프레임이 조용히 "full"(=분석 전량 폐기)로 찍힌다.
    our_handles = resolve_capture_handles(tool_window, handle, resolved_title, title)

    meta_writer = FrameMetaWriter(out_dir) if settings.meta_enabled else None
    started_at = time.time()
    session = RecordingSession(
        tool_window, out_dir, tag=tag,
        poll_sec=settings.poll_sec,
        heartbeat_sec=settings.heartbeat_sec,
        change_min_px=settings.change_min_px,
        max_sec=settings.max_sec,
        jpeg_quality=settings.jpeg_quality,
        capture_fn=_make_capture_fn(
            tool_window, meta_writer, started_at, our_handles,
        ),
    )
    session.start()
    print("[INFO] 녹화 중입니다. 중지하려면 Ctrl+C 를 누르세요.")

    stop_reason = _watch_until_stop(session, out_dir, settings)

    frames = session.stop(stop_reason)
    if meta_writer is not None:
        meta_writer.close()
    print(f"[INFO] ===== 녹화 종료: {len(frames)} 프레임, 사유={session.stop_reason} =====")
    print(f"[INFO] 프레임 경로: {out_dir}")
    print("[INFO] 분석하려면: RECORDING_FILTER_INPUT_DIR=<위 경로> "
          "uv run python poc/workflow_3/recording_filter/filter_recording.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
