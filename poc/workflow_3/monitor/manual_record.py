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

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.config import (
    DEFAULT_RECORDING_CHANGE_MIN_PX,
    DEFAULT_RECORDING_HEARTBEAT_SEC,
    DEFAULT_RECORDING_POLL_SEC,
)
from poc.workflow_3.rcs.login_rcs_common import REMOTE_MONITORING_WINDOW_TITLE_PREFIX
from poc.workflow_3.util import (
    capture_window,
    env_flag,
    env_float,
    env_int,
    make_timestamp_tag,
)

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
# 파생 프레임 백스톱의 여유 배수 (frame_backstop 참조).
_BACKSTOP_HEADROOM = 1.25


# ==========================================================================
# 실행 인자 = 이 블록의 상수 (CLI 인자를 쓰지 않는 프로젝트 규약)
# --------------------------------------------------------------------------
# 우선순위: 실제 셸 env > 이 상수 > (없음). 상수가 곧 기본값이라 코드 안쪽에 숨은
# 세 번째 기본값이 없다 - 값을 바꾸려면 여기만 고친다.
#
# 이 스크립트는 알람과 무관하다. **이미 열려 있는** Remote Monitoring 창에 붙어
# 엔지니어의 수동 조작을 녹화한다(접속은 하지 않는다). 엔지니어와 "지금부터
# 녹화하겠다"고 약속한 뒤 실행할 것.
# ==========================================================================

EQP_ID = ""                  # 창이 여럿일 때 고를 장비. 비우면 창이 1개일 때만 진행한다
                             # (부분 일치가 모호하면 임의 선택하지 않고 거부한다 -
                             #  엉뚱한 장비를 10분 녹화하느니 다시 실행하는 편이 낫다).
MAX_SEC = 600.0              # 세션 상한(초). 실질 상한은 이것 하나다.
POLL_SEC = None              # 샘플링 주기(초). None=녹화 기본값(약 0.05 = 20fps).
JPEG_QUALITY = 85            # 알람 녹화(95)보다 낮다 - 수동 세션은 길고 장수가 많다.
META = 1                     # frame_meta.jsonl 사이드카(창 rect/전면 창/가림/커서).
                             # 끄면 recording_filter 의 커서 경로가 VLM 폴백으로 내려간다.
WATCH_INTERVAL_SEC = 5.0     # 창 생존 확인 주기(초).

# 아래 둘은 백스톱이다. MAX_SEC 보다 먼저 걸리면 안 된다 - MAX_FRAMES 를 고정값으로
# 두었다가 POLL_SEC 를 0.05 로 내리자 10분 세션이 ~3분에 끊긴 적이 있다.
MAX_FRAMES = None            # None = MAX_SEC/POLL_SEC 에서 파생(여유 1.25배).
MAX_DISK_MB = 4000.0         # 디스크 백스톱(MB).


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
    # 백스톱. 0 이면 무제한. 기본값은 max_sec/poll_sec 에서 파생한다 - 아래
    # frame_backstop() 참조. 고정 숫자로 두면 샘플링 주기를 올릴 때마다 백스톱이
    # 실질 상한보다 먼저 걸려 "정상이면 안 걸린다"는 전제가 조용히 깨진다.
    max_frames: int = 0
    max_disk_mb: float = 0.0      # 백스톱. 0 이면 무제한.
    poll_sec: float = DEFAULT_RECORDING_POLL_SEC            # 샘플링 요청 간격.
    heartbeat_sec: float = DEFAULT_RECORDING_HEARTBEAT_SEC  # 변화 없어도 이 간격마다 1장.
    change_min_px: int = DEFAULT_RECORDING_CHANGE_MIN_PX    # 변화 판정 임계(알람 녹화와 동일).
    jpeg_quality: int = 85        # q95 대비 용량 약 절반.
    eqp_id: str = ""              # 모니터링 창이 여럿일 때만 필요.
    meta_enabled: bool = True     # 사이드카 기록 on/off.
    watch_interval_sec: float = 5.0   # 예산 감시 주기.


def frame_backstop(max_sec: float, poll_sec: float) -> int:
    """max_sec 동안 매 샘플을 저장해도 걸리지 않는 프레임 백스톱을 계산한다.

    백스톱의 목적은 폭주 방지지 세션 단축이 아니다. 따라서 "정상 최악"(모든 샘플이
    변화로 판정되어 저장되는 경우 = max_sec/poll_sec)보다 넉넉히 위여야 한다.
    고정 상수로 두면 poll_sec 을 6배로 올린 순간 백스톱이 max_sec 보다 먼저 걸려,
    10분을 녹화하려던 세션이 3분에 frame_budget 으로 끝난다.
    """
    if max_sec <= 0 or poll_sec <= 0:
        return 0  # 무제한 세션에는 파생 백스톱을 걸지 않는다.
    return int((max_sec / poll_sec) * _BACKSTOP_HEADROOM)


def load_manual_record_settings() -> ManualRecordSettings:
    """env 오버라이드를 적용한 설정을 만든다."""
    import os

    max_sec = env_float("MANUAL_RECORD_MAX_SEC", MAX_SEC)
    poll_sec = env_float("MANUAL_RECORD_POLL_SEC", POLL_SEC or DEFAULT_RECORDING_POLL_SEC)
    return ManualRecordSettings(
        max_sec=max_sec,
        max_frames=env_int("MANUAL_RECORD_MAX_FRAMES", MAX_FRAMES or frame_backstop(max_sec, poll_sec)),
        max_disk_mb=env_float("MANUAL_RECORD_MAX_DISK_MB", MAX_DISK_MB),
        poll_sec=poll_sec,
        heartbeat_sec=env_float(
            "MANUAL_RECORD_HEARTBEAT_SEC", DEFAULT_RECORDING_HEARTBEAT_SEC
        ),
        change_min_px=env_int(
            "MANUAL_RECORD_CHANGE_MIN_PX", DEFAULT_RECORDING_CHANGE_MIN_PX
        ),
        jpeg_quality=env_int("MANUAL_RECORD_JPEG_QUALITY", JPEG_QUALITY),
        eqp_id=os.getenv("MANUAL_RECORD_EQP_ID", EQP_ID).strip(),
        meta_enabled=env_flag("MANUAL_RECORD_META", bool(META)),
        watch_interval_sec=env_float("MANUAL_RECORD_WATCH_INTERVAL_SEC", WATCH_INTERVAL_SEC),
    )


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


from poc.workflow_3.monitor.frame_meta import FrameMetaRecorder
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


def _watch_until_stop(session, settings) -> str:
    """세션이 끝나거나 중단 신호가 올 때까지 감시하고 중지 사유를 돌려준다.

    이 함수는 session 을 직접 멈추지 않는다 - 관찰만 하고 사유 문자열을
    리턴한다. 그래야 호출부(main)가 반환값과 무관하게 항상
    session.stop()/meta_writer.close() 를 실행할 수 있다("teardown 은 항상
    완료된다"는 저장소 규칙, 2026-08-10 리뷰 FINDING 2). KeyboardInterrupt 는
    기존과 동일하게 "user_interrupt" 다. 그 외의 예기치 못한 예외(is_alive/sleep
    어디서든 날 수 있다)는 "watch_error" 로 잡아 삼킨다 - 그러지 않으면 예외가
    main() 밖으로 새어나가 녹화 스레드가 계속 돌고 manifest 도 안 써지는 채로
    프로세스가 끝난다.

    프레임/디스크 예산은 여기서 재지 않는다 - RecordingSession 이 프레임을 쓰는
    바로 그 자리에서 판정하고 스스로 멈춘다. 5초짜리 감시 주기로 재면 20fps 에서
    최대 100 프레임을 초과 저장한 뒤에야 알아차린다.
    """
    try:
        while session.is_alive():
            time.sleep(settings.watch_interval_sec)
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

    # 사이드카 래퍼는 알람 사이클과 **공용**이다(frame_meta.FrameMetaRecorder).
    # 여기서 포크하면 두 녹화의 사이드카 스키마가 조용히 갈린다.
    meta = (
        FrameMetaRecorder(tool_window, out_dir, our_handles=our_handles)
        if settings.meta_enabled
        else None
    )
    session = RecordingSession(
        tool_window, out_dir, tag=tag,
        poll_sec=settings.poll_sec,
        heartbeat_sec=settings.heartbeat_sec,
        change_min_px=settings.change_min_px,
        max_sec=settings.max_sec,
        max_frames=settings.max_frames,
        max_disk_mb=settings.max_disk_mb,
        jpeg_quality=settings.jpeg_quality,
        capture_fn=(
            meta.wrap(lambda: capture_window(tool_window)) if meta is not None else None
        ),
    )
    session.start()
    print("[INFO] 녹화 중입니다. 중지하려면 Ctrl+C 를 누르세요.")

    stop_reason = _watch_until_stop(session, settings)

    frames = session.stop(stop_reason)
    if meta is not None:
        meta.close()
    print(f"[INFO] ===== 녹화 종료: {len(frames)} 프레임, 사유={session.stop_reason} =====")
    print(f"[INFO] 프레임 경로: {out_dir}")
    print("[INFO] 분석하려면: RECORDING_FILTER_INPUT_DIR=<위 경로> "
          "uv run python poc/workflow_3/recording_filter/filter_recording.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
