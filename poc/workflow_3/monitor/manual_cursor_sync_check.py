"""수동 시험 - tool 창(Remote Monitoring)의 커서가 로컬 마우스와 동기화되어 있는지 본다.

RCS tool 창은 장비 화면을 비추는 뷰라, 창 안에 보이는 커서는 **장비 쪽 커서**다.
동기화가 깨지면(drift) 로컬 포인터와 장비 커서가 서로 다른 자리에 있게 되고, 그 상태로
클릭하면 엉뚱한 곳이 눌린다. 이 스크립트는 클릭 없이 **이동만** 해서 그것을 확인한다.

원리 - "커서 두 개의 거리":
  * 로컬 포인터: 우리가 옮긴 자리라 위치를 **정확히 안다**. mss 캡처에는 로컬 포인터가
    찍히지 않으므로(프레임에 보이는 것은 장비 커서뿐) 이미지에서 찾을 필요가 없다.
  * 장비 커서: 캡처 프레임에서 VLM 이 찾는다(recording_filter 의 커서 프롬프트 재사용).
  probe 점 몇 곳으로 커서를 옮겨 가며 둘의 거리를 재고, 다수결로 판정한다.
    probe 하나는 |dx| <= SYNC_DX 그리고 |dy| <= SYNC_DY 면 sync, 아니면 drift.
    synced   - 찾은 probe 의 과반이 sync
    drifted  - drift 가 sync 보다 많음 (중앙값 offset 과 '장비 커서가 따라왔는가' 를 같이 찍는다)
    unknown  - 장비 커서를 MIN_FOUND 곳 미만에서 찾음 / 표가 갈림

주의: 커서 bbox 중심을 쓰므로 화살표 끝(hotspot)과 ~16-24px 차이가 난다. 축별 허용치
(SYNC_DX/SYNC_DY)가 그 편향을 흡수한다 - 오피스 첫 실행의 동기화 상태 offset 이
dx~45/dy~12 였다(2026-09-18). 커서 탐지는 VLM 이라 고정 그래픽(손바닥 아이콘 등) 오탐이 있을 수
있다 - 판정이 이상하면 산출 폴더의 probe 프레임을 볼 것.

안전: 클릭/휠/키 입력 없음. 물리 마우스만 움직인다(SAFE_MODE=0 필요, 기본값으로 못박음).
긴급 해제 단축키가 뜬다.

사용법: 이 파일 상단 상수를 고친 뒤
  uv run python poc/workflow_3/monitor/manual_cursor_sync_check.py
env override: MANUAL_CURSOR_SYNC_<상수> (EQP_ID / SYNC_DX / SYNC_DY)
"""

import math
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 셸 env(MANUAL_CURSOR_SYNC_<이름>)가 이긴다.
# ===========================================================================
EQP_ID = ""            # 비우면 열려 있는 아무 tool 창
# probe 점 (창 이미지 비율 x, y). 서로 멀리 떨어지게 둬야 '장비 커서가 따라오는가' 가 보인다.
PROBES = [(0.15, 0.25), (0.85, 0.25), (0.50, 0.80)]
SYNC_DX = 50           # |dx| 이 이하 + |dy| 가 SYNC_DY 이하면 같은 자리 (이미지 px, hotspot 편향 포함)
SYNC_DY = 15           # 2026-09-18 오피스: 동기화 상태 offset 이 dx~45, dy~12 (사용자 판정)
MIN_FOUND = 2          # 장비 커서를 최소 이만큼의 probe 에서 찾아야 판정 (3곳 중 2곳)
SETTLE_SEC = 0.8       # 이동 후 원격 화면이 따라올 시간
RECHECK_MAX = 2        # 첫 판독이 sync 범위 밖이면 다시 읽는 횟수 (원격 커서 지연 흡수)
RECHECK_SEC = 1.0      # 재판독 전 대기
VLM_SERVICE = "mai-ui"

EXIT_OK = 0            # synced
EXIT_DRIFTED = 3
EXIT_UNKNOWN = 4
EXIT_PREFLIGHT_FAILED = 2


def is_synced(truth: dict, found: dict, sync_dx: float, sync_dy: float) -> bool:
    """probe 하나: 장비 커서가 로컬 위치의 축별 허용치 안에 있는가."""
    return abs(found["x"] - truth["x"]) <= sync_dx and abs(found["y"] - truth["y"]) <= sync_dy


def judge_sync(probes: list[dict], sync_dx: float, sync_dy: float, min_found: int) -> dict:
    """probe 결과(truth/found 이미지 px)로 동기화 판정을 낸다. 순수 함수.

    probe: {"truth": {"x","y"}, "found": {"x","y"} | None}
    """
    found = [p for p in probes if p.get("found")]
    dists = [math.hypot(p["found"]["x"] - p["truth"]["x"],
                        p["found"]["y"] - p["truth"]["y"]) for p in found]
    n_sync = sum(is_synced(p["truth"], p["found"], sync_dx, sync_dy) for p in found)
    n_drift = len(found) - n_sync
    result = {"n_probes": len(probes), "n_found": len(found),
              "n_sync": n_sync, "n_drift": n_drift,
              "distances_px": [round(d, 1) for d in dists],
              "offset_px": None, "remote_followed": None}
    if found:
        result["offset_px"] = {
            "dx": round(statistics.median(p["found"]["x"] - p["truth"]["x"] for p in found)),
            "dy": round(statistics.median(p["found"]["y"] - p["truth"]["y"] for p in found)),
        }
        # 장비 커서가 움직였는가: 찾은 위치들이 전부 한 점(SYNC_DX 안)에 모여 있으면 안 따라온 것.
        xs = [p["found"]["x"] for p in found]
        ys = [p["found"]["y"] for p in found]
        result["remote_followed"] = (
            len(found) < 2 or max(max(xs) - min(xs), max(ys) - min(ys)) > sync_dx
        )

    if len(found) < min_found:
        result["verdict"] = "unknown"
        result["reason"] = f"장비 커서를 {len(found)}/{len(probes)} 곳에서만 찾음"
    elif n_drift > n_sync:
        result["verdict"] = "drifted"
        result["reason"] = (
            "장비 커서가 로컬 이동을 따라오지 않음" if result["remote_followed"] is False
            else f"장비 커서가 일정 offset {result['offset_px']} 만큼 어긋남"
        )
    elif n_sync > n_drift and n_sync * 2 > len(found):
        result["verdict"] = "synced"
        result["reason"] = f"{n_sync}/{len(found)} probe 가 |dx|<={sync_dx}, |dy|<={sync_dy} 이내"
    else:
        result["verdict"] = "unknown"
        result["reason"] = f"표가 갈림 (sync={n_sync}, drift={n_drift}, found={len(found)})"
    return result


def _arg(name: str, default):
    """셸 env(MANUAL_CURSOR_SYNC_<name>) > 파일 상수. 출처를 콘솔에 남긴다."""
    raw = os.environ.get(f"MANUAL_CURSOR_SYNC_{name}", "").strip()
    if not raw:
        return default
    print(f"[INFO] {name}={raw} (env 가 파일 상수 {default!r} 를 덮음)")
    return type(default)(raw)


def main() -> int:
    """커서 동기화 확인 1회. 종료 코드: 0=synced, 3=drifted, 4=unknown, 2=사전조건 실패."""
    os.environ.setdefault("SAFE_MODE", "0")
    if os.environ["SAFE_MODE"] != "0":
        print("[ERROR] SAFE_MODE=1 이면 커서가 움직이지 않아 판정이 무의미합니다. 종료.")
        return EXIT_PREFLIGHT_FAILED

    from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
    from poc.workflow_3.recording_filter.click_detect import _locate_cursor
    from poc.workflow_3.util import (
        bbox_center, capture_window, image_point_to_screen, make_timestamp_tag,
        move_cursor_to_screen,
    )
    from poc.workflow_3.util.abort_switch import is_aborted, start_abort_hotkey
    from poc.workflow_3.util.event_dir import debug_root
    from poc.workflow_3.util.window_utils import foreground_window
    from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

    eqp_id = _arg("EQP_ID", EQP_ID)
    sync_dx = _arg("SYNC_DX", SYNC_DX)
    sync_dy = _arg("SYNC_DY", SYNC_DY)

    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] tool 창이 없습니다: EQP_ID={eqp_id!r}. 먼저 직접 접속하세요.")
        return EXIT_PREFLIGHT_FAILED
    print(f"[INFO] 대상 tool 창: {title!r}")
    print("[WARNING] 마우스가 움직입니다(클릭 없음). 끝날 때까지 마우스를 만지지 마세요.")

    start_abort_hotkey(os.environ.get("ALIGN_FAIL_ABORT_HOTKEY", "<ctrl>+<alt>+q"))
    if not foreground_window(window, debug_label="cursor_sync"):
        print("[WARNING] tool 창 전면화 실패 - 원격이 이동을 못 받을 수 있습니다. 계속 진행.")

    out_dir = debug_root() / "cursor_sync" / make_timestamp_tag(time.time())
    client = Workflow1VLMClient(service_slug=VLM_SERVICE, log_name="manual_cursor_sync_check")
    image_size = capture_window(window).size
    probes: list[dict] = []
    for idx, (rx, ry) in enumerate(PROBES, start=1):
        if is_aborted():
            print("[WARNING] 긴급 해제됨 - 중단합니다.")
            break
        truth = {"x": int(image_size[0] * rx), "y": int(image_size[1] * ry)}
        screen = image_point_to_screen(window, truth, image_size=image_size)
        if screen is None:
            print(f"[WARNING] probe{idx}: 스크린 좌표 변환 실패 - 건너뜀")
            continue
        move_cursor_to_screen(screen, f"cursor_sync_probe{idx}", action_enabled=True)
        time.sleep(SETTLE_SEC)
        # 원격 커서는 로컬 이동을 한 박자 늦게 따라온다. 첫 판독이 멀면 기다렸다 다시 읽고
        # 마지막 판독을 쓴다 - 지연이면 가까워지고, 진짜 drift 면 같은 자리에 머문다.
        readings = []
        for attempt in range(1 + RECHECK_MAX):
            if attempt:
                time.sleep(RECHECK_SEC)
            frame_path = out_dir / f"probe{idx}_r{attempt}.jpg"
            save_debug_jpeg(capture_window(window), frame_path)
            try:
                parsed, bbox, _w, _h = _locate_cursor(client, frame_path)
            except Exception as exc:
                print(f"[WARNING] probe{idx} r{attempt}: 커서 탐지 실패 - {exc}")
                parsed, bbox = {}, None
            found = bbox_center(bbox) if bbox else None
            dist = math.hypot(found["x"] - truth["x"], found["y"] - truth["y"]) if found else None
            delta = f"dx={found['x'] - truth['x']:+d} dy={found['y'] - truth['y']:+d}" if found else ""
            size = f"bbox={bbox['right'] - bbox['left']}x{bbox['bottom'] - bbox['top']}" if bbox else ""
            print(f"[INFO] probe{idx} r{attempt}: local={truth} remote={found} "
                  f"dist={'n/a' if dist is None else f'{dist:.0f}px'} {delta} {size} "
                  f"kind={parsed.get('cursor_kind')}")
            readings.append({"found": found, "bbox": bbox, "kind": parsed.get("cursor_kind"),
                             "frame": str(frame_path)})
            if found and is_synced(truth, found, sync_dx, sync_dy):
                break
        probes.append({"truth": truth, "found": readings[-1]["found"], "readings": readings})

    result = judge_sync(probes, sync_dx, sync_dy, MIN_FOUND)
    save_debug_json(out_dir / "result.json", {"title": title, "probes": probes, **result})
    print("=" * 70)
    print(f"[DIGEST] cursor_sync verdict={result['verdict']} found={result['n_found']}/"
          f"{result['n_probes']} sync={result['n_sync']} drift={result['n_drift']} "
          f"offset={result['offset_px']} followed={result['remote_followed']}")
    print(f"[DIGEST] reason: {result['reason']}")
    print(f"[DIGEST] artifacts: {out_dir}")
    print("=" * 70)
    return {"synced": EXIT_OK, "drifted": EXIT_DRIFTED}.get(result["verdict"], EXIT_UNKNOWN)


if __name__ == "__main__":
    raise SystemExit(main())
