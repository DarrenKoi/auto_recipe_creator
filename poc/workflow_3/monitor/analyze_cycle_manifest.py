"""알람 사이클 manifest 를 집계해 '접속이 어디서 실패하는가' 를 보여준다.

`align_fail_monitor` 는 알람 1건마다 `logs/align_fail_cycles.csv` 에 한 줄을 남긴다
(run_status / failed_step / failure_class ...). 이 스크립트는 그 파일을 읽어 실패
분포를 낸다. **추측 대신 실제 기록으로** 다음에 고칠 지점을 정하기 위한 도구다.

왜 필요한가: tool 접속 실패는 원인이 여러 가지인데 증상이 비슷하다.
  connect_not_clicked  - 로케이터가 tool 을 못 찾아 더블클릭 자체를 안 함
                         (화면 밖 -> 스크롤 실패 / VLM 미검출)
  wrong_tool_opened    - 옆 행을 눌러 다른 tool 창이 열림 (오클릭)
  rcs_occupied         - 더블클릭은 했는데 목표 창이 안 뜸 (타인 점유 추정)
  rcs_occupied_select  - 점유 'select' 팝업을 조기 감지
  connect_error        - 예외
어느 것이 지배적인지에 따라 고칠 곳이 완전히 달라진다.

사용법:
  uv run python poc/workflow_3/monitor/analyze_cycle_manifest.py
  CYCLE_MANIFEST_CSV=<경로> uv run python poc/workflow_3/monitor/analyze_cycle_manifest.py
  CYCLE_MANIFEST_RECENT=20 uv run python poc/workflow_3/monitor/analyze_cycle_manifest.py
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

from poc.workflow_3 import LOG_DIR

DEFAULT_MANIFEST_PATH = LOG_DIR / "align_fail_cycles.csv"

# 접속 단계에서 나오는 failure_class - 사람이 읽을 설명.
FAILURE_HINTS = {
    "connect_not_clicked": "로케이터가 tool 을 못 찾아 더블클릭 안 함 (화면 밖/스크롤/VLM 미검출)",
    "connect_error": "접속 중 예외",
    "wrong_tool_opened": "옆 행 클릭 - 다른 tool 창이 열림 (오클릭)",
    "rcs_occupied": "더블클릭 후 목표 창 미출현 (타인 점유 추정)",
    "rcs_occupied_select": "점유 'select' 팝업 조기 감지",
}


def _load_rows(path: Path) -> list[dict]:
    """manifest CSV 를 dict 목록으로 읽는다."""
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _counter_lines(counter: Counter, total: int, indent: str = "  ") -> list[str]:
    """카운터를 '값 n (xx.x%)' 줄 목록으로."""
    lines: list[str] = []
    for key, count in counter.most_common():
        share = (count / total * 100) if total else 0.0
        label = key or "(빈값)"
        hint = FAILURE_HINTS.get(key, "")
        suffix = f"  <- {hint}" if hint else ""
        lines.append(f"{indent}{label:24s} {count:5d} ({share:5.1f}%){suffix}")
    return lines


def main() -> str:
    """manifest 를 읽어 실패 분포 digest 를 출력한다."""
    path_text = os.getenv("CYCLE_MANIFEST_CSV", "").strip()
    path = Path(path_text).expanduser() if path_text else DEFAULT_MANIFEST_PATH
    recent_n = int(os.getenv("CYCLE_MANIFEST_RECENT", "10") or "10")

    if not path.is_file():
        print(f"[ERROR] manifest 가 없습니다: {path}")
        print("[INFO] align_fail_monitor 를 최소 1회 돌린 뒤 다시 실행하세요.")
        return "manifest_not_found"

    try:
        rows = _load_rows(path)
    except Exception as exc:
        print(f"[ERROR] manifest 읽기 실패: {type(exc).__name__}: {exc}")
        return "manifest_read_failed"

    if not rows:
        print(f"[WARNING] manifest 가 비어 있습니다: {path}")
        return "manifest_empty"

    total = len(rows)
    failures = [row for row in rows if (row.get("failure_class") or "").strip()]
    status_counter = Counter((row.get("run_status") or "").strip() for row in rows)
    failure_counter = Counter((row.get("failure_class") or "").strip() for row in failures)
    step_counter = Counter((row.get("failed_step") or "").strip() for row in failures)
    eqp_counter = Counter((row.get("eqp_id") or "").strip() for row in failures)

    print("")
    print("[DIGEST] ===== align fail cycle manifest =====")
    print(f"[DIGEST] file={path}")
    print(f"[DIGEST] 총 사이클={total}  실패(failure_class 있음)={len(failures)}")
    print("[DIGEST] run_status:")
    for line in _counter_lines(status_counter, total):
        print(f"[DIGEST] {line}")

    if failures:
        print("[DIGEST] failure_class (실패 사이클 기준):")
        for line in _counter_lines(failure_counter, len(failures)):
            print(f"[DIGEST] {line}")
        print("[DIGEST] failed_step:")
        for line in _counter_lines(step_counter, len(failures)):
            print(f"[DIGEST] {line}")

        top_eqp = eqp_counter.most_common(5)
        if top_eqp:
            detail = ", ".join(f"{name or '(빈값)'}x{count}" for name, count in top_eqp)
            print(f"[DIGEST] 실패 많은 EQP: {detail}")

        dominant, dominant_count = failure_counter.most_common(1)[0]
        share = dominant_count / len(failures) * 100
        print(
            f"[DIGEST] 지배적 실패={dominant or '(빈값)'} "
            f"{dominant_count}/{len(failures)} ({share:.1f}%)"
        )
        hint = FAILURE_HINTS.get(dominant)
        if hint:
            print(f"[DIGEST] -> {hint}")
    else:
        print("[DIGEST] 실패 사이클 없음.")

    if recent_n > 0:
        print(f"[DIGEST] 최근 {min(recent_n, total)}건:")
        for row in rows[-recent_n:]:
            print(
                f"[DIGEST]   {row.get('detected_at','')} {row.get('eqp_id',''):10s} "
                f"status={row.get('run_status',''):12s} step={row.get('failed_step',''):20s} "
                f"class={row.get('failure_class','')}"
            )
    print("[DIGEST] =====================================")
    return "success"


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
