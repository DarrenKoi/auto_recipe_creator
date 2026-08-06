"""벤치의 미검출(no_detect)이 **어느 단계에서** 났는지 집계한다.

digest 의 `nodet` 은 "VLM 이 좌표를 아예 안 냈다" 는 뜻인데, 그게 coarse 단계 거부인지
fine 단계 거부인지는 digest 로 알 수 없다. 둘은 의미가 다르다:

  ui_venus (coarse) 거부 - bbox 를 못/안 냈다. crop 이 없으니 fine 모델은 호출조차
                           되지 않는다. => coarse 모델이 약한 고리다.
  mai_ui  (fine) 거부    - crop 은 만들어졌는데 그 안에서 점을 못 찍었다.
                           => crop 범위나 fine 모델 문제다.

`analyze_window_target` 이 실패할 때마다 `*_ui_venus_mai_result.json` 에 남기는
`failure_stage` 필드를 조합 디렉터리별로 세어 준다. 이미 저장된 artifact 만 읽으므로
**재측정도 VLM 호출도 없다** - 어느 PC 에서든 돈다.

읽는 위치: `debug_images/<BENCH_DIR_NAME>/<coarse_model>__<refine_model>/*.json`
(파일이 조합별 모델 디렉터리로 나뉘어 저장되기 때문에 조합별 집계가 가능하다.)

사용법:
  uv run python poc/workflow_3/rcs/bench_stage_report.py

tool 창 벤치를 보려면 아래 BENCH_DIR_NAME 을 "bench_tool_window_reader" 로 바꾼다.
"""

import json
import sys
from collections import Counter

from poc.workflow_3 import DEBUG_IMAGE_DIR

# 볼 벤치 디렉터리 - "bench_tool_locator" (List 탭) 또는 "bench_tool_window_reader" (tool 창).
BENCH_DIR_NAME = "bench_tool_locator"

RESULT_GLOB = "*_ui_venus_mai_result.json"


def main() -> str:
    """조합별 failure_stage 분포를 출력한다."""
    root = DEBUG_IMAGE_DIR / BENCH_DIR_NAME
    if not root.is_dir():
        print(f"[ERROR] 벤치 artifact 디렉터리가 없습니다: {root}")
        print("[INFO] 해당 벤치를 먼저 1회 실행하세요.")
        return "artifact_dir_not_found"

    files = sorted(root.rglob(RESULT_GLOB))
    print(f"[INFO] root={root}")
    print(f"[INFO] result json={len(files)}개")
    if not files:
        print("[WARNING] 집계할 result json 이 없습니다.")
        return "no_result_json"

    by_combo: dict[str, Counter] = {}
    unreadable = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            unreadable += 1
            continue
        # 파일은 <coarse_model>__<refine_model> 디렉터리 아래에 저장된다.
        combo = path.parent.name
        stage = data.get("failure_stage") or "ok"
        by_combo.setdefault(combo, Counter())[stage] += 1

    print("")
    print("[DIGEST] ===== bench stage attribution =====")
    print(f"[DIGEST] bench={BENCH_DIR_NAME}")
    print(f"[DIGEST] {'combo(model dir)':38s} {'n':>5s}  분포")
    for combo, counter in sorted(by_combo.items()):
        total = sum(counter.values())
        detail = "  ".join(f"{key}={value}" for key, value in counter.most_common())
        print(f"[DIGEST] {combo:38s} {total:5d}  {detail}")

    coarse_refusals = sum(counter.get("ui_venus", 0) for counter in by_combo.values())
    fine_refusals = sum(counter.get("mai_ui", 0) for counter in by_combo.values())
    print(f"[DIGEST] 합계: coarse 거부={coarse_refusals}  fine 거부={fine_refusals}")
    if coarse_refusals > fine_refusals:
        print("[DIGEST] -> coarse 단계 거부가 지배적: coarse 모델이 약한 고리다.")
    elif fine_refusals > coarse_refusals:
        print("[DIGEST] -> fine 단계 거부가 지배적: crop 범위/fine 모델을 보라.")
    elif coarse_refusals or fine_refusals:
        print("[DIGEST] -> coarse/fine 거부가 비슷하다.")
    else:
        print("[DIGEST] -> 거부 없음 (모든 호출이 좌표를 냈다).")
    if unreadable:
        print(f"[DIGEST] 파싱 실패 파일={unreadable}개 (집계 제외)")
    print("[DIGEST] ===================================")
    return "success"


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
