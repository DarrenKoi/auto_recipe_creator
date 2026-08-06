"""tool row 클릭점 로케이터의 VLM 조합 A/B 벤치 - 알람 없이, 클릭 없이 돌린다.

목적: coarse->fine 2단계에 정말 서로 다른 모델 두 개가 필요한지 재는 것. 한 모델만
써도 같은 정확도가 나오면 이 프로젝트에서 VLM 하나를 뺄 수 있다.

측정 방식:
  1. RCS 메인 창(List 탭)을 **한 번만** 캡처한다. 모든 조합이 같은 프레임을 본다
     (화면 드리프트가 모델 차이로 둔갑하지 않게 하는 게 A/B 의 핵심).
  2. 조합 x tool x 반복 마다 production 과 **같은 함수**(`_locate_tool_via_vlm`)를
     돌려 클릭점을 얻는다. 확인 게이트는 꺼서(confirm_policy=off) 모델의 raw 출력을
     그대로 잰다.
  3. 채점은 `verify_tool_row_at_point` (좁은 strip OCR)로 한다. 그 지점 텍스트가
     목표 ID 면 correct, 다른 ID 면 wrong_row, 못 읽으면 unreadable.
  4. 반복 사이 변동(run-to-run)까지 세서, 정확도가 같아도 불안정한 조합을 가려낸다.

**클릭하지 않는다.** 캡처와 VLM 호출뿐이라 장비에 영향이 없고 알람을 기다릴 필요도
없다. RCS 가 로그인돼 List 탭이 보이는 상태이기만 하면 된다.

채점 오라클(strip OCR)이 틀릴 수 있으므로 조합/tool/반복마다 overlay 이미지를 남긴다.
숫자가 이상하면 overlay 를 눈으로 확인하면 된다.

설정 (모두 env, CLI 인자 없음):
  BENCH_TOOL_NAMES    쉼표 구분 tool ID 목록 (기본 MCDN01,MCDA01,MDSK10)
  BENCH_COMBOS        쉼표 구분 "coarse>fine" 조합 (기본 아래 DEFAULT_COMBOS)
  BENCH_REPEATS       조합x tool 당 반복 횟수 (기본 3)

저장된 결과로 digest 만 다시 보고 싶으면 아래 `REPLAY_RESULT_JSON` 상수에 결과 JSON
경로를 적고 실행한다 (측정/RCS/VLM 불필요). 다 보고 나면 다시 "" 로 비운다.

사용법:
  uv run python poc/workflow_3/rcs/bench_tool_locator.py
  BENCH_TOOL_NAMES=MCDN01,MCDA01 BENCH_REPEATS=5 uv run python poc/workflow_3/rcs/bench_tool_locator.py

주의: 호출 수 = 조합 x tool x 반복 x (VLM 2 + OCR 1). 기본값(4조합 x 12 tool x 3반복)
이면 144 run / 약 432 콜이라 20~30분 걸린다. 순서를 권한다:
  1) BENCH_REPEATS=1 로 스모크 (48 run) - 경로가 돌고 tool 이 화면에 보이는지 확인
  2) 그대로 기본값(3반복)으로 본 측정 - 반복이 있어야 run-to-run 흔들림이 보인다

**한 프레임만 캡처하므로, 스크롤해야 보이는 tool 은 어떤 조합도 못 찾는다.** 그런
tool 은 모델 실패가 아니라 화면 밖이라, digest 가 `unresolved` 로 따로 표시하고
'보이는 tool 만' 기준 정확도를 함께 낸다. unresolved 가 많으면 창을 키우거나
BENCH_TOOL_NAMES 를 한 화면에 보이는 것들로 좁혀 다시 돌린다.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json, save_marked_bboxes
from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_OFF,
    verify_tool_row_at_point,
)
from poc.workflow_3.rcs.workflow_select_tool import (
    _capture_main_window,
    _locate_tool_via_vlm,
)
from poc.workflow_3.util import (
    is_window_maximized,
    make_timestamp_tag,
    maximize_window,
    point_to_tiny_bbox,
)

load_dotenv()

LOG_NAME = "bench_tool_locator"
COMPONENT_NAME = LOG_NAME
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "bench_tool_locator"

# (coarse, fine). 같은 모델을 두 번 쓰는 조합이 핵심 비교 대상이다.
DEFAULT_COMBOS = [
    ("ui-venus", "ui-venus"),
    ("mai-ui", "mai-ui"),
    ("ui-venus", "mai-ui"),  # 현재 production 설정
    ("mai-ui", "ui-venus"),
]
PRODUCTION_COMBO = ("ui-venus", "mai-ui")

# 여기에 결과 JSON 경로를 적으면 **측정하지 않고** 그 결과로 digest 만 다시 출력한다.
# digest 형식이 바뀌었을 때 20~30분짜리 벤치를 다시 돌리지 않으려는 용도.
# 예: "poc/workflow_3/debug_images/bench_tool_locator/260806_1500_bench_result.json"
REPLAY_RESULT_JSON = ""
# 실제 List 탭 장비 ID. 길이(6/8자)와 접두(MCD/CCDM/RKHV/숫자 시작)가 섞여 있어야
# 로케이터가 특정 ID 모양에만 강한 것인지 드러난다. MCDN01/MCDN02, MCDC12/MCDC22
# 처럼 **한 글자만 다른 쌍**이 옆 행 오클릭을 잡아내는 핵심 케이스다.
DEFAULT_TOOL_NAMES = [
    "4DCDB807",
    "6MCD3401",
    "6PCDL101",
    "CCDM21",
    "MCD427",
    "MCD717",
    "MCDC12",
    "MCDC22",
    "MCDN01",
    "MCDN02",
    "RKHV3101",
    "RMHV3301",
]

# 채점 결과 코드.
SCORE_CORRECT = "correct"
SCORE_WRONG_ROW = "wrong_row"
SCORE_UNREADABLE = "unreadable"
SCORE_NO_DETECT = "no_detect"
SCORE_ERROR = "error"


def _load_tool_names() -> list[str]:
    """BENCH_TOOL_NAMES 에서 대상 tool 목록을 읽는다."""
    raw = os.getenv("BENCH_TOOL_NAMES", "").strip()
    if not raw:
        return list(DEFAULT_TOOL_NAMES)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return names or list(DEFAULT_TOOL_NAMES)


def _load_combos() -> list[tuple[str, str]]:
    """BENCH_COMBOS ("coarse>fine,coarse>fine") 에서 조합 목록을 읽는다."""
    raw = os.getenv("BENCH_COMBOS", "").strip()
    if not raw:
        return list(DEFAULT_COMBOS)

    combos: list[tuple[str, str]] = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ">" not in chunk:
            print(f"[WARNING] BENCH_COMBOS 항목 {chunk!r} 에 '>' 가 없어 건너뜁니다.")
            continue
        coarse, fine = (piece.strip() for piece in chunk.split(">", 1))
        if not coarse or not fine:
            print(f"[WARNING] BENCH_COMBOS 항목 {chunk!r} 파싱 실패 - 건너뜁니다.")
            continue
        combos.append((coarse, fine))
    return combos or list(DEFAULT_COMBOS)


def _load_repeats() -> int:
    """BENCH_REPEATS 에서 반복 횟수를 읽는다."""
    raw = os.getenv("BENCH_REPEATS", "").strip()
    if not raw:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        print(f"[WARNING] BENCH_REPEATS={raw!r} 파싱 실패 -> 3 사용")
        return 3


def _combo_label(combo: tuple[str, str]) -> str:
    """조합을 로그/파일명용 라벨로 만든다."""
    return f"{combo[0]}__{combo[1]}"


def _save_run_overlay(
    image,
    point: dict | None,
    strip_box: dict | None,
    *,
    timestamp_tag: str,
    filename: str,
) -> str:
    """클릭점과 채점에 쓴 strip 을 원본 프레임 위에 그려 저장한다."""
    if point is None:
        return ""
    img_w, img_h = image.size
    elements = {
        "click_point": {
            "bbox": point_to_tiny_bbox(point, img_w, img_h),
            "center": point,
        },
    }
    colors = {"click_point": "deepskyblue"}
    if strip_box is not None:
        elements["ocr_strip"] = {"bbox": strip_box}
        colors["ocr_strip"] = "yellow"

    out_path = DEBUG_ARTIFACT_DIR / f"{timestamp_tag}_{filename}"
    save_marked_bboxes(image, elements, colors, out_path)
    return str(out_path)


def _score_point(
    image,
    point: dict | None,
    tool_name: str,
    *,
    timestamp_tag: str,
    artifact_label: str,
) -> tuple[str, dict]:
    """클릭점을 strip OCR 로 채점한다. (score, detail) 반환."""
    if point is None:
        return SCORE_NO_DETECT, {"read_text": "", "strip_box": None}

    verdict = verify_tool_row_at_point(
        image,
        point,
        tool_name,
        debug_image_dir=DEBUG_ARTIFACT_DIR,
        timestamp_tag=timestamp_tag,
        log_name=LOG_NAME,
        component_name=COMPONENT_NAME,
        artifact_label=artifact_label,
    )
    score_map = {
        "confirmed": SCORE_CORRECT,
        "mismatch": SCORE_WRONG_ROW,
        "unreadable": SCORE_UNREADABLE,
        "error": SCORE_ERROR,
    }
    detail = {
        "read_text": verdict.read_text,
        "mismatch_token": verdict.mismatch_token,
        "strip_box": verdict.strip_box,
    }
    return score_map.get(verdict.status, SCORE_ERROR), detail


def _run_one(
    main_window,
    window_title: str,
    backend: str,
    image,
    tool_name: str,
    combo: tuple[str, str],
    repeat_idx: int,
) -> dict:
    """조합 1개 x tool 1개 x 반복 1회를 돌리고 채점 결과를 만든다."""
    label = _combo_label(combo)
    timestamp_tag = make_timestamp_tag(time.time())
    artifact_label = f"{label}_{tool_name}_r{repeat_idx}"

    started_at = time.time()
    located, attempt_record = _locate_tool_via_vlm(
        main_window,
        window_title,
        backend,
        tool_name,
        image,
        debug_image_dir=DEBUG_ARTIFACT_DIR,
        log_name=LOG_NAME,
        component_name=COMPONENT_NAME,
        timestamp_tag=timestamp_tag,
        coarse_service_slug=combo[0],
        refine_service_slug=combo[1],
        # 확인 게이트는 끈다 - 모델 raw 출력을 재고, 채점은 아래에서 따로 한다.
        confirm_policy=CONFIRM_POLICY_OFF,
    )
    locate_sec = time.time() - started_at

    point = located["full_image_point"] if located else None
    score, detail = _score_point(
        image,
        point,
        tool_name,
        timestamp_tag=timestamp_tag,
        artifact_label=artifact_label,
    )
    overlay_path = _save_run_overlay(
        image,
        point,
        detail.get("strip_box"),
        timestamp_tag=timestamp_tag,
        filename=f"{artifact_label}_overlay.jpg",
    )

    print(
        f"[INFO] {label:24s} tool={tool_name} r{repeat_idx} -> {score:10s} "
        f"point={point} read={detail.get('read_text', '')[:24]!r} "
        f"({locate_sec:.1f}s)"
    )
    return {
        "combo": label,
        "coarse_service": combo[0],
        "refine_service": combo[1],
        "tool_name": tool_name,
        "repeat": repeat_idx,
        "score": score,
        "point": point,
        "locate_sec": round(locate_sec, 2),
        "read_text": detail.get("read_text", ""),
        "mismatch_token": detail.get("mismatch_token", ""),
        "overlay_path": overlay_path,
        "iters": attempt_record.get("iters", []),
    }


def _stability(rows: list[dict]) -> float:
    """같은 조합/tool 안에서 반복 결과가 얼마나 일관적인지 (0~1).

    반복마다 점수가 뒤집히는 조합은 평균 정확도가 같아도 실전에서 간헐 실패를 낸다.
    tool 별로 '가장 흔한 점수'가 차지하는 비율을 내고, 그것을 tool 평균으로 낸다.
    """
    by_tool: dict[str, list[str]] = {}
    for row in rows:
        by_tool.setdefault(row["tool_name"], []).append(row["score"])

    ratios: list[float] = []
    for scores in by_tool.values():
        if not scores:
            continue
        top = max(set(scores), key=scores.count)
        ratios.append(scores.count(top) / len(scores))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _unresolved_tools(results: list[dict], tool_names: list[str]) -> list[str]:
    """어떤 조합에서도 한 번도 correct 가 안 나온 tool.

    보통은 그 tool 이 캡처한 한 프레임 안에 없다는 뜻이다(스크롤해야 보임). 모델 탓이
    아니므로 정확도 해석에서 분리해야 한다 - 조용히 빼면 '전부 커버했다'로 읽힌다.
    """
    resolved = {row["tool_name"] for row in results if row["score"] == SCORE_CORRECT}
    return [name for name in tool_names if name not in resolved]


def _hard_tools(results: list[dict]) -> list[tuple[str, int]]:
    """wrong_row(옆 행 클릭)가 한 번이라도 난 tool 을 많은 순으로."""
    counts: dict[str, int] = {}
    for row in results:
        if row["score"] == SCORE_WRONG_ROW:
            counts[row["tool_name"]] = counts.get(row["tool_name"], 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1])


def _summarize(
    results: list[dict],
    combos: list[tuple[str, str]],
    unresolved: list[str],
) -> list[dict]:
    """조합별 집계를 만든다 (보이는 tool 기준 정확도 내림차순)."""
    unresolved_set = set(unresolved)
    summary: list[dict] = []
    for combo in combos:
        label = _combo_label(combo)
        rows = [row for row in results if row["combo"] == label]
        if not rows:
            continue
        total = len(rows)
        counts = {
            score: sum(1 for row in rows if row["score"] == score)
            for score in (
                SCORE_CORRECT,
                SCORE_WRONG_ROW,
                SCORE_UNREADABLE,
                SCORE_NO_DETECT,
                SCORE_ERROR,
            )
        }
        latencies = [row["locate_sec"] for row in rows]

        # 화면 밖 tool 을 뺀 기준 - 모델 비교는 이 값으로 한다.
        visible_rows = [row for row in rows if row["tool_name"] not in unresolved_set]
        visible_total = len(visible_rows)
        visible_correct = sum(1 for row in visible_rows if row["score"] == SCORE_CORRECT)

        summary.append(
            {
                "combo": label,
                "coarse_service": combo[0],
                "refine_service": combo[1],
                "single_model": combo[0] == combo[1],
                "is_production": tuple(combo) == PRODUCTION_COMBO,
                "total": total,
                "accuracy_all": round(counts[SCORE_CORRECT] / total, 3),
                "visible_total": visible_total,
                "accuracy": round(visible_correct / visible_total, 3) if visible_total else 0.0,
                "wrong_row_rate": round(counts[SCORE_WRONG_ROW] / total, 3),
                "stability": round(_stability(visible_rows or rows), 3),
                "median_sec": round(statistics.median(latencies), 2),
                **{f"n_{key}": value for key, value in counts.items()},
            }
        )
    summary.sort(key=lambda item: (-item["accuracy"], item["wrong_row_rate"], item["median_sec"]))
    return summary


def _print_digest(
    summary: list[dict],
    tool_names: list[str],
    repeats: int,
    unresolved: list[str],
    hard_tools: list[tuple[str, int]],
) -> None:
    """오피스에서 그대로 복사해 올 수 있는 한 덩어리 digest 를 출력한다."""
    visible_count = len(tool_names) - len(unresolved)
    print("")
    print("[DIGEST] ===== tool locator VLM combo bench =====")
    print(f"[DIGEST] tools={len(tool_names)} visible={visible_count} repeats={repeats}")
    print(f"[DIGEST] tool_names={','.join(tool_names)}")
    if unresolved:
        print(
            f"[DIGEST] unresolved(어떤 조합도 못 찾음, 화면 밖 추정)={','.join(unresolved)}"
        )
        print("[DIGEST] -> acc 는 보이는 tool 기준. unresolved 는 창 확대/목록 축소 후 재측정.")
    # 실패를 항상 종류별로 쪼개 보여준다. 합계만 보면 '옆 행 클릭(wrong)'과 '아무것도
    # 못 찾음(nodet)' 이 같은 점수로 뭉뚱그려지는데, 둘은 성질이 완전히 다르다:
    # wrong = 엉뚱한 tool 접속(위험), nodet = 재시도로 회복(양호),
    # unread = 채점 OCR 이 못 읽음(모델 탓이 아닐 수 있음 - overlay 확인 필요).
    print(
        f"[DIGEST] {'combo':26s} {'acc':>6s} {'stab':>6s} {'med_s':>6s} "
        f"{'ok':>4s} {'wrong':>5s} {'unread':>6s} {'nodet':>5s} {'err':>4s}"
    )
    for item in summary:
        marker = " *prod" if item["is_production"] else ("  1vlm" if item["single_model"] else "")
        print(
            f"[DIGEST] {item['combo']:26s} {item['accuracy']:6.3f} "
            f"{item['stability']:6.3f} {item['median_sec']:6.2f} "
            f"{item['n_' + SCORE_CORRECT]:4d} {item['n_' + SCORE_WRONG_ROW]:5d} "
            f"{item['n_' + SCORE_UNREADABLE]:6d} {item['n_' + SCORE_NO_DETECT]:5d} "
            f"{item['n_' + SCORE_ERROR]:4d}{marker}"
        )

    if hard_tools:
        detail = ", ".join(f"{name}x{count}" for name, count in hard_tools)
        print(f"[DIGEST] wrong_row 발생 tool: {detail}")
        print("[DIGEST] -> 해당 overlay 를 열어 어느 행을 눌렀는지 확인 (혼동 쌍 후보)")
    else:
        print("[DIGEST] wrong_row 없음 - 옆 행 클릭은 한 건도 발생하지 않았다.")
        print("[DIGEST] -> 실패는 전부 nodet(미검출)/unread(채점 OCR 판독 실패)다.")
        print("[DIGEST] -> nodet 는 production 에서 재시도로 회복되는 실패다(오접속 아님).")

    if not summary:
        print("[DIGEST] verdict: 결과 없음")
        return

    best = summary[0]
    prod = next((item for item in summary if item["is_production"]), None)
    best_single = next((item for item in summary if item["single_model"]), None)

    print(f"[DIGEST] best={best['combo']} acc={best['accuracy']:.3f}")
    if prod is not None:
        print(f"[DIGEST] prod={prod['combo']} acc={prod['accuracy']:.3f}")
    if best_single is not None and prod is not None:
        delta = best_single["accuracy"] - prod["accuracy"]
        print(
            f"[DIGEST] best_single={best_single['combo']} acc={best_single['accuracy']:.3f} "
            f"delta_vs_prod={delta:+.3f}"
        )
        if delta >= 0:
            print(
                f"[DIGEST] verdict: 단일 모델 {best_single['coarse_service']} 이 production "
                f"2모델 조합 이상 - VLM 한 개 제거 검토 가능"
            )
        else:
            print(
                f"[DIGEST] verdict: 2모델 조합이 단일 모델보다 {abs(delta):.3f} 우위 - 두 모델 유지"
            )
    print("[DIGEST] ==========================================")


def _reprint_from_json(path_text: str) -> str:
    """이미 저장된 결과 JSON 으로 digest 만 다시 출력한다 (재측정 없음).

    digest 형식이 바뀌었을 때 20~30분짜리 벤치를 다시 돌리지 않아도 되도록 한다.
    RCS 도 VLM 도 필요 없으므로 어느 PC 에서든 돈다.
    """
    path = Path(path_text).expanduser()
    if not path.is_file():
        print(f"[ERROR] 결과 JSON 을 찾을 수 없습니다: {path}")
        return "result_json_not_found"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] 결과 JSON 파싱 실패: {type(exc).__name__}: {exc}")
        return "result_json_parse_failed"

    results = payload.get("runs") or []
    if not results:
        print(f"[ERROR] runs 가 비어 있습니다: {path}")
        return "result_json_empty"

    tool_names = payload.get("tool_names") or sorted({row["tool_name"] for row in results})
    repeats = payload.get("repeats", 0)
    combos: list[tuple[str, str]] = []
    for row in results:
        combo = (row.get("coarse_service", ""), row.get("refine_service", ""))
        if all(combo) and combo not in combos:
            combos.append(combo)
    if not combos:
        print("[ERROR] runs 에 coarse/refine 서비스 정보가 없어 조합을 복원할 수 없습니다.")
        return "result_json_no_combos"

    print(f"[INFO] 저장된 결과로 digest 재출력: {path}")
    unresolved = _unresolved_tools(results, tool_names)
    hard_tools = _hard_tools(results)
    _print_digest(_summarize(results, combos, unresolved), tool_names, repeats, unresolved, hard_tools)
    return "success"


def main() -> str:
    """List 탭 프레임 1장을 캡처해 모든 VLM 조합을 채점한다.

    REPLAY_RESULT_JSON 상수가 비어 있지 않으면 측정하지 않고 그 결과로 digest 만 낸다.
    """
    if REPLAY_RESULT_JSON.strip():
        return _reprint_from_json(REPLAY_RESULT_JSON.strip())

    tool_names = _load_tool_names()
    combos = _load_combos()
    repeats = _load_repeats()
    total_runs = len(combos) * len(tool_names) * repeats

    print(
        f"[INFO] bench 시작: combos={len(combos)} tools={len(tool_names)} "
        f"repeats={repeats} -> {total_runs} runs (VLM 콜 약 {total_runs * 3})"
    )
    print(f"[INFO] combos={[_combo_label(combo) for combo in combos]}")
    print(f"[INFO] tools={tool_names}")
    print("[INFO] 클릭은 하지 않습니다 (캡처 + VLM 호출만).")

    main_window, window_title, backend = wait_for_rcs_main_window(timeout_sec=15.0)
    if main_window is None:
        print("[ERROR] RCS 메인 창을 찾지 못했습니다. 로그인 후 List 탭을 띄우고 다시 실행하세요.")
        return "main_window_not_found"

    if not is_window_maximized(main_window):
        print("[INFO] 메인 창 최대화 - 한 프레임에 더 많은 행이 보이게 합니다.")
        maximize_window(main_window, debug_label="bench_tool_locator_maximize")

    image = _capture_main_window(main_window, window_title, backend)
    if image is None:
        print("[ERROR] 메인 창 캡처 실패")
        return "capture_failed"

    run_tag = make_timestamp_tag(time.time())
    save_debug_jpeg(image, DEBUG_ARTIFACT_DIR / f"{run_tag}_bench_frame.jpg")
    print(f"[INFO] 기준 프레임 캡처 완료: {image.size} (모든 조합이 이 프레임을 공유)")

    results: list[dict] = []
    for combo in combos:
        for tool_name in tool_names:
            for repeat_idx in range(1, repeats + 1):
                try:
                    results.append(
                        _run_one(
                            main_window,
                            window_title,
                            backend,
                            image,
                            tool_name,
                            combo,
                            repeat_idx,
                        )
                    )
                except Exception as exc:
                    print(
                        f"[ERROR] run 실패: combo={_combo_label(combo)} tool={tool_name} "
                        f"r{repeat_idx}: {type(exc).__name__}: {exc}"
                    )
                    results.append(
                        {
                            "combo": _combo_label(combo),
                            "coarse_service": combo[0],
                            "refine_service": combo[1],
                            "tool_name": tool_name,
                            "repeat": repeat_idx,
                            "score": SCORE_ERROR,
                            "point": None,
                            "locate_sec": 0.0,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    unresolved = _unresolved_tools(results, tool_names)
    hard_tools = _hard_tools(results)
    summary = _summarize(results, combos, unresolved)
    result_path = DEBUG_ARTIFACT_DIR / f"{run_tag}_bench_result.json"
    save_debug_json(
        result_path,
        {
            "window_title": window_title,
            "backend": backend,
            "frame_size": {"width": image.size[0], "height": image.size[1]},
            "tool_names": tool_names,
            "combos": [_combo_label(combo) for combo in combos],
            "repeats": repeats,
            "unresolved_tools": unresolved,
            "hard_tools": [{"tool_name": name, "wrong_row": count} for name, count in hard_tools],
            "summary": summary,
            "runs": results,
        },
    )
    _print_digest(summary, tool_names, repeats, unresolved, hard_tools)
    print(f"[INFO] 상세 결과: {result_path}")
    print(f"[INFO] overlay/strip 이미지: {DEBUG_ARTIFACT_DIR}")
    return "success"


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
