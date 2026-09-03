"""열린 tool 창(Remote Monitoring)에서 VLM 조합 A/B - 버튼 찾기 + 커서 추적.

`bench_tool_locator.py` 는 **List 탭**(빽빽한 텍스트 표)에서 행을 고르는 능력을 쟀다.
이 벤치는 **tool 창**(계기 UI)에서의 능력을 잰다 - 다른 문제다. 표에서 행을 구분하는
것과, 서로 다르게 생긴 위젯들 사이에서 라벨 달린 버튼을 찾는 것은 요구 능력이 다르므로
list 결과를 그대로 옮겨 적용하면 안 된다.

두 축(arm):

  [buttons]  Stop / Queue / PM 같은 라벨 버튼을 coarse->fine 조합으로 찾는다.
             채점: 찾은 점 주변을 좁게 잘라 OCR 해서 그 라벨이 읽히는지 (label_verify).
             다른 버튼 라벨이 읽히면 wrong_label - 이게 오클릭 위험 신호다.

  [cursor]   마우스 커서 추적 정확도. **정답을 우리가 만든다**: 이미지 좌표로 목표점을
             정하고 -> 스크린 좌표로 변환해 커서를 그리로 옮기고 -> 다시 캡처해서
             VLM 에게 커서를 찾게 한 뒤, 원래 이미지 좌표와의 픽셀 오차를 잰다.
             커서 탐지는 1단계 태스크라 조합이 아니라 **모델 단독**으로 비교한다.

안전:
  - buttons arm 은 캡처 + VLM 호출뿐 - **클릭도 이동도 없다** (기본 동작).
  - cursor arm 은 물리 마우스를 움직이므로 opt-in 이다: `BENCH_CURSOR_ARM=1` 그리고
    `SAFE_MODE=0` 둘 다 필요. 클릭/휠은 절대 하지 않고 이동만 한다.
    probe 점은 기본이 창 왼쪽 여백(x=8%)이다 - 가운데 라이브 SEM 박스를 피한다.
    레이아웃이 다르면 `BENCH_CURSOR_PROBES` 로 옮겨라.

설정 (모두 env, CLI 인자 없음):
  BENCH_TOOL_ID        대상 tool 창의 장비 ID (없으면 열려 있는 아무 tool 창)
  BENCH_WINDOW_TARGETS 찾을 버튼 라벨 목록 (기본 Stop,Queue,PM)
  BENCH_COMBOS         "coarse>fine" 조합 (기본 bench_tool_locator 와 동일 4종)
  BENCH_REPEATS        반복 (기본 3)
  BENCH_CURSOR_ARM     1 이면 커서 추적 arm 실행 (기본 0)
  BENCH_CURSOR_PROBES  "0.08x0.2,0.08x0.5,0.08x0.8" 형태의 이미지 비율 좌표
  BENCH_CURSOR_HIT_PX  커서 명중 판정 반경 px (기본 24)

사용법:
  uv run python poc/workflow_3/rcs/bench_tool_window_reader.py
  BENCH_CURSOR_ARM=1 SAFE_MODE=0 uv run python poc/workflow_3/rcs/bench_tool_window_reader.py

주: 커서 프롬프트는 `recording_filter.cursor_prompt` 를 재사용한다(순수 문자열 모듈).
벤치 전용 sibling import 이며 production 루프 경로가 아니다.
"""

import math
import os
import statistics
import sys
import time

from dotenv import load_dotenv

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json, save_marked_bboxes
from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
from poc.workflow_3.recording_filter.cursor_prompt import (
    cursor_system_prompt,
    cursor_user_prompt,
)
from poc.workflow_3.util import (
    bbox_center,
    capture_window,
    image_point_to_screen,
    make_timestamp_tag,
    move_cursor_to_screen,
    point_to_tiny_bbox,
)
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_3.vlm.label_verify import (
    crop_box_around_point,
    label_matches,
    read_text_near_point,
)
from poc.workflow_3.vlm.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "bench_tool_window_reader"
COMPONENT_NAME = LOG_NAME
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "bench_tool_window_reader"

# 2026-09-03: bench_tool_locator 와 같은 결함이 여기 남아 있었다 - production 은
# 2026-08-07 부터 mai-ui>mai-ui 인데 PRODUCTION_COMBO 가 ui-venus>mai-ui 라
# `is_production` 이 모든 행에서 False 였고, ui-venus 는 가중치까지 삭제돼
# 기본 조합 4개 중 3개가 404 만 쌓는다. **기본값에는 실제로 서빙 중인 것만 둔다.**
DEFAULT_COMBOS = [
    ("mai-ui", "mai-ui"),  # 현재 production 설정 (8B 양단)
]
PRODUCTION_COMBO = ("mai-ui", "mai-ui")
CURSOR_MODELS = ["mai-ui"]

DEFAULT_TARGET_LABELS = ["Stop", "Queue", "PM"]

# 라벨별 추가 설명 - 화면에서 헷갈릴 만한 이웃을 명시해 grounding 을 돕는다.
TARGET_HINTS = {
    "Stop": (
        "the 'Stop' button that halts the running measurement. It sits in the operation "
        "button row together with other run-control buttons."
    ),
    "Queue": (
        "the 'Queue' button that opens the measurement queue / job list. It sits in the "
        "operation button row."
    ),
    "PM": (
        "the small 'PM' button next to the magnification value readout near the live SEM "
        "image panel. It opens the magnification dropdown."
    ),
}

# 버튼 라벨 확인용 crop - 행 strip 보다 작다(버튼은 짧고 낮다).
LABEL_LEFT_RATIO = 0.035
LABEL_RIGHT_RATIO = 0.035
LABEL_HALF_HEIGHT_RATIO = 0.018

SCORE_CORRECT = "correct"
SCORE_WRONG_LABEL = "wrong_label"
SCORE_UNREADABLE = "unreadable"
SCORE_NO_DETECT = "no_detect"
SCORE_ERROR = "error"


def _env_int(name: str, default: int) -> int:
    """env 정수값 로드."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARNING] {name}={raw!r} 파싱 실패 -> 기본값 {default} 사용")
        return default


def _env_on(name: str) -> bool:
    """env 불리언 토글."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_labels() -> list[str]:
    """BENCH_WINDOW_TARGETS 에서 버튼 라벨 목록을 읽는다."""
    raw = os.getenv("BENCH_WINDOW_TARGETS", "").strip()
    if not raw:
        return list(DEFAULT_TARGET_LABELS)
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels or list(DEFAULT_TARGET_LABELS)


def _load_combos() -> list[tuple[str, str]]:
    """BENCH_COMBOS ("coarse>fine,...") 에서 조합 목록을 읽는다."""
    raw = os.getenv("BENCH_COMBOS", "").strip()
    if not raw:
        return list(DEFAULT_COMBOS)
    combos: list[tuple[str, str]] = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk or ">" not in chunk:
            if chunk:
                print(f"[WARNING] BENCH_COMBOS 항목 {chunk!r} 파싱 실패 - 건너뜁니다.")
            continue
        coarse, fine = (piece.strip() for piece in chunk.split(">", 1))
        if coarse and fine:
            combos.append((coarse, fine))
    return combos or list(DEFAULT_COMBOS)


def _load_cursor_probes() -> list[tuple[float, float]]:
    """BENCH_CURSOR_PROBES ("0.08x0.2,...") 에서 probe 비율 좌표를 읽는다."""
    raw = os.getenv("BENCH_CURSOR_PROBES", "").strip()
    if not raw:
        return [(0.08, 0.2), (0.08, 0.5), (0.08, 0.8)]
    probes: list[tuple[float, float]] = []
    for part in raw.split(","):
        chunk = part.strip().lower()
        if not chunk or "x" not in chunk:
            continue
        try:
            rx, ry = (float(piece) for piece in chunk.split("x", 1))
        except ValueError:
            print(f"[WARNING] BENCH_CURSOR_PROBES 항목 {chunk!r} 파싱 실패 - 건너뜁니다.")
            continue
        if 0.0 < rx < 1.0 and 0.0 < ry < 1.0:
            probes.append((rx, ry))
        else:
            print(f"[WARNING] probe {chunk!r} 가 0~1 범위 밖 - 건너뜁니다.")
    return probes


def _combo_label(combo: tuple[str, str]) -> str:
    """조합 라벨."""
    return f"{combo[0]}__{combo[1]}"


def _button_target(label: str) -> TargetConfig:
    """버튼 라벨용 grounding 타겟."""
    hint = TARGET_HINTS.get(
        label,
        f"the '{label}' button in this CD-SEM tool control window",
    )
    return TargetConfig(
        key=f"button_{label.lower()}",
        description=(
            f"{hint} Return a safe click point on the button surface itself, not on "
            f"nearby labels, readouts, or the live image."
        ),
        left_pad_ratio=0.8,
        right_pad_ratio=0.8,
        vertical_pad_ratio=0.8,
        min_crop_width=320,
        min_crop_height=96,
        vertical_pad_min_px=16,
    )


def _score_button_point(
    image,
    point: dict | None,
    label: str,
    all_labels: list[str],
    *,
    timestamp_tag: str,
    artifact_label: str,
) -> tuple[str, dict]:
    """버튼 클릭점을 라벨 OCR 로 채점한다."""
    if point is None:
        return SCORE_NO_DETECT, {"raw_text": "", "box": None}

    width, height = image.size
    box = crop_box_around_point(
        point,
        width,
        height,
        left_ratio=LABEL_LEFT_RATIO,
        right_ratio=LABEL_RIGHT_RATIO,
        half_height_ratio=LABEL_HALF_HEIGHT_RATIO,
    )
    read = read_text_near_point(
        image,
        box,
        debug_image_dir=DEBUG_ARTIFACT_DIR,
        timestamp_tag=timestamp_tag,
        artifact_label=artifact_label,
        log_name=LOG_NAME,
    )
    detail = {"raw_text": read.raw_text, "tokens": read.tokens, "box": box}
    if not read.ok:
        return SCORE_ERROR, detail

    if label_matches(read.tokens, label):
        return SCORE_CORRECT, detail

    # 다른 버튼 라벨을 읽었으면 명확한 오검출이다 (오클릭 위험).
    for other in all_labels:
        if other != label and label_matches(read.tokens, other):
            detail["wrong_label"] = other
            return SCORE_WRONG_LABEL, detail
    return SCORE_UNREADABLE, detail


def _save_overlay(image, point: dict | None, box: dict | None, filename: str) -> str:
    """클릭점 + 채점 crop 을 프레임 위에 그려 저장한다."""
    if point is None:
        return ""
    img_w, img_h = image.size
    elements = {"point": {"bbox": point_to_tiny_bbox(point, img_w, img_h), "center": point}}
    colors = {"point": "deepskyblue"}
    if box is not None:
        elements["ocr_box"] = {"bbox": box}
        colors["ocr_box"] = "yellow"
    out_path = DEBUG_ARTIFACT_DIR / filename
    save_marked_bboxes(image, elements, colors, out_path)
    return str(out_path)


def _run_button_arm(
    window,
    window_title: str,
    backend: str,
    image,
    labels: list[str],
    combos: list[tuple[str, str]],
    repeats: int,
) -> list[dict]:
    """조합 x 버튼 x 반복 - 버튼 찾기 정확도."""
    results: list[dict] = []
    for combo in combos:
        for label in labels:
            for repeat_idx in range(1, repeats + 1):
                timestamp_tag = make_timestamp_tag(time.time())
                artifact_label = f"{_combo_label(combo)}_{label}_r{repeat_idx}"
                started_at = time.time()
                try:
                    target_result = analyze_window_target(
                        window,
                        window_title,
                        backend,
                        _button_target(label),
                        debug_image_dir=DEBUG_ARTIFACT_DIR,
                        log_name=LOG_NAME,
                        component_name=COMPONENT_NAME,
                        artifact_prefix=f"btn_{artifact_label}",
                        coarse_service_slug=combo[0],
                        refine_service_slug=combo[1],
                        result_mode="bench_tool_window_button",
                        image=image,
                    )
                    point = (
                        target_result.point
                        if target_result.exit_code == DETECT_SUCCESS
                        else None
                    )
                except Exception as exc:
                    print(f"[ERROR] button run 실패 {artifact_label}: {type(exc).__name__}: {exc}")
                    results.append(
                        {
                            "arm": "buttons",
                            "combo": _combo_label(combo),
                            "label": label,
                            "repeat": repeat_idx,
                            "score": SCORE_ERROR,
                            "point": None,
                            "elapsed_sec": round(time.time() - started_at, 2),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                score, detail = _score_button_point(
                    image,
                    point,
                    label,
                    labels,
                    timestamp_tag=timestamp_tag,
                    artifact_label=artifact_label,
                )
                overlay = _save_overlay(
                    image, point, detail.get("box"), f"{timestamp_tag}_{artifact_label}_overlay.jpg"
                )
                elapsed = time.time() - started_at
                print(
                    f"[INFO] [buttons] {_combo_label(combo):24s} {label:6s} r{repeat_idx} "
                    f"-> {score:12s} point={point} read={detail.get('raw_text','')[:24]!r} "
                    f"({elapsed:.1f}s)"
                )
                results.append(
                    {
                        "arm": "buttons",
                        "combo": _combo_label(combo),
                        "coarse_service": combo[0],
                        "refine_service": combo[1],
                        "label": label,
                        "repeat": repeat_idx,
                        "score": score,
                        "point": point,
                        "raw_text": detail.get("raw_text", ""),
                        "wrong_label": detail.get("wrong_label", ""),
                        "overlay_path": overlay,
                        "elapsed_sec": round(elapsed, 2),
                    }
                )
    return results


def _locate_cursor_once(client, image) -> dict | None:
    """한 프레임에서 커서 중심점(이미지 px)을 찾는다."""
    image_b64, width, height = encode_image_webp(image)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=cursor_system_prompt(),
        user_text=cursor_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return None
    bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
    if bbox_1000 is None:
        return None
    return bbox_center(bbox_1000_to_pixels(bbox_1000, width, height))


def _run_cursor_arm(
    window,
    window_title: str,
    backend: str,
    base_image,
    probes: list[tuple[float, float]],
    repeats: int,
    hit_px: int,
) -> list[dict]:
    """커서를 아는 좌표로 옮기고, 모델별로 그 좌표를 되찾게 해 픽셀 오차를 잰다.

    정답은 우리가 만든 이미지 좌표다 - 별도 라벨링이 필요 없다.
    """
    results: list[dict] = []
    img_w, img_h = base_image.size
    clients: dict[str, Workflow1VLMClient] = {}
    for slug in CURSOR_MODELS:
        try:
            clients[slug] = Workflow1VLMClient(service_slug=slug, log_name=LOG_NAME)
        except Exception as exc:
            print(f"[WARNING] cursor arm: {slug} client 생성 실패 - 제외: {exc}")

    for probe_idx, (ratio_x, ratio_y) in enumerate(probes, start=1):
        truth = {"x": int(img_w * ratio_x), "y": int(img_h * ratio_y)}
        screen_point = image_point_to_screen(window, truth, image_size=base_image.size)
        if screen_point is None:
            print(f"[WARNING] probe{probe_idx}: 스크린 좌표 변환 실패 - 건너뜁니다.")
            continue

        # 이동만 한다 - 클릭/휠 없음.
        move_cursor_to_screen(
            screen_point,
            f"bench_cursor_probe{probe_idx}",
            action_enabled=True,
        )
        time.sleep(0.4)
        try:
            frame = capture_window(window)
        except Exception as exc:
            print(f"[WARNING] probe{probe_idx}: 캡처 실패 - 건너뜁니다: {exc}")
            continue
        timestamp_tag = make_timestamp_tag(time.time())
        save_debug_jpeg(frame, DEBUG_ARTIFACT_DIR / f"{timestamp_tag}_cursor_probe{probe_idx}.jpg")

        for slug, client in clients.items():
            for repeat_idx in range(1, repeats + 1):
                started_at = time.time()
                try:
                    found = _locate_cursor_once(client, frame)
                except Exception as exc:
                    print(f"[ERROR] cursor {slug} probe{probe_idx} r{repeat_idx}: {exc}")
                    found = None
                error_px = None
                if found is not None:
                    error_px = math.hypot(found["x"] - truth["x"], found["y"] - truth["y"])
                hit = error_px is not None and error_px <= hit_px
                print(
                    f"[INFO] [cursor] {slug:10s} probe{probe_idx} r{repeat_idx} -> "
                    f"{'HIT ' if hit else 'MISS'} err="
                    f"{'n/a' if error_px is None else f'{error_px:.0f}px'} "
                    f"truth={truth} found={found}"
                )
                results.append(
                    {
                        "arm": "cursor",
                        "model": slug,
                        "probe": probe_idx,
                        "repeat": repeat_idx,
                        "truth": truth,
                        "found": found,
                        "error_px": None if error_px is None else round(error_px, 1),
                        "hit": hit,
                        "elapsed_sec": round(time.time() - started_at, 2),
                    }
                )
    return results


def _summarize_buttons(results: list[dict], combos: list[tuple[str, str]]) -> list[dict]:
    """조합별 버튼 집계."""
    summary: list[dict] = []
    for combo in combos:
        label = _combo_label(combo)
        rows = [row for row in results if row.get("combo") == label]
        if not rows:
            continue
        total = len(rows)
        counts = {
            score: sum(1 for row in rows if row["score"] == score)
            for score in (
                SCORE_CORRECT,
                SCORE_WRONG_LABEL,
                SCORE_UNREADABLE,
                SCORE_NO_DETECT,
                SCORE_ERROR,
            )
        }
        latencies = [row["elapsed_sec"] for row in rows]
        summary.append(
            {
                "combo": label,
                "single_model": combo[0] == combo[1],
                "is_production": tuple(combo) == PRODUCTION_COMBO,
                "total": total,
                "accuracy": round(counts[SCORE_CORRECT] / total, 3),
                "wrong_label_rate": round(counts[SCORE_WRONG_LABEL] / total, 3),
                "median_sec": round(statistics.median(latencies), 2),
                "n_correct": counts[SCORE_CORRECT],
                **{f"n_{key}": value for key, value in counts.items()},
            }
        )
    summary.sort(key=lambda item: (-item["accuracy"], item["wrong_label_rate"]))
    return summary


def _summarize_cursor(results: list[dict]) -> list[dict]:
    """모델별 커서 집계."""
    summary: list[dict] = []
    for slug in CURSOR_MODELS:
        rows = [row for row in results if row.get("model") == slug]
        if not rows:
            continue
        hits = sum(1 for row in rows if row["hit"])
        errors = [row["error_px"] for row in rows if row["error_px"] is not None]
        summary.append(
            {
                "model": slug,
                "total": len(rows),
                "hit_rate": round(hits / len(rows), 3),
                "found_rate": round(len(errors) / len(rows), 3),
                "median_err_px": round(statistics.median(errors), 1) if errors else None,
            }
        )
    summary.sort(key=lambda item: -item["hit_rate"])
    return summary


def _per_label_misses(results: list[dict]) -> list[tuple[str, int]]:
    """라벨별 실패(정답 아님) 횟수."""
    counts: dict[str, int] = {}
    for row in results:
        if row.get("score") not in (SCORE_CORRECT, None):
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1])


def _print_digest(
    button_summary: list[dict],
    cursor_summary: list[dict],
    label_misses: list[tuple[str, int]],
    labels: list[str],
    repeats: int,
) -> None:
    """한 덩어리 digest."""
    print("")
    print("[DIGEST] ===== tool window reader bench =====")
    print(f"[DIGEST] targets={','.join(labels)} repeats={repeats}")
    if button_summary:
        # 실패 종류를 항상 쪼개 보여준다 - wrong(다른 버튼 라벨을 읽음 = 오클릭 위험)과
        # nodet(미검출, 재시도로 회복)은 성질이 다른데 합계만 보면 구분되지 않는다.
        print(
            f"[DIGEST] [buttons] {'combo':26s} {'acc':>6s} {'med_s':>6s} "
            f"{'ok':>4s} {'wrong':>5s} {'unread':>6s} {'nodet':>5s} {'err':>4s}"
        )
        for item in button_summary:
            marker = " *prod" if item["is_production"] else ("  1vlm" if item["single_model"] else "")
            print(
                f"[DIGEST] [buttons] {item['combo']:26s} {item['accuracy']:6.3f} "
                f"{item['median_sec']:6.2f} {item['n_' + SCORE_CORRECT]:4d} "
                f"{item['n_' + SCORE_WRONG_LABEL]:5d} {item['n_' + SCORE_UNREADABLE]:6d} "
                f"{item['n_' + SCORE_NO_DETECT]:5d} {item['n_' + SCORE_ERROR]:4d}{marker}"
            )
        best = button_summary[0]
        prod = next((item for item in button_summary if item["is_production"]), None)
        best_single = next((item for item in button_summary if item["single_model"]), None)
        print(f"[DIGEST] [buttons] best={best['combo']} acc={best['accuracy']:.3f}")
        if prod and best_single:
            delta = best_single["accuracy"] - prod["accuracy"]
            print(
                f"[DIGEST] [buttons] best_single={best_single['combo']} "
                f"acc={best_single['accuracy']:.3f} delta_vs_prod={delta:+.3f}"
            )
    else:
        print("[DIGEST] [buttons] 결과 없음")

    if label_misses:
        detail = ", ".join(f"{name}x{count}" for name, count in label_misses)
        print(f"[DIGEST] [buttons] 실패 많은 라벨: {detail}")

    if cursor_summary:
        print(f"[DIGEST] [cursor]  {'model':12s} {'hit':>6s} {'found':>6s} {'med_err_px':>10s}")
        for item in cursor_summary:
            err = "n/a" if item["median_err_px"] is None else f"{item['median_err_px']:.1f}"
            print(
                f"[DIGEST] [cursor]  {item['model']:12s} {item['hit_rate']:6.3f} "
                f"{item['found_rate']:6.3f} {err:>10s}  ({item['total']} runs)"
            )
    else:
        print("[DIGEST] [cursor]  미실행 (BENCH_CURSOR_ARM=1 + SAFE_MODE=0 필요)")
    print("[DIGEST] 주의: 이 결과는 tool 창 기준이다. List 탭 결과와 별개로 읽어라.")
    print("[DIGEST] =====================================")


def main() -> str:
    """열린 tool 창을 캡처해 버튼 arm (+옵션 커서 arm) 을 돌린다."""
    labels = _load_labels()
    combos = _load_combos()
    repeats = _env_int("BENCH_REPEATS", 3)
    tool_id = os.getenv("BENCH_TOOL_ID", "").strip()

    button_runs = len(combos) * len(labels) * repeats
    print(
        f"[INFO] bench 시작: combos={len(combos)} targets={len(labels)} repeats={repeats} "
        f"-> buttons {button_runs} runs (VLM 콜 약 {button_runs * 3})"
    )
    print("[INFO] buttons arm 은 클릭도 이동도 하지 않습니다 (캡처 + VLM 호출만).")

    window, window_title, backend = find_remote_monitoring_window(tool_id)
    if window is None:
        print(
            "[ERROR] 열린 tool 창(Remote Monitoring System)을 찾지 못했습니다. "
            "tool 에 접속한 뒤 다시 실행하세요."
        )
        return "tool_window_not_found"
    print(f"[INFO] tool 창: title={window_title!r} backend={backend}")

    # capture_window(window) 한 인자만 받고, 실패하면 None 이 아니라 예외를 던진다.
    try:
        image = capture_window(window)
    except Exception as exc:
        print(f"[ERROR] tool 창 캡처 실패: {exc}")
        return "capture_failed"

    run_tag = make_timestamp_tag(time.time())
    save_debug_jpeg(image, DEBUG_ARTIFACT_DIR / f"{run_tag}_window_frame.jpg")
    print(f"[INFO] 기준 프레임 캡처 완료: {image.size} (버튼 arm 전체가 이 프레임을 공유)")

    results = _run_button_arm(window, window_title, backend, image, labels, combos, repeats)

    cursor_results: list[dict] = []
    cursor_requested = _env_on("BENCH_CURSOR_ARM")
    safe_mode = _env_on("SAFE_MODE")
    if cursor_requested and safe_mode:
        print("[WARNING] BENCH_CURSOR_ARM=1 이지만 SAFE_MODE=1 - 커서 arm 을 건너뜁니다.")
    elif cursor_requested:
        probes = _load_cursor_probes()
        if not probes:
            print("[WARNING] 유효한 커서 probe 가 없어 커서 arm 을 건너뜁니다.")
        else:
            print(f"[INFO] 커서 arm 시작: probes={probes} (이동만, 클릭/휠 없음)")
            cursor_results = _run_cursor_arm(
                window, window_title, backend, image, probes, repeats,
                _env_int("BENCH_CURSOR_HIT_PX", 24),
            )
    else:
        print("[INFO] 커서 arm 미실행 (BENCH_CURSOR_ARM=1 + SAFE_MODE=0 이면 실행).")

    button_summary = _summarize_buttons(results, combos)
    cursor_summary = _summarize_cursor(cursor_results)
    label_misses = _per_label_misses(results)

    result_path = DEBUG_ARTIFACT_DIR / f"{run_tag}_bench_result.json"
    save_debug_json(
        result_path,
        {
            "window_title": window_title,
            "backend": backend,
            "frame_size": {"width": image.size[0], "height": image.size[1]},
            "labels": labels,
            "combos": [_combo_label(combo) for combo in combos],
            "repeats": repeats,
            "button_summary": button_summary,
            "cursor_summary": cursor_summary,
            "runs": results,
            "cursor_runs": cursor_results,
        },
    )
    _print_digest(button_summary, cursor_summary, label_misses, labels, repeats)
    print(f"[INFO] 상세 결과: {result_path}")
    print(f"[INFO] overlay/crop 이미지: {DEBUG_ARTIFACT_DIR}")
    return "success"


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
