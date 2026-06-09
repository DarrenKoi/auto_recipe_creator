"""캡처된 Tool 창 프레임에 대해 align-key matcher 를 돌리는 실데이터 eval 하네스.

흐름:

```
poc/workflow_1/capture_window_frames_tool.py
  → poc/workflow_1/recordings/capture_window_frames_tool/<tag>/frames/*.jpg
  (사용자가 폴더를 통째로 poc/workflow_2/recordings/<run-tag>/ 로 복사)

본 스크립트
  - templates/<recipe_id>.jpg  →  build_template()
  - templates/sem_panel_landmarks/<model>/...  →  sem_panel_locator.load_landmarks()
  - recordings/<run-tag>/frames/*.jpg 의 앞 N 장 순회
    · locate_panel() 으로 SEM panel ROI 추출
    · compute_align_key_score(template, frame, roi_hint=panel_roi)
    · overlay JPEG + JSONL row 저장
  - summary.json + 스튜드아웃 표 출력
```

설계 근거: ``docs/workflow_2/03-research-notes-and-better-options.md`` 의
"다음 구현 제안" 1번 — Real-data evaluation output (JSONL).

실행:
    uv run python poc/workflow_2/test_match_on_captured_frames.py
"""

import json
import time
from pathlib import Path

import cv2

from poc.workflow_3 import DEBUG_IMAGE_DIR, WORKFLOW_3_DIR
from poc.workflow_3.vision.align_key_matcher import (
    AlignKeyMatchResult,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)
from poc.workflow_3.vision.sem_panel_locator import (
    LANDMARK_CONF_MIN,
    SEMPanelMatch,
    load_landmarks,
    locate_panel,
)


# ------------------------------------------------------------------
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정.
# ------------------------------------------------------------------

# fixture(녹화 프레임/landmark 샘플)는 legacy workflow_2 폴더에 남아 있다 —
# 코드 import 없이 경로만 참조한다.
_LEGACY_WORKFLOW_2_DIR = WORKFLOW_3_DIR.parent / "workflow_2"
DEFAULT_RECORDINGS_DIR = _LEGACY_WORKFLOW_2_DIR / "recordings"
DEFAULT_TEMPLATES_DIR = _LEGACY_WORKFLOW_2_DIR / "templates"
DEFAULT_LANDMARKS_DIR = DEFAULT_TEMPLATES_DIR / "sem_panel_landmarks"

# None → recordings/ 아래에서 가장 최신 서브폴더를 자동 선택.
DEFAULT_RUN_TAG: str | None = None
# templates/<recipe_id>.jpg (+ <recipe_id>_meta.json) 를 읽는다.
DEFAULT_RECIPE_ID: str = "demo"
# 첫 N 장만 처리. None 이면 전체.
DEFAULT_FRAME_LIMIT: int | None = 30
DEFAULT_FRAME_GLOB: str = "*.jpg"

# landmark 매칭이 실패했을 때 frame 전체를 매칭에 사용할지 여부.
# False 이면 해당 프레임은 skip 로 기록.
FALLBACK_TO_FULL_FRAME: bool = True

# landmark 신뢰도 최저값 — 필요시 module 단에서 override.
PANEL_MIN_CONFIDENCE: float = LANDMARK_CONF_MIN


# ------------------------------------------------------------------
# 경로 해석.
# ------------------------------------------------------------------


def _resolve_run_tag(recordings_dir: Path, run_tag: str | None) -> Path:
    """``run_tag`` 가 지정되면 그 폴더를, 없으면 recordings/ 아래의 최신 폴더를 반환."""
    if not recordings_dir.exists():
        raise FileNotFoundError(
            f"recordings 디렉터리가 없습니다: {recordings_dir} "
            "— workflow_1 의 capture 폴더를 이 경로로 복사하세요"
        )

    if run_tag is not None:
        candidate = recordings_dir / run_tag
        if not candidate.is_dir():
            raise FileNotFoundError(f"run_tag 디렉터리를 찾지 못함: {candidate}")
        return candidate

    subdirs = [p for p in recordings_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"{recordings_dir} 아래에 캡처 폴더가 없습니다. "
            "workflow_1 의 recordings/capture_window_frames_tool/<tag>/ 폴더를 복사하세요"
        )
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _resolve_frames_dir(run_dir: Path) -> Path:
    """workflow_1 캡처 폴더는 ``frames/`` 서브폴더를 가진다. 없으면 run_dir 자체를 사용."""
    frames_dir = run_dir / "frames"
    if frames_dir.is_dir():
        return frames_dir
    return run_dir


# ------------------------------------------------------------------
# 템플릿 로딩.
# ------------------------------------------------------------------


def _load_template_meta(templates_dir: Path, recipe_id: str) -> dict:
    meta_path = templates_dir / f"{recipe_id}_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARNING] template meta 파싱 실패, 무시: {meta_path} ({exc})")
        return {}


def _load_recipe_template(templates_dir: Path, recipe_id: str):
    template_path = templates_dir / f"{recipe_id}.jpg"
    if not template_path.exists():
        raise FileNotFoundError(
            f"recipe template 이미지를 찾을 수 없습니다: {template_path} "
            "(templates/<recipe_id>.jpg 형식으로 배치하세요)"
        )
    raw = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"template 이미지를 디코드하지 못함: {template_path}")
    meta = _load_template_meta(templates_dir, recipe_id)
    template = build_template(
        raw,
        recipe_id=recipe_id,
        version=str(meta.get("version", "v0")),
        nm_per_pixel=meta.get("nm_per_pixel"),
        key_type=meta.get("key_type"),
    )
    print(
        f"[INFO] recipe template 로드: {template_path} "
        f"shape={template.raw_image.shape} "
        f"nm_per_pixel={template.nm_per_pixel} key_type={template.key_type}"
    )
    return template


# ------------------------------------------------------------------
# 한 프레임 처리.
# ------------------------------------------------------------------


def _record_for_skip(
    frame_path: Path,
    frame_index: int,
    recipe_id: str,
    reason: str,
) -> dict:
    return {
        "frame_path": str(frame_path),
        "frame_index": frame_index,
        "recipe_id": recipe_id,
        "model_id": None,
        "panel_confidence": None,
        "panel_roi": None,
        "score": None,
        "chamfer": None,
        "orb": None,
        "best_xy": None,
        "best_scale": None,
        "decision": "skipped",
        "fallback_full_frame": False,
        "skip_reason": reason,
    }


def _record_for_result(
    frame_path: Path,
    frame_index: int,
    recipe_id: str,
    match: SEMPanelMatch | None,
    result: AlignKeyMatchResult,
    fallback_full_frame: bool,
) -> dict:
    return {
        "frame_path": str(frame_path),
        "frame_index": frame_index,
        "recipe_id": recipe_id,
        "model_id": match.model_id if match is not None else None,
        "panel_confidence": float(match.confidence) if match is not None else None,
        "panel_roi": list(match.panel_roi) if match is not None else None,
        "score": float(result.score),
        "chamfer": float(result.chamfer_score),
        "orb": float(result.orb_inlier_ratio),
        "best_xy": [int(result.best_xy[0]), int(result.best_xy[1])],
        "best_scale": float(result.best_scale),
        "decision": result.decision,
        "fallback_full_frame": bool(fallback_full_frame),
    }


# ------------------------------------------------------------------
# 요약 통계.
# ------------------------------------------------------------------


def _summarize(records: list[dict]) -> dict:
    decisions = ("match", "adjust", "low", "skipped")
    by_decision: dict[str, dict] = {}
    for d in decisions:
        rows = [r for r in records if r["decision"] == d]
        scores = [r["score"] for r in rows if r["score"] is not None]
        by_decision[d] = {
            "count": len(rows),
            "score_mean": (sum(scores) / len(scores)) if scores else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }

    processed = [r for r in records if r["decision"] != "skipped"]
    panel_hits = sum(1 for r in processed if r["model_id"] is not None)
    return {
        "by_decision": by_decision,
        "processed_count": len(processed),
        "panel_hit_count": panel_hits,
        "panel_hit_rate": (panel_hits / len(processed)) if processed else None,
        "total_count": len(records),
    }


def _print_summary_table(summary: dict) -> None:
    print("[INFO] ───── 요약 ─────")
    print(
        f"[INFO] total={summary['total_count']} processed={summary['processed_count']} "
        f"panel_hits={summary['panel_hit_count']} "
        f"panel_hit_rate={summary['panel_hit_rate']}"
    )
    for decision, stats in summary["by_decision"].items():
        score_mean = stats["score_mean"]
        score_min = stats["score_min"]
        score_max = stats["score_max"]
        mean_txt = f"{score_mean:.3f}" if score_mean is not None else "-"
        min_txt = f"{score_min:.3f}" if score_min is not None else "-"
        max_txt = f"{score_max:.3f}" if score_max is not None else "-"
        print(
            f"[INFO] {decision:<8} count={stats['count']:<4} "
            f"score(mean/min/max)={mean_txt}/{min_txt}/{max_txt}"
        )


# ------------------------------------------------------------------
# 메인.
# ------------------------------------------------------------------


def main() -> int:
    print("[INFO] test_match_on_captured_frames 시작")

    run_dir = _resolve_run_tag(DEFAULT_RECORDINGS_DIR, DEFAULT_RUN_TAG)
    frames_dir = _resolve_frames_dir(run_dir)
    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] frames_dir={frames_dir}")

    frame_paths = sorted(frames_dir.glob(DEFAULT_FRAME_GLOB))
    if not frame_paths:
        print(f"[ERROR] 프레임이 없습니다: {frames_dir}/{DEFAULT_FRAME_GLOB}")
        return 1
    if DEFAULT_FRAME_LIMIT is not None:
        frame_paths = frame_paths[: DEFAULT_FRAME_LIMIT]
    print(f"[INFO] 처리 대상 프레임: {len(frame_paths)} 장")

    template = _load_recipe_template(DEFAULT_TEMPLATES_DIR, DEFAULT_RECIPE_ID)
    landmarks = load_landmarks(DEFAULT_LANDMARKS_DIR)

    run_tag = run_dir.name
    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    out_dir = (
        DEBUG_IMAGE_DIR
        / "real_eval"
        / f"{DEFAULT_RECIPE_ID}__{run_tag}__{timestamp}"
    )
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    print(f"[INFO] 출력 디렉터리: {out_dir}")

    records: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_fp:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARNING] 프레임 디코드 실패: {frame_path}")
                rec = _record_for_skip(
                    frame_path,
                    frame_index,
                    DEFAULT_RECIPE_ID,
                    reason="decode_failed",
                )
                records.append(rec)
                jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            match = locate_panel(
                frame,
                landmarks,
                min_confidence=PANEL_MIN_CONFIDENCE,
            )

            fallback_full_frame = False
            roi_hint: tuple[int, int, int, int] | None
            frame_nm_per_pixel: float | None

            if match is not None:
                roi_hint = match.panel_roi
                frame_nm_per_pixel = match.nm_per_pixel
            else:
                if FALLBACK_TO_FULL_FRAME:
                    fallback_full_frame = True
                    roi_hint = None
                    frame_nm_per_pixel = None
                    print(
                        f"[WARNING] panel 미검출 → 전체 frame 매칭: "
                        f"frame_index={frame_index} path={frame_path.name}"
                    )
                else:
                    rec = _record_for_skip(
                        frame_path,
                        frame_index,
                        DEFAULT_RECIPE_ID,
                        reason="panel_not_found",
                    )
                    records.append(rec)
                    jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(
                        f"[WARNING] panel 미검출 → skip: "
                        f"frame_index={frame_index} path={frame_path.name}"
                    )
                    continue

            try:
                result = compute_align_key_score(
                    template,
                    frame,
                    frame_nm_per_pixel=frame_nm_per_pixel,
                    roi_hint=roi_hint,
                )
            except ValueError as exc:
                # roi_hint 가 너무 작거나 frame 과 교차하지 않는 경우.
                print(
                    f"[WARNING] matcher 호출 실패 → skip: "
                    f"frame_index={frame_index} reason={exc}"
                )
                rec = _record_for_skip(
                    frame_path,
                    frame_index,
                    DEFAULT_RECIPE_ID,
                    reason=f"matcher_error: {exc}",
                )
                records.append(rec)
                jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            overlay_name = f"{frame_path.stem}_overlay.jpg"
            save_overlay_jpeg(result.debug_overlay, overlay_dir / overlay_name)

            rec = _record_for_result(
                frame_path,
                frame_index,
                DEFAULT_RECIPE_ID,
                match,
                result,
                fallback_full_frame,
            )
            records.append(rec)
            jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"[INFO] iter={frame_index:03d} {frame_path.name} "
                f"model={rec['model_id']} conf={rec['panel_confidence']} "
                f"decision={result.decision} score={result.score:.3f} "
                f"(chamfer={result.chamfer_score:.3f} orb={result.orb_inlier_ratio:.3f})"
            )

    summary = _summarize(records)
    summary_path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "recipe_id": DEFAULT_RECIPE_ID,
                "frames_dir": str(frames_dir),
                "frame_limit": DEFAULT_FRAME_LIMIT,
                "fallback_to_full_frame": FALLBACK_TO_FULL_FRAME,
                "panel_min_confidence": PANEL_MIN_CONFIDENCE,
                "jsonl_path": str(jsonl_path),
                "overlay_dir": str(overlay_dir),
                "started_at": timestamp,
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _print_summary_table(summary)
    print(f"[INFO] JSONL: {jsonl_path}")
    print(f"[INFO] summary: {summary_path}")

    return 0 if summary["total_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
