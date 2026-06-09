"""Align point correction batch 결과를 *텍스트 한 덩어리* 로 집계한다 (스크린샷 대체용).

오피스 Windows 에서 한 줄 실행하고, 콘솔 출력(또는 digest.txt)을 그대로 붙여넣으면
스크린샷 없이도 진단 신호를 전달할 수 있게 하는 게 목적이다.

집계하는 진단 신호 (세 곳에 흩어진 데이터를 한 텍스트로):
  1. GLOBAL STATUS COUNTS  — 전 recipe status 빈도 (worst-first). 문제 규모.
  2. RCP KEY box detection  — recipe 별 흰 box 검출/ fallback 비율 (C1 key 품질 프록시).
  3. WINNER best_scale 분포 — om/sem payload 의 best_scale 히스토그램. 1.0 에 몰리면
     COMPARE_SCALES(<=1.0) 가 확대 방향을 못 봐서 생기는 C4 scale-miss 의심.
  4. WINNER score 분포      — adjust_threshold 미만 개수.
  5. SUSPECT_SUCCESS 목록   — S 라벨인데 보정 거리 큰 케이스 (사용자 최우선).
  6. NOT_DISTINCTIVE 목록   — distinctiveness ratio 나쁜 케이스.
  7. PER-RECIPE one-liner   — recipe 별 한 줄 요약.

데이터 소스: align_review 의 helper 를 재사용해 최신 batch 의 recipe out_dir 들을 모은다.
  - 행 단위 신호 (status/scale/score/suspect)는 results.jsonl 에서.
  - rcp_box 검출은 행에 없으므로 recipe summary.json 에서.

실행:
    uv run python poc/workflow_2/align_digest.py
출력: stdout + <review_or_out_root>/digest_<batch_ts>.txt
"""

import json
import shutil
import tempfile
from pathlib import Path

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_review import (
    STATUS_ORDER,
    _all_recipe_dirs,
    _iter_rows,
    _latest_batch_summary,
    _recipe_dirs_from_batch,
)

# 목록 출력 상한 — 너무 길면 붙여넣기 부담. 잘리면 "(+N more)" 로 명시 (no silent caps).
MAX_LIST_ROWS = 40
# best_scale 히스토그램 버킷 (COMPARE_SCALES + 확대 방향까지 — 1.0 초과가 나오면 C4 확정 신호).
_SCALE_BUCKETS = (0.6, 0.75, 0.85, 1.0, 1.2, 1.4)
# winner_score 가 이 미만이면 "낮음" 으로 센다. STRUCTURE_POLICY.adjust_threshold 와 맞춰야 정확하지만
# import 부작용을 피해 상수로 둔다 (summary.json.thresholds 에서 실제값 확인 가능).
_LOW_SCORE_HINT = 0.35


def _fmt(v, nd: int = 2) -> str:
    """None 안전 숫자 포매팅."""
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return "-"


def _winner_payload(row: dict) -> dict | None:
    """winner_modality 에 해당하는 om/sem payload dict (없으면 None)."""
    wm = row.get("winner_modality")
    if wm in ("om", "sem"):
        p = row.get(wm)
        if isinstance(p, dict):
            return p
    return None


def _recipe_tag(row: dict) -> str:
    return f"{row.get('eqp_id')}/{row.get('class_name')}/{row.get('recipe_id')}"


def _nearest_bucket(scale: float) -> float:
    return min(_SCALE_BUCKETS, key=lambda b: abs(b - scale))


def build_digest(out_root: Path, *, batch_summary_path: Path | None = None) -> str:
    """집계 텍스트를 만들어 반환하고 digest 파일로도 저장한다."""
    out_root = Path(out_root)
    if batch_summary_path is None:
        batch_summary_path = _latest_batch_summary(out_root)

    if batch_summary_path is not None:
        recipe_dirs = _recipe_dirs_from_batch(out_root, batch_summary_path)
        batch_ts = batch_summary_path.stem.replace("batch_summary_", "")
    else:
        recipe_dirs = _all_recipe_dirs(out_root)
        batch_ts = "unknown"

    if not recipe_dirs:
        return f"[ERROR] digest 할 recipe 결과 없음: {out_root}"

    # 누적 집계.
    status_counts: dict[str, int] = {}
    scale_hist: dict[float, int] = {b: 0 for b in _SCALE_BUCKETS}
    scale_na = 0
    scores: list[float] = []
    low_score = 0
    suspects: list[dict] = []
    not_distinct: list[dict] = []
    om_box_yes = om_box_total = sem_box_yes = sem_box_total = 0
    per_recipe: list[str] = []
    total = 0

    for recipe_dir in recipe_dirs:
        # recipe 단위 카운트 (one-liner 용).
        rc: dict[str, int] = {}
        rtag = recipe_dir.name
        for row, _overlay in _iter_rows(recipe_dir):
            total += 1
            st = row.get("status", "ok")
            status_counts[st] = status_counts.get(st, 0) + 1
            rc[st] = rc.get(st, 0) + 1

            payload = _winner_payload(row)
            if payload is not None and isinstance(payload.get("best_scale"), (int, float)):
                scale_hist[_nearest_bucket(float(payload["best_scale"]))] += 1
            else:
                scale_na += 1

            sc = row.get("winner_score")
            if isinstance(sc, (int, float)):
                scores.append(float(sc))
                if sc < _LOW_SCORE_HINT:
                    low_score += 1

            if row.get("tool_label_suspect"):
                suspects.append({
                    "tag": _recipe_tag(row), "msr": row.get("msr_image"),
                    "dist": row.get("correction_magnitude_px"), "score": row.get("winner_score"),
                    "scale": (payload or {}).get("best_scale"), "mod": row.get("winner_modality"),
                })
            if st == "not_distinctive":
                not_distinct.append({
                    "tag": _recipe_tag(row), "msr": row.get("msr_image"),
                    "ratio": (payload or {}).get("distinctiveness_ratio"),
                    "scale": (payload or {}).get("best_scale"), "mod": row.get("winner_modality"),
                })

        # rcp_box (행에 없음 → summary.json).
        box_str = "box(om=? sem=?)"
        sj = recipe_dir / "summary.json"
        if sj.is_file():
            try:
                summ = json.loads(sj.read_text(encoding="utf-8"))
                box = summ.get("rcp_box", {})
                om_d, sem_d = box.get("om_detected"), box.get("sem_detected")
                if om_d is not None:
                    om_box_total += 1
                    om_box_yes += 1 if om_d else 0
                if sem_d is not None:
                    sem_box_total += 1
                    sem_box_yes += 1 if sem_d else 0
                box_str = f"box(om={'Y' if om_d else 'N'} sem={'Y' if sem_d else 'N'})"
            except Exception:
                pass

        rc_str = " ".join(f"{k}={v}" for k, v in sorted(rc.items()))
        per_recipe.append(f"  {rtag}  n={sum(rc.values())}  {rc_str}   {box_str}")

    # ---- 텍스트 조립 ----
    L: list[str] = []
    L.append("=" * 60)
    L.append(f"ALIGN CORRECTION DIGEST   batch={batch_ts}")
    L.append(f"recipes={len(recipe_dirs)}  images={total}")
    L.append("=" * 60)

    L.append("\n--- GLOBAL STATUS COUNTS (worst-first) ---")
    seen = set()
    for st in STATUS_ORDER:
        if st == "suspect_success":
            continue  # status 값이 아니라 별도 집계.
        if st in status_counts:
            n = status_counts[st]
            L.append(f"  {st:<20}: {n:>4}  ({n / total * 100:4.1f}%)")
            seen.add(st)
    for st, n in sorted(status_counts.items()):  # STATUS_ORDER 에 없는 status 도 누락 없이.
        if st not in seen:
            L.append(f"  {st:<20}: {n:>4}  ({n / total * 100:4.1f}%)  [unlisted]")
    L.append(f"  {'suspect_success':<20}: {len(suspects):>4}  (S 라벨 & dist 큼; status 와 중복 가능)")

    L.append("\n--- RCP KEY box detection (C1 key 품질 프록시) ---")
    if om_box_total:
        L.append(f"  OM  box detected: {om_box_yes}/{om_box_total} recipes  (fallback crop: {om_box_total - om_box_yes})")
    if sem_box_total:
        L.append(f"  SEM box detected: {sem_box_yes}/{sem_box_total} recipes  (fallback crop: {sem_box_total - sem_box_yes})")
    if not (om_box_total or sem_box_total):
        L.append("  (summary.json 없음 — box 검출 통계 불가)")

    L.append("\n--- WINNER best_scale 분포 (C4 scale-miss check) ---")
    hist_str = "  ".join(f"{b:.2f}:{scale_hist[b]}" for b in _SCALE_BUCKETS)
    L.append(f"  {hist_str}   n/a:{scale_na}")
    L.append("  * 1.00 에 몰리면 확대방향(>1.0) 누락 의심 (COMPARE_SCALES max=1.0; base matcher 는 1.2/1.4 도 봄)")
    L.append("  * 1.20/1.40 에 값이 잡히면(현재는 못 봄) C4 확정 — correction path scale band 확장 필요")

    L.append("\n--- WINNER score 분포 ---")
    if scores:
        ss = sorted(scores)
        med = ss[len(ss) // 2]
        L.append(f"  min/median/max: {_fmt(ss[0])} / {_fmt(med)} / {_fmt(ss[-1])}   "
                 f"(<{_LOW_SCORE_HINT} 낮음: {low_score}/{len(scores)})")
        L.append("  * 정확한 임계는 recipe summary.json.thresholds['STRUCTURE_POLICY.adjust_threshold'] 참고")
    else:
        L.append("  (winner_score 없음)")

    def _emit_list(title: str, rows: list[dict], fmt) -> None:
        L.append(f"\n--- {title} ({len(rows)}) ---")
        if not rows:
            L.append("  (없음)")
            return
        rows_sorted = sorted(rows, key=lambda r: (r.get("dist") or r.get("ratio") or 0), reverse=True)
        for r in rows_sorted[:MAX_LIST_ROWS]:
            L.append(fmt(r))
        if len(rows_sorted) > MAX_LIST_ROWS:
            L.append(f"  (+{len(rows_sorted) - MAX_LIST_ROWS} more — 상한 {MAX_LIST_ROWS})")

    _emit_list(
        "SUSPECT_SUCCESS (S label, dist 큼 — 최우선)", suspects,
        lambda r: f"  {r['tag']}  {r['msr']}  dist={_fmt(r['dist'], 0)}px "
                  f"score={_fmt(r['score'])} scale={_fmt(r['scale'])} mod={(r['mod'] or '-').upper()}",
    )
    _emit_list(
        "NOT_DISTINCTIVE (ratio 나쁨)", not_distinct,
        lambda r: f"  {r['tag']}  {r['msr']}  ratio={_fmt(r['ratio'])} "
                  f"scale={_fmt(r['scale'])} mod={(r['mod'] or '-').upper()}",
    )

    L.append("\n--- PER-RECIPE one-liner ---")
    L.extend(per_recipe)
    L.append("\n" + "=" * 60)

    text = "\n".join(L)

    # 파일로도 저장 — review 폴더가 있으면 거기, 없으면 out_root.
    review_dir = out_root / f"review_{batch_ts}"
    dest_dir = review_dir if review_dir.is_dir() else out_root
    digest_path = dest_dir / f"digest_{batch_ts}.txt"
    try:
        digest_path.write_text(text, encoding="utf-8")
        text += f"\n[INFO] digest 저장: {digest_path}"
    except Exception as exc:
        text += f"\n[WARNING] digest 파일 저장 실패: {exc}"
    return text


# ====================================================================
# 엔트리 + Mac self-test.
# ====================================================================


def _self_test() -> bool:
    """align_review 합성 fixture 로 digest 가 예외 없이 돌고 핵심 섹션이 나오는지 검증."""
    from poc.workflow_3.vision.align_review import _build_synthetic_fixture
    tmp = Path(tempfile.mkdtemp(prefix="align_digest_selftest_"))
    try:
        out_root = tmp / "align_correction"
        out_root.mkdir(parents=True)
        bp = _build_synthetic_fixture(out_root)
        text = build_digest(out_root, batch_summary_path=bp)
        assert "GLOBAL STATUS COUNTS" in text, "status 섹션 누락"
        assert "best_scale" in text, "scale 섹션 누락"
        assert "SUSPECT_SUCCESS" in text, "suspect 섹션 누락"
        assert "PER-RECIPE" in text, "per-recipe 섹션 누락"
        print(text)
        print("\n[INFO] self-test 통과 — digest 텍스트 생성 검증 완료.")
        return True
    except AssertionError as exc:
        print(f"[ERROR] self-test 실패: {exc}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> str:
    out_root = DEBUG_IMAGE_DIR / "align_correction"
    if out_root.is_dir() and (
        _latest_batch_summary(out_root) is not None or list(out_root.glob("*/results.jsonl"))
    ):
        print(build_digest(out_root))
        return "success"
    print("[WARNING] 실제 align_correction 결과 없음 — 합성 self-test 로 대체합니다.\n")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
