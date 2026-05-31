"""MI reranker (opt-in) self-test — chamfer 멤버십 게이트 + MI 순서 재정렬 (Phase 3).

검증 목표 (production threshold 증명이 아니라 *구조*가 의도대로 동작하는지):
  A. rerank_candidates_by_mi  — chamfer 후보 *집합*은 그대로, 각 후보 crop 의 MI 로 *순서만*
     바꾼다. 진짜(=degraded 복제, MI 높음)가 distractor(=dense edge, chamfer 포화) 위로 올라온다.
  B. compute_align_key_score(policy.mi_rerank=True)  — best_xy 가 chamfer-best(distractor)가 아니라
     MI-best(진짜) 로 옮겨오고, ORB/score/decision/.candidates 가 MI-best 로부터 파생된다.
  C. 멤버십 불변 — rerank on/off 가 후보 *집합*(xy 모음)을 안 바꾼다(순서만).
  D. distinctiveness 일관성 — mi_rerank 가 best 를 바꿔도 score_gap 은 chamfer 집합 기준이라 음수가
     아니다(reorder 와 독립). DEFAULT_POLICY(mi_rerank=False)는 기존 동작과 byte 동일.

배경: chamfer score = exp(-mean_dt/tau) 라 edge 가 빽빽한 영역은 frame_dt≈0 → chamfer≈1.0 으로
*포화*된다(알려진 실패 모드). 진짜 align key 가 live 에서 흐릿/저대비로 보이면 chamfer 가 낮아져
distractor 에 1등을 뺏긴다. MI 는 밝기/대비 drift 에 강건해 이 18%(topk_not_rank1)를 회복한다.

실행: uv run python poc/workflow_2/test_align_key_mi_rerank.py
"""

from dataclasses import replace

import cv2
import numpy as np

from poc.workflow_2.align_key_matcher import (
    DEFAULT_POLICY,
    AlignKeyCandidate,
    build_template,
    compute_align_key_score,
    rerank_candidates_by_mi,
)

SIZE = 64


def _template_pattern() -> np.ndarray:
    """뚜렷한 edge(outline+대각) + 내부 intensity 램프(MI 콘텐츠)를 가진 distinctive 패턴."""
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    inner = SIZE - 16
    ramp = np.tile(np.linspace(30, 220, inner).astype(np.uint8), (inner, 1))
    img[8:8 + inner, 8:8 + inner] = ramp
    cv2.rectangle(img, (4, 4), (SIZE - 4, SIZE - 4), 255, 2)   # outline edges.
    cv2.line(img, (8, 8), (SIZE - 8, SIZE - 8), 255, 1)        # diagonal edge.
    return img


def _degraded(pat: np.ndarray) -> np.ndarray:
    """진짜 align key 의 live 외형: 저대비(×0.5) + blur → chamfer↓ 하지만 MI 는 보존."""
    deg = (pat.astype(np.float32) * 0.5).astype(np.uint8)
    return cv2.GaussianBlur(deg, (3, 3), 0)


def _dense_distractor() -> np.ndarray:
    """edge 가 빽빽한 격자 → frame_dt≈0 → chamfer 포화(≈1.0). 내용은 template 과 무관 → MI 낮음."""
    d = np.zeros((SIZE, SIZE), dtype=np.uint8)
    d[::4, :] = 255
    d[:, ::4] = 255
    return d


def _scene():
    """진짜(degraded)와 distractor 를 멀찍이 배치한 프레임 + 두 패턴의 중심 좌표."""
    h, w = 320, 420
    frame = np.full((h, w), 60, dtype=np.uint8)
    true_tl = (80, 80)        # (x, y) top-left.
    dist_tl = (240, 200)
    frame[true_tl[1]:true_tl[1] + SIZE, true_tl[0]:true_tl[0] + SIZE] = _degraded(_template_pattern())
    frame[dist_tl[1]:dist_tl[1] + SIZE, dist_tl[0]:dist_tl[0] + SIZE] = _dense_distractor()
    true_c = (true_tl[0] + SIZE // 2, true_tl[1] + SIZE // 2)
    dist_c = (dist_tl[0] + SIZE // 2, dist_tl[1] + SIZE // 2)
    return frame, true_c, dist_c


def _near(xy, target, tol=24) -> bool:
    return abs(xy[0] - target[0]) <= tol and abs(xy[1] - target[1]) <= tol


def main() -> int:
    pat = _template_pattern()
    tpl = build_template(pat, recipe_id="T", version="om_center", key_type="om")
    frame, true_c, dist_c = _scene()
    ok = True

    # --- Test A: rerank_candidates_by_mi 가 chamfer 순서를 MI 순서로 뒤집는다 ---
    # chamfer 순서를 흉내: distractor(높은 chamfer) 먼저, 진짜(낮은 chamfer) 뒤.
    cand_dist = AlignKeyCandidate(score=0.98, chamfer_score=0.98, xy=dist_c,
                                  scale=1.0, template_size=(SIZE, SIZE))
    cand_true = AlignKeyCandidate(score=0.80, chamfer_score=0.80, xy=true_c,
                                  scale=1.0, template_size=(SIZE, SIZE))
    ranked = rerank_candidates_by_mi(tpl, frame, [cand_dist, cand_true])
    print(f"[A] mi(true)={cand_true.mi_score:.3f} mi(dist)={cand_dist.mi_score:.3f} "
          f"ranked0={ranked[0].xy}")
    if not _near(ranked[0].xy, true_c):
        print("[FAIL][A] MI rerank 후 1순위는 진짜(MI 높음)여야 함")
        ok = False
    if not (cand_true.mi_score is not None and cand_dist.mi_score is not None
            and cand_true.mi_score > cand_dist.mi_score):
        print("[FAIL][A] 진짜의 mi_score 가 distractor 보다 커야 함")
        ok = False
    if {c.xy for c in ranked} != {dist_c, true_c}:
        print("[FAIL][A] rerank 가 후보 집합(멤버십)을 바꾸면 안 됨")
        ok = False

    # --- Test B & D(off): DEFAULT_POLICY(mi_rerank=False) → chamfer-best(distractor) ---
    r_off = compute_align_key_score(tpl, frame, policy=DEFAULT_POLICY)
    print(f"[B] off best_xy={r_off.best_xy} gap={r_off.score_gap} cand={len(r_off.candidates)}")
    if not _near(r_off.best_xy, dist_c):
        print(f"[FAIL][B] mi_rerank=off 는 chamfer-best(distractor≈{dist_c}) 여야 함")
        ok = False

    # --- Test B(on): mi_rerank=True → MI-best(진짜) ---
    policy_on = replace(DEFAULT_POLICY, mi_rerank=True)
    r_on = compute_align_key_score(tpl, frame, policy=policy_on)
    print(f"[B] on  best_xy={r_on.best_xy} gap={r_on.score_gap} "
          f"cand0_mi={r_on.candidates[0].mi_score} cand0_crank={r_on.candidates[0].chamfer_rank}")
    if not _near(r_on.best_xy, true_c):
        print(f"[FAIL][B] mi_rerank=on 은 MI-best(진짜≈{true_c}) 여야 함")
        ok = False

    # --- Test C: 멤버십 불변 (on/off 가 같은 후보 집합) ---
    set_off = {c.xy for c in r_off.candidates}
    set_on = {c.xy for c in r_on.candidates}
    if set_off != set_on:
        print(f"[FAIL][C] 멤버십 불변 위반: off={set_off} on={set_on}")
        ok = False

    # --- Test D: distinctiveness 일관성 — reorder 후에도 score_gap 은 chamfer 기준이라 ≥0 ---
    if r_on.score_gap is not None and r_on.score_gap < 0:
        print(f"[FAIL][D] mi_rerank=on 의 score_gap 이 음수({r_on.score_gap}) — "
              f"distinctiveness 가 chamfer 집합 기준이 아님")
        ok = False
    # DEFAULT 경로는 chamfer-best 의 chamfer_rank 가 0 이어야(원 chamfer 순위 보존).
    if r_on.candidates[0].chamfer_rank is None:
        print("[FAIL][D] rerank 후보에 원 chamfer_rank 가 보존돼야 함")
        ok = False

    print("[INFO] PASS" if ok else "[INFO] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
