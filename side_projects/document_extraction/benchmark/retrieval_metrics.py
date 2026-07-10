"""Phase 1.5 검색 벤치 메트릭 (순수).

질의별 hit rank 로부터 Recall@k(hit-rate), MRR 을 tier 별로 집계하고,
**parser-loss recovery** — baseline arm(텍스트/BM25)이 놓친 질의를 후보 arm 이
회수한 비율(PixelRAG 논문의 핵심 우위 지표를 우리 하네스 언어로 옮긴 것) — 를
잰다. Phase 1 은 bm25/dense/hybrid 3-arm 비교, Phase 2 에서 vision arm 이
붙으면 같은 함수로 즉시 A/B 가 된다.

모든 함수는 순수라 사외에서 검증된다. 검색 실행(arm -> hits)은 run_retrieval_benchmark 몫.
"""

from side_projects.document_extraction.benchmark.retrieval_golden import (
    GoldenQuery,
    TIERS,
    is_relevant,
)


def hit_rank(query: GoldenQuery, hits: list[dict]) -> int:
    """정답이 처음 등장하는 rank(1-기반). 없으면 0."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(query, hit):
            return rank
    return 0


def evaluate_arm(
    queries: list[GoldenQuery],
    hits_by_query: dict[str, list[dict]],
    *,
    k: int,
) -> list[dict]:
    """arm 1개의 질의별 채점 레코드를 만든다.

    레코드: {query_id, tier, rank(0=miss), hit_at_k, rr}
    hits_by_query 에 없는 질의는 miss 로 채점(검색 실패도 성적).
    """
    records: list[dict] = []
    for query in queries:
        rank = hit_rank(query, hits_by_query.get(query.query_id) or [])
        records.append(
            {
                "query_id": query.query_id,
                "tier": query.tier,
                "rank": rank,
                "hit_at_k": bool(rank and rank <= k),
                "rr": (1.0 / rank) if rank else 0.0,
            }
        )
    return records


def aggregate_by_tier(records: list[dict], *, k: int) -> dict:
    """질의별 레코드 -> tier 별 + overall 집계.

    반환: {tier: {"n", "recall_at_k", "mrr"}, ..., "overall": {...}}
    질의가 없는 tier 는 n=0, 점수 0.0 (분모 0 방지).
    """
    def _agg(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0, "recall_at_k": 0.0, "mrr": 0.0}
        return {
            "n": n,
            "recall_at_k": round(sum(1 for r in subset if r["hit_at_k"]) / n, 4),
            "mrr": round(sum(r["rr"] for r in subset) / n, 4),
        }

    result = {tier: _agg([r for r in records if r["tier"] == tier]) for tier in TIERS}
    result["overall"] = _agg(records)
    result["k"] = k
    return result


def parser_loss_recovery(
    baseline_records: list[dict],
    candidate_records: list[dict],
    *,
    tier: str = "chart_only",
) -> dict:
    """baseline arm 이 놓친(tier 한정) 질의를 후보 arm 이 회수한 비율.

    recovered = baseline miss(hit_at_k=False) AND candidate hit(hit_at_k=True).
    반환: {"tier", "baseline_misses", "recovered", "recovery_rate",
           "recovered_query_ids"}. baseline miss 가 0 이면 rate=0.0 (회수할 게 없음).
    """
    baseline_by_id = {r["query_id"]: r for r in baseline_records}
    candidate_by_id = {r["query_id"]: r for r in candidate_records}

    misses = [
        qid for qid, rec in baseline_by_id.items()
        if rec["tier"] == tier and not rec["hit_at_k"]
    ]
    recovered = [
        qid for qid in misses
        if (candidate_by_id.get(qid) or {}).get("hit_at_k")
    ]
    return {
        "tier": tier,
        "baseline_misses": len(misses),
        "recovered": len(recovered),
        "recovery_rate": round(len(recovered) / len(misses), 4) if misses else 0.0,
        "recovered_query_ids": sorted(recovered),
    }


def comparison_matrix(arm_aggregates: dict, *, k: int) -> str:
    """{arm_name -> aggregate} -> 마크다운 비교 표(순수)."""
    arms = sorted(arm_aggregates)
    lines = [
        "| tier | n | " + " | ".join(f"{a} R@{k} / MRR" for a in arms) + " |",
        "|---|---|" + "|".join("---" for _ in arms) + "|",
    ]
    tiers = list(TIERS) + ["overall"]
    for tier in tiers:
        first = arm_aggregates[arms[0]].get(tier) or {}
        cells = []
        for arm in arms:
            agg = arm_aggregates[arm].get(tier) or {}
            cells.append(f"{agg.get('recall_at_k', 0.0):.3f} / {agg.get('mrr', 0.0):.3f}")
        lines.append(f"| {tier} | {first.get('n', 0)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def digest_line(arm_aggregates: dict, recoveries: dict, *, k: int) -> str:
    """콘솔에서 그대로 복사해 전달할 한 줄 요약(workflow_2 [DIGEST] 컨벤션).

    recoveries: {candidate_arm_name -> parser_loss_recovery dict}
    """
    parts = []
    for arm in sorted(arm_aggregates):
        overall = arm_aggregates[arm].get("overall") or {}
        chart = arm_aggregates[arm].get("chart_only") or {}
        parts.append(
            f"{arm}: R@{k}={overall.get('recall_at_k', 0.0):.3f}"
            f"(chart {chart.get('recall_at_k', 0.0):.3f})"
        )
    for arm, rec in sorted(recoveries.items()):
        parts.append(
            f"recovery[{arm}]={rec['recovered']}/{rec['baseline_misses']}"
            f"({rec['recovery_rate']:.2f})"
        )
    return "[DIGEST] " + " | ".join(parts)


__all__ = [
    "aggregate_by_tier",
    "comparison_matrix",
    "digest_line",
    "evaluate_arm",
    "hit_rank",
    "parser_loss_recovery",
]
