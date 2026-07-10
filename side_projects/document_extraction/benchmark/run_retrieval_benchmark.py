"""Phase 1.5 검색 벤치 실행 엔트리 (CLI 인자 없음).

golden 질의셋(retrieval_golden)을 3-arm(bm25 / dense / hybrid)으로 검색해
tier 별 Recall@k / MRR + parser-loss recovery(chart_only, baseline=bm25)를
채점한다. 결과는 파일 + 콘솔 [DIGEST] 한 줄(workflow_2 컨벤션 - 사외 전달용).

출력 (<OUT_DIR>/):
    retrieval_scores.json    arm 별 질의 레코드 + tier 집계 + recovery
    retrieval_matrix.md      tier x arm 비교 표
    digest.txt               [DIGEST] 한 줄

사내 실행(OpenSearch + 색인 완료 후):
    uv run python -m side_projects.document_extraction.benchmark.run_retrieval_benchmark

searchers 를 주입하면(테스트/오프라인) OpenSearch 없이도 채점 로직이 돈다.
"""

import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.benchmark.retrieval_golden import (
    GoldenQuery,
    load_golden,
)
from side_projects.document_extraction.benchmark.retrieval_metrics import (
    aggregate_by_tier,
    comparison_matrix,
    digest_line,
    evaluate_arm,
    parser_loss_recovery,
)


# === 실행 전 매번 채워 넣을 것 =================================================
GOLDEN_JSON: Path = Path("")   # 예: Path(r"C:\...\_bench\golden_retrieval_queries.json")
OUT_DIR: Path = Path("")       # 비우면 <GOLDEN_JSON 폴더>/retrieval_results
TOP_K: int = 5                 # Recall@k 의 k
ARM_SIZE: int = 20             # arm 당 후보 수
# ==============================================================================

# parser-loss recovery 의 baseline arm(텍스트 arm). 나머지 arm 별로 회수율 계산.
BASELINE_ARM = "bm25"
RECOVERY_TIER = "chart_only"


def build_default_searchers(*, arm_size: int = ARM_SIZE) -> dict:
    """OpenSearch 3-arm searcher 를 만든다(사내; lazy import).

    반환: {arm_name -> callable(query_text) -> hits}
    """
    from side_projects.document_extraction.extraction.embeddings import EmbeddingClient
    from side_projects.document_extraction.extraction.hybrid_search import (
        build_bm25_query,
        build_knn_query,
        hits_from_response,
        search_hybrid,
    )
    from side_projects.document_extraction.extraction.opensearch_index import (
        OpenSearchClient,
        resolve_index_name,
    )

    client = OpenSearchClient()
    embedder = EmbeddingClient()
    index_name = resolve_index_name()

    def bm25_arm(query_text: str) -> list[dict]:
        resp = client.search(index_name, build_bm25_query(query_text, size=arm_size))
        return hits_from_response(resp)

    def dense_arm(query_text: str) -> list[dict]:
        vec = embedder.embed_one(query_text)
        resp = client.search(index_name, build_knn_query(vec, size=arm_size))
        return hits_from_response(resp)

    def hybrid_arm(query_text: str) -> list[dict]:
        return search_hybrid(
            query_text, client=client, embedder=embedder,
            index_name=index_name, top_k=arm_size, arm_size=arm_size,
        )

    if not client.ping():
        raise RuntimeError("OpenSearch 에 연결할 수 없습니다 (사내 서버/env 확인).")
    return {"bm25": bm25_arm, "dense": dense_arm, "hybrid": hybrid_arm}


def run_retrieval_benchmark(
    queries: list[GoldenQuery],
    searchers: dict,
    *,
    top_k: int = TOP_K,
    baseline_arm: str = BASELINE_ARM,
    out_dir: Path | None = None,
) -> dict:
    """golden 질의를 arm 별로 검색/채점하고 결과 dict 를 반환(+파일 저장).

    searchers: {arm_name -> callable(query_text) -> hits}. baseline_arm 이
    searchers 에 없으면 recovery 는 건너뛴다(경고).
    """
    records_by_arm: dict[str, list[dict]] = {}
    aggregates: dict[str, dict] = {}

    for arm_name in sorted(searchers):
        search = searchers[arm_name]
        hits_by_query: dict[str, list[dict]] = {}
        for query in queries:
            try:
                hits_by_query[query.query_id] = search(query.query_text)
            except Exception as exc:
                print(f"[WARNING] {arm_name}/{query.query_id} 검색 실패 -> miss: {exc}")
                hits_by_query[query.query_id] = []
        records = evaluate_arm(queries, hits_by_query, k=top_k)
        records_by_arm[arm_name] = records
        aggregates[arm_name] = aggregate_by_tier(records, k=top_k)
        overall = aggregates[arm_name]["overall"]
        print(
            f"[INFO] arm={arm_name}: R@{top_k}={overall['recall_at_k']:.3f} "
            f"MRR={overall['mrr']:.3f} (n={overall['n']})"
        )

    recoveries: dict[str, dict] = {}
    if baseline_arm in records_by_arm:
        for arm_name, records in records_by_arm.items():
            if arm_name == baseline_arm:
                continue
            recoveries[arm_name] = parser_loss_recovery(
                records_by_arm[baseline_arm], records, tier=RECOVERY_TIER
            )
    else:
        print(f"[WARNING] baseline arm '{baseline_arm}' 없음 -> recovery 생략")

    matrix = comparison_matrix(aggregates, k=top_k)
    digest = digest_line(aggregates, recoveries, k=top_k)

    result = {
        "k": top_k,
        "baseline_arm": baseline_arm,
        "recovery_tier": RECOVERY_TIER,
        "aggregates": aggregates,
        "recoveries": recoveries,
        "records_by_arm": records_by_arm,
        "digest": digest,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval_scores.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "retrieval_matrix.md").write_text(
            "# Retrieval Benchmark (Phase 1.5)\n\n" + matrix + "\n\n" + digest + "\n",
            encoding="utf-8",
        )
        (out_dir / "digest.txt").write_text(digest + "\n", encoding="utf-8")
        print(f"[INFO] 결과 저장 -> {out_dir}")

    print("\n" + matrix)
    print("\n" + digest)
    return result


def main() -> int:
    if str(GOLDEN_JSON) in {"", "."}:
        print("[ERROR] GOLDEN_JSON 이 비어 있습니다. 상단 상수를 수정하세요.")
        print("        예시 템플릿: benchmark/golden_retrieval_queries.example.json")
        return 1
    golden_path = GOLDEN_JSON.expanduser().resolve()
    out_dir = (
        OUT_DIR.expanduser().resolve()
        if str(OUT_DIR) not in {"", "."}
        else golden_path.parent / "retrieval_results"
    )
    print(f"[INFO] GOLDEN_JSON = {golden_path}")
    print(f"[INFO] OUT_DIR     = {out_dir}")

    try:
        queries = load_golden(golden_path)
        searchers = build_default_searchers(arm_size=ARM_SIZE)
        run_retrieval_benchmark(
            queries, searchers, top_k=TOP_K, out_dir=out_dir
        )
    except Exception as exc:
        print(f"[ERROR] 벤치 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASELINE_ARM",
    "RECOVERY_TIER",
    "build_default_searchers",
    "run_retrieval_benchmark",
]
