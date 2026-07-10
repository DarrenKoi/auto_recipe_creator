"""Phase 1.5 검색 벤치 스모크 (순수 메트릭 + stub searcher e2e, 서버 불필요).

실행:
    uv run python -m side_projects.document_extraction.benchmark.test_retrieval_benchmark_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.benchmark.retrieval_golden import (
    GoldenQuery,
    load_golden,
    matcher_matches_hit,
    validate_query,
    write_example_golden,
)
from side_projects.document_extraction.benchmark.retrieval_metrics import (
    aggregate_by_tier,
    comparison_matrix,
    digest_line,
    evaluate_arm,
    hit_rank,
    parser_loss_recovery,
)
from side_projects.document_extraction.benchmark.run_retrieval_benchmark import (
    run_retrieval_benchmark,
)


def _hit(chunk_id: str, screenshot_id: str = "", region_type: str = "other") -> dict:
    return {"chunk_id": chunk_id, "_id": chunk_id,
            "screenshot_id": screenshot_id, "region_type": region_type}


def test_example_golden_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "golden.json"
        write_example_golden(path)
        queries = load_golden(path)
        assert len(queries) == 5
        assert sum(1 for q in queries if q.tier == "chart_only") == 2
    print("[PASS] test_example_golden_roundtrip")


def test_validation_catches_errors() -> None:
    assert validate_query({"query_id": "q1", "tier": "chart_only",
                           "query_text": "x", "relevant": [{"chunk_id": "c1"}]}) == []
    errors = validate_query({"query_id": "", "tier": "nope", "query_text": "",
                             "relevant": [{"bogus": 1}]})
    assert len(errors) >= 4, errors
    # query_id 중복은 load 에서 잡힘
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.json"
        entry = {"query_id": "q1", "tier": "text", "query_text": "x",
                 "relevant": [{"chunk_id": "c1"}]}
        path.write_text(json.dumps({"queries": [entry, entry]}), encoding="utf-8")
        try:
            load_golden(path)
            raise AssertionError("query_id 중복은 ValueError 여야 함")
        except ValueError:
            pass
    print("[PASS] test_validation_catches_errors")


def test_matcher_semantics() -> None:
    hit = _hit("doc1_s003_c001", "doc1_s003", "chart_summary")
    assert matcher_matches_hit({"chunk_id": "doc1_s003_c001"}, hit)
    assert not matcher_matches_hit({"chunk_id": "other"}, hit)
    assert matcher_matches_hit({"screenshot_id": "doc1_s003"}, hit)
    assert matcher_matches_hit(
        {"screenshot_id": "doc1_s003", "region_type": "chart_summary"}, hit)
    assert not matcher_matches_hit(
        {"screenshot_id": "doc1_s003", "region_type": "table_row"}, hit)
    # chunk_id 가 있으면 정확 일치만(스크린샷 무시)
    assert not matcher_matches_hit(
        {"chunk_id": "other", "screenshot_id": "doc1_s003"}, hit)
    print("[PASS] test_matcher_semantics")


def _queries() -> list[GoldenQuery]:
    return [
        GoldenQuery("q1", "chart_only", "차트 질의 1",
                    relevant=[{"chunk_id": "c_a"}]),
        GoldenQuery("q2", "chart_only", "차트 질의 2",
                    relevant=[{"chunk_id": "c_b"}]),
        GoldenQuery("q3", "text", "텍스트 질의",
                    relevant=[{"chunk_id": "t_a"}]),
    ]


def test_metrics_math() -> None:
    queries = _queries()
    hits_by_query = {
        "q1": [_hit("x"), _hit("c_a")],   # rank 2
        "q2": [_hit("x"), _hit("y")],     # miss
        "q3": [_hit("t_a")],              # rank 1
    }
    assert hit_rank(queries[0], hits_by_query["q1"]) == 2
    records = evaluate_arm(queries, hits_by_query, k=5)
    agg = aggregate_by_tier(records, k=5)
    assert agg["chart_only"]["n"] == 2
    assert abs(agg["chart_only"]["recall_at_k"] - 0.5) < 1e-9
    assert abs(agg["chart_only"]["mrr"] - 0.25) < 1e-9   # (1/2 + 0) / 2
    assert agg["overall"]["n"] == 3
    assert abs(agg["overall"]["mrr"] - (0.5 + 0.0 + 1.0) / 3) < 1e-9
    # k 컷: rank 2 는 k=1 에서 miss
    records_k1 = evaluate_arm(queries, hits_by_query, k=1)
    agg_k1 = aggregate_by_tier(records_k1, k=1)
    assert abs(agg_k1["chart_only"]["recall_at_k"] - 0.0) < 1e-9
    print("[PASS] test_metrics_math")


def test_parser_loss_recovery_math() -> None:
    queries = _queries()
    baseline = evaluate_arm(queries, {
        "q1": [], "q2": [], "q3": [_hit("t_a")],   # chart 2건 모두 miss
    }, k=5)
    candidate = evaluate_arm(queries, {
        "q1": [_hit("c_a")], "q2": [], "q3": [_hit("t_a")],   # 1건 회수
    }, k=5)
    rec = parser_loss_recovery(baseline, candidate, tier="chart_only")
    assert rec["baseline_misses"] == 2
    assert rec["recovered"] == 1
    assert abs(rec["recovery_rate"] - 0.5) < 1e-9
    assert rec["recovered_query_ids"] == ["q1"]
    # baseline miss 0 -> rate 0.0 (분모 0 안전)
    rec_none = parser_loss_recovery(candidate, candidate, tier="text")
    assert rec_none["baseline_misses"] == 0 and rec_none["recovery_rate"] == 0.0
    print("[PASS] test_parser_loss_recovery_math")


def test_run_benchmark_e2e_stub_searchers() -> None:
    """stub searcher 3-arm -> 집계/recovery/matrix/digest/파일 저장 e2e."""
    queries = _queries()

    def bm25(query_text: str) -> list[dict]:
        return [_hit("t_a")] if "텍스트" in query_text else [_hit("x")]

    def dense(query_text: str) -> list[dict]:
        if query_text == "차트 질의 1":
            return [_hit("c_a")]
        return [_hit("t_a")] if "텍스트" in query_text else []

    def hybrid(query_text: str) -> list[dict]:
        return dense(query_text) or bm25(query_text)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "results"
        result = run_retrieval_benchmark(
            queries, {"bm25": bm25, "dense": dense, "hybrid": hybrid},
            top_k=5, out_dir=out_dir,
        )
        # bm25 는 chart 전멸, dense/hybrid 가 q1 회수 -> recovery 1/2
        assert result["recoveries"]["dense"]["recovered"] == 1
        assert result["recoveries"]["dense"]["baseline_misses"] == 2
        assert result["aggregates"]["bm25"]["chart_only"]["recall_at_k"] == 0.0
        assert result["aggregates"]["hybrid"]["chart_only"]["recall_at_k"] == 0.5
        assert result["digest"].startswith("[DIGEST] ")
        assert "recovery[dense]=1/2" in result["digest"]
        for name in ("retrieval_scores.json", "retrieval_matrix.md", "digest.txt"):
            assert (out_dir / name).exists(), name
        matrix = (out_dir / "retrieval_matrix.md").read_text(encoding="utf-8")
        assert "chart_only" in matrix and "overall" in matrix
    print("[PASS] test_run_benchmark_e2e_stub_searchers")


def test_matrix_and_digest_pure() -> None:
    records = evaluate_arm(_queries(), {"q1": [_hit("c_a")], "q2": [], "q3": []}, k=5)
    aggs = {"armA": aggregate_by_tier(records, k=5)}
    matrix = comparison_matrix(aggs, k=5)
    assert "| chart_only |" in matrix and "armA" in matrix
    digest = digest_line(aggs, {}, k=5)
    assert digest.startswith("[DIGEST] armA:")
    print("[PASS] test_matrix_and_digest_pure")


def main() -> int:
    test_example_golden_roundtrip()
    test_validation_catches_errors()
    test_matcher_semantics()
    test_metrics_math()
    test_parser_loss_recovery_math()
    test_run_benchmark_e2e_stub_searchers()
    test_matrix_and_digest_pure()
    print("\n[INFO] 모든 retrieval 벤치 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
