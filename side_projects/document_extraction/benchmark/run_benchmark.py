"""벤치마크 실행 엔트리포인트 (CLI 인자 없음).

벤치 폴더 레이아웃:
    <BENCH_DIR>/
      ground_truth/<screenshot_id>.json          # 사람이 만든 정답
      extractions/<pipeline_name>/<screenshot_id>.json   # 추출 산출물(raw_evidence)

각 pipeline x screenshot 을 채점해 다음을 쓴다:
    <BENCH_DIR>/results/scores.json               # 전체 점수
    <BENCH_DIR>/results/comparison_matrix.md      # 파이프라인 비교 표
    <BENCH_DIR>/results/acceptance.json           # acceptance criteria 판정

추출 자체는 사내 PC 에서 돌려 extractions/ 에 떨궈 두고, 채점은 어디서나 실행한다.

실행:
    uv run python -m side_projects.document_extraction.benchmark.run_benchmark
"""

import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.benchmark import scorer
from side_projects.document_extraction.benchmark.ground_truth import GroundTruth


# === 실행 전 매번 채워 넣을 것 =================================================
BENCH_DIR: Path = Path("")   # 예: Path("side_projects/document_extraction/benchmark/_bench")
# ==============================================================================


def run_benchmark(bench_dir: Path) -> dict:
    """벤치 폴더를 채점하고 결과 파일을 쓴다. 집계 dict 를 반환."""
    gt_dir = bench_dir / "ground_truth"
    ex_root = bench_dir / "extractions"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ground_truth 폴더가 없습니다: {gt_dir}")
    if not ex_root.is_dir():
        raise FileNotFoundError(f"extractions 폴더가 없습니다: {ex_root}")

    # 정답 로드
    ground_truths: dict[str, GroundTruth] = {}
    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = GroundTruth.load(gt_path)
        sid = gt.screenshot_id or gt_path.stem
        ground_truths[sid] = gt
    if not ground_truths:
        raise FileNotFoundError(f"정답 파일이 없습니다: {gt_dir}")

    pipelines = sorted(p for p in ex_root.iterdir() if p.is_dir())
    if not pipelines:
        raise FileNotFoundError(f"extractions/<pipeline> 가 없습니다: {ex_root}")

    print(
        f"[INFO] 벤치 시작: GT {len(ground_truths)}장, "
        f"pipeline {len(pipelines)}개 ({', '.join(p.name for p in pipelines)})"
    )

    all_scores: dict[str, list] = {}
    scores_json: dict[str, list] = {}

    for pipe_dir in pipelines:
        pipe_name = pipe_dir.name
        pipe_scores = []
        for sid, gt in ground_truths.items():
            ex_path = pipe_dir / f"{sid}.json"
            if not ex_path.exists():
                print(f"[WARNING] {pipe_name}: {sid} 추출 산출물 없음 -> 스킵")
                continue
            extraction = json.loads(ex_path.read_text(encoding="utf-8"))
            score = scorer.score_screenshot(extraction, gt)
            pipe_scores.append(score)
            print(
                f"[INFO]   {pipe_name}/{sid}: text={score.text_recall} "
                f"table={score.table_accuracy} chart={score.chart_understanding} "
                f"layout={score.layout_accuracy} rag={score.rag_readiness} "
                f"halluc={score.hallucination_rate}"
            )
        all_scores[pipe_name] = pipe_scores
        scores_json[pipe_name] = [s.to_dict() for s in pipe_scores]

    # 결과 저장
    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "scores.json").write_text(
        json.dumps(scores_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix_md = scorer.comparison_matrix(all_scores)
    (results_dir / "comparison_matrix.md").write_text(
        "# Benchmark Comparison Matrix\n\n" + matrix_md + "\n", encoding="utf-8"
    )

    acceptance = {
        name: scorer.check_acceptance(scores) for name, scores in all_scores.items()
    }
    (results_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[INFO] 결과 저장 -> {results_dir}")
    print("\n" + matrix_md)
    for name, acc in acceptance.items():
        flag = "PASS" if acc.get("all_passed") else "FAIL"
        print(f"[INFO] acceptance[{name}] = {flag} ({acc.get('passed')})")

    return {"scores": all_scores, "acceptance": acceptance}


def main() -> int:
    if str(BENCH_DIR) in {"", "."}:
        print(
            "[ERROR] BENCH_DIR 가 비어 있습니다. run_benchmark.py 상단 상수를 수정하세요."
        )
        return 1
    bench_dir = BENCH_DIR.expanduser().resolve()
    print(f"[INFO] BENCH_DIR = {bench_dir}")
    try:
        run_benchmark(bench_dir)
    except Exception as exc:
        print(f"[ERROR] 벤치 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
