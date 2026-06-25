"""raw_evidence/*.json -> Marp deck(.md) 빌드 엔트리 (CLI 인자 없음).

extract_screenshot 이 만든 raw_evidence 폴더를 읽어 한 문서의 Marp deck 을 만든다.
Stage 6 렌더(marp-cli)/Stage 7 SSIM 검증은 office TODO (아래 주석 참고).

실행 전 상단 상수 수정:
    RAW_EVIDENCE_DIR  raw_evidence JSON 들이 있는 폴더
    OUTPUT_MD         생성할 deck.md 경로

실행:
    uv run python -m side_projects.document_extraction.marp.build_marp
"""

import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.schemas import ExtractionResult
from side_projects.document_extraction.marp.generate import results_to_deck


# === 실행 전 매번 채워 넣을 것 =================================================
RAW_EVIDENCE_DIR: Path = Path("")   # 예: Path(r"C:\...\_rag\raw_evidence")
OUTPUT_MD: Path = Path("")          # 예: Path(r"C:\...\_rag\deck.md")
# crop_lookups: {screenshot_id -> {region_id -> 이미지경로}} (래스터 재삽입용; 선택)
CROP_LOOKUPS: dict = {}
# ==============================================================================


def build_deck(raw_evidence_dir: Path, output_md: Path, crop_lookups: dict) -> int:
    """raw_evidence 폴더의 JSON 들을 screenshot_index 순으로 deck 으로 합친다."""
    files = sorted(raw_evidence_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"raw_evidence JSON 이 없습니다: {raw_evidence_dir}")

    results = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(ExtractionResult.from_dict(data))
    results.sort(key=lambda r: r.screenshot_index)

    deck = results_to_deck(results, crop_lookups=crop_lookups)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(deck, encoding="utf-8")
    print(f"[INFO] Marp deck 생성: {len(results)} 슬라이드 -> {output_md}")
    # Stage 6/7 (구현됨; 원본 캡처가 있으면 이어서 호출):
    #   from .render import render_deck            # marp-cli -> png/pptx/pdf/html
    #   from .verify import verify_and_downgrade   # 슬라이드별 SSIM -> 저충실도 자동 강등
    #   verify_and_downgrade(results, output_md, capture_paths, out_dir=output_md.parent)
    return len(results)


def main() -> int:
    if str(RAW_EVIDENCE_DIR) in {"", "."} or str(OUTPUT_MD) in {"", "."}:
        print("[ERROR] RAW_EVIDENCE_DIR / OUTPUT_MD 가 비어 있습니다. 상단 상수를 수정하세요.")
        return 1
    raw_dir = RAW_EVIDENCE_DIR.expanduser().resolve()
    out_md = OUTPUT_MD.expanduser().resolve()
    print(f"[INFO] RAW_EVIDENCE_DIR = {raw_dir}")
    print(f"[INFO] OUTPUT_MD        = {out_md}")
    try:
        build_deck(raw_dir, out_md, CROP_LOOKUPS)
    except Exception as exc:
        print(f"[ERROR] deck 빌드 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
