"""스크린샷 1장 -> 추출 산출물 오케스트레이터 (Stage 1~8).

document_extraction 캡처 단계가 만든 page WebP 를 입력으로 받아:
    Stage 1 preprocess (크기 측정; 캡처가 이미 WebP 라 재인코딩은 생략)
    Stage 2 OCR (paddleocr-vl-1.5)
    Stage 3 layout (ui-venus)
    Stage 4 crop refine (mai-ui)         <- skeleton: 훅만, 기본 비활성
    Stage 5 merge evidence (순수 함수)
    Stage 6 synthesis (kimi-k2.6)
    Stage 7 human review packet           <- skeleton: 산출물 저장만
    Stage 8 RAG chunks + JSONL

CLI 인자 없음. 이 모듈 상단 상수를 직접 수정해서 사용한다(repo 컨벤션).
모델 서버가 없으면 OFFLINE 폴백으로 stub evidence 를 만들어 골격을 검증한다.

실행:
    uv run python -m side_projects.document_extraction.extraction.extract_screenshot
"""

import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction import merge, rag_chunks
from side_projects.document_extraction.extraction.models import StageRunner
from side_projects.document_extraction.extraction.schemas import ExtractionResult


# === 실행 전 매번 채워 넣을 것 =================================================
# 입력: 캡처 단계가 만든 page WebP 들이 들어 있는 폴더(문서 1개의 출력 폴더).
INPUT_IMAGE_DIR: Path = Path("")   # 예: Path(r"C:\...\extracted\presentation_A")
OUTPUT_DIR: Path = Path("")        # 예: Path(r"C:\...\extracted\presentation_A\_rag")
COLLECTION_ID: str = "default_collection"
ENABLE_CROP_REFINE: bool = False   # skeleton: 기본 비활성(office 캘리브레이션 후 활성)
ENABLE_SYNTHESIS: bool = True      # kimi 합성 스테이지 on/off
OFFLINE: bool | None = None        # None=env(DOC_EXTRACT_OFFLINE)로 결정, True/False 강제
# ==============================================================================


def _image_size(image_path: Path) -> tuple[int, int]:
    """이미지 width/height 를 반환한다(실패 시 0,0)."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return int(img.width), int(img.height)
    except Exception as exc:
        print(f"[WARNING] 이미지 크기 측정 실패({image_path.name}): {exc}")
        return 0, 0


def _infer_source_type(image_dir: Path) -> str:
    """폴더/파일명 힌트로 source type 을 추정한다(없으면 unknown)."""
    name = image_dir.name.lower()
    if any(k in name for k in ("ppt", "slide", "presentation")):
        return "powerpoint"
    if any(k in name for k in ("xls", "excel", "sheet", "workbook")):
        return "excel"
    if "pdf" in name:
        return "pdf"
    return "unknown"


def extract_one(
    image_path: Path,
    *,
    screenshot_index: int,
    document_id: str,
    collection_id: str,
    source_type_hint: str,
    runner: StageRunner,
    enable_crop_refine: bool = False,
    enable_synthesis: bool = True,
) -> ExtractionResult:
    """스크린샷 1장에 대한 Stage 1~8 을 실행한다."""
    width, height = _image_size(image_path)
    screenshot_id = f"{document_id}_s{screenshot_index:03d}"

    # Stage 2 / 3
    ocr = runner.run_ocr(str(image_path), width, height)
    layout = runner.run_layout(str(image_path), width, height)

    # layout 이 source_type 을 모르면 폴더 힌트로 보강
    if not (layout.get("source_type") or "").strip() or layout.get("source_type") == "unknown":
        layout["source_type"] = source_type_hint

    # Stage 4 (옵션): table/chart/formula/legend 영역만 crop 재인식
    if enable_crop_refine:
        _apply_crop_refine(runner, image_path, layout, ocr, width, height)

    # Stage 5: merge
    result = merge.merge_evidence(
        source_image=str(image_path),
        ocr=ocr,
        layout=layout,
        screenshot_index=screenshot_index,
        document_id=document_id,
        collection_id=collection_id,
        screenshot_id=screenshot_id,
    )

    # Stage 6: synthesis (옵션)
    if enable_synthesis:
        evidence_json = json.dumps(
            {
                "regions": [r.to_dict() for r in result.regions],
                "tables": [t.to_dict() for t in result.tables],
                "charts": [c.to_dict() for c in result.charts],
                "formulas": [f.to_dict() for f in result.formulas],
            },
            ensure_ascii=False,
        )
        synth = runner.run_synthesis(str(image_path), result.source_type, evidence_json)
        result.summary_markdown = (synth.get("summary_markdown") or "").strip()
        try:
            result.overall_confidence = float(synth.get("overall_confidence") or 0.0)
        except (TypeError, ValueError):
            result.overall_confidence = 0.0
        for item in synth.get("unresolved") or []:
            result.unresolved.append(str(item))

    # Stage 8: RAG chunks
    rag_chunks.generate_chunks(result)
    result.stage_log = list(runner.stage_log)
    return result


def _apply_crop_refine(runner, image_path, layout, ocr, width, height) -> None:
    """Stage 4 훅(skeleton): dense region 을 crop 해서 재인식.

    실제 crop 저장/재병합은 office 캘리브레이션 후 구현한다. 지금은 어떤 region 이
    crop 후보인지 식별만 하고, 호출 경로가 살아 있음을 보장한다.
    """
    crop_targets = {"table", "chart", "formula", "legend"}
    for raw_region in layout.get("regions") or []:
        if not isinstance(raw_region, dict):
            continue
        rtype = (raw_region.get("type") or "").strip().lower()
        if rtype not in crop_targets:
            continue
        # TODO(office): bbox 로 원본 crop 저장 후 crop_path 로 run_crop_refine 호출,
        # 결과를 ocr 의 tables/charts/formulas 에 재병합.
        print(f"[INFO] (skeleton) crop refine 후보 region: type={rtype}")


def extract_folder(
    input_image_dir: Path,
    output_dir: Path,
    *,
    collection_id: str,
    enable_crop_refine: bool,
    enable_synthesis: bool,
    offline: bool | None,
) -> int:
    """문서 1개의 page 이미지 폴더 전체를 추출하고 chunk JSONL 을 누적 저장한다."""
    if not input_image_dir.exists() or not input_image_dir.is_dir():
        raise FileNotFoundError(f"입력 이미지 폴더가 없습니다: {input_image_dir}")

    images = sorted(
        p for p in input_image_dir.glob("*")
        if p.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}
    )
    if not images:
        print(f"[WARNING] 추출할 이미지가 없습니다: {input_image_dir}")
        return 0

    document_id = input_image_dir.name
    source_type_hint = _infer_source_type(input_image_dir)
    runner = StageRunner(offline=offline)

    chunks_path = output_dir / "rag_chunks.jsonl"
    if chunks_path.exists():
        chunks_path.unlink()  # 재실행 시 중복 누적 방지

    print(
        f"[INFO] 문서 추출 시작: doc={document_id}, 이미지={len(images)}장, "
        f"source_hint={source_type_hint}, offline={runner.offline}"
    )

    total_chunks = 0
    for idx, image_path in enumerate(images, start=1):
        result = extract_one(
            image_path,
            screenshot_index=idx,
            document_id=document_id,
            collection_id=collection_id,
            source_type_hint=source_type_hint,
            runner=runner,
            enable_crop_refine=enable_crop_refine,
            enable_synthesis=enable_synthesis,
        )
        # Stage 7: raw evidence(=review packet 핵심) 저장
        raw_path = output_dir / "raw_evidence" / f"{result.screenshot_id}.json"
        rag_chunks.write_raw_evidence(result, raw_path)
        # Stage 8: chunk JSONL append
        written = rag_chunks.write_chunks_jsonl(result.rag_chunks, chunks_path)
        total_chunks += written
        print(
            f"[INFO]   page {idx:03d}: regions={len(result.regions)}, "
            f"tables={len(result.tables)}, charts={len(result.charts)}, "
            f"chunks={written}"
        )

    print(
        f"[INFO] 완료 -> chunk {total_chunks}개 -> {chunks_path}, "
        f"raw evidence -> {output_dir / 'raw_evidence'}"
    )
    return total_chunks


def main() -> int:
    # Path("") 는 str 로 "." 이 되므로 빈 값 판정은 원본 문자열로 한다.
    if str(INPUT_IMAGE_DIR) in {"", "."} or str(OUTPUT_DIR) in {"", "."}:
        print(
            "[ERROR] INPUT_IMAGE_DIR / OUTPUT_DIR 가 비어 있습니다. "
            "extract_screenshot.py 상단 상수를 수정하세요."
        )
        return 1

    input_dir = INPUT_IMAGE_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    print(f"[INFO] INPUT_IMAGE_DIR = {input_dir}")
    print(f"[INFO] OUTPUT_DIR      = {output_dir}")

    try:
        extract_folder(
            input_dir,
            output_dir,
            collection_id=COLLECTION_ID,
            enable_crop_refine=ENABLE_CROP_REFINE,
            enable_synthesis=ENABLE_SYNTHESIS,
            offline=OFFLINE,
        )
    except Exception as exc:
        print(f"[ERROR] 추출 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
