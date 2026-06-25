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

from side_projects.document_extraction.extraction import crop, merge, rag_chunks
from side_projects.document_extraction.extraction.models import StageRunner
from side_projects.document_extraction.extraction.schemas import ExtractionResult
from side_projects.document_extraction.extraction.synthesis import synthesize_deterministic


# === 실행 전 매번 채워 넣을 것 =================================================
# 입력: 캡처 단계가 만든 page WebP 들이 들어 있는 폴더(문서 1개의 출력 폴더).
INPUT_IMAGE_DIR: Path = Path("")   # 예: Path(r"C:\...\extracted\presentation_A")
OUTPUT_DIR: Path = Path("")        # 예: Path(r"C:\...\extracted\presentation_A\_rag")
COLLECTION_ID: str = "default_collection"
ENABLE_CROP_REFINE: bool = False   # skeleton: 기본 비활성(office 캘리브레이션 후 활성)
# 합성 모드: "deterministic"(모델 0콜, 기본) | "kimi"(kimi-k2.6 호출) | "none"(생략)
SYNTHESIS_MODE: str = "deterministic"
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
    synthesis_mode: str = "deterministic",
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
        crop_dir = image_path.parent / "_crops" / screenshot_id
        _apply_crop_refine(runner, image_path, layout, ocr, (width, height), crop_dir)

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

    # Stage 6: synthesis
    #   deterministic = 모델 0콜로 evidence 조립(기본)
    #   kimi          = kimi-k2.6 호출(고품질 합성/충돌 해소)
    #   none          = 합성 생략
    if synthesis_mode == "kimi":
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
        if result.summary_markdown:
            result.summary_model_sources = ["kimi-k2.6"]
    elif synthesis_mode == "deterministic":
        synth = synthesize_deterministic(result)
        result.summary_markdown = synth["summary_markdown"]
        result.overall_confidence = synth["overall_confidence"]
        result.unresolved = synth["unresolved"]
        if result.summary_markdown:
            result.summary_model_sources = ["deterministic"]
    # synthesis_mode == "none": 합성 생략(summary_markdown 빈 채로 둠)

    # Stage 8: RAG chunks
    rag_chunks.generate_chunks(result)
    result.stage_log = list(runner.stage_log)
    return result


def _apply_crop_refine(runner, image_path, layout, ocr, frame_wh, crop_dir) -> None:
    """Stage 4: dense region(table/chart/formula/legend)을 잘라 재인식하고 ocr 에 병합.

    - bbox 로 원본 crop 을 저장(crop.crop_region; 순수 CV).
    - crop 을 run_crop_refine 로 재인식(offline 이면 stub).
    - 재인식 결과를 ocr 의 tables/charts/formulas/raw_text 에 보강 병합.
    표/차트는 기존 항목이 비어 있을 때만 채워, 좋은 first-pass 결과를 덮지 않는다.
    """
    crop_targets = {"table", "chart", "formula", "legend"}
    width, height = frame_wh
    for idx, raw_region in enumerate(layout.get("regions") or []):
        if not isinstance(raw_region, dict):
            continue
        rtype = (raw_region.get("type") or "").strip().lower()
        if rtype not in crop_targets:
            continue
        bbox = raw_region.get("bbox") or {}
        box_tuple = (
            bbox.get("left", 0), bbox.get("top", 0),
            bbox.get("right", 0), bbox.get("bottom", 0),
        )
        meta = crop.crop_region(
            image_path, f"r{idx + 1:03d}", rtype, box_tuple, crop_dir
        )
        if meta is None:
            print(f"[INFO] crop refine 스킵(유효하지 않은 bbox): type={rtype}")
            continue

        refined = runner.run_crop_refine(
            meta.crop_path, meta.crop_wh[0], meta.crop_wh[1], rtype
        )
        _merge_crop_refine(rtype, refined, ocr)


def _merge_crop_refine(region_type: str, refined: dict, ocr: dict) -> None:
    """crop 재인식 결과를 ocr evidence 에 병합.

    원칙: 데이터 손실 금지(별개 표/차트는 보존) + 정확 중복만 제거. 각 crop 영역은
    distinct 한 표/차트이므로, 같은 header(표)/legend(차트)가 이미 있을 때만 skip,
    아니면 새 항목으로 추가한다. (좌표 기반 first-pass<->crop 매칭/정교한 dedup 은
    office 캘리브레이션 TODO; 지금은 손실보다 보존을 택한다.)
    """
    text = (refined.get("text") or "").strip()
    if text:
        ocr["raw_text"] = ((ocr.get("raw_text") or "") + "\n" + text).strip()

    if region_type == "table":
        header = [str(h) for h in (refined.get("header") or [])]
        rows = [[str(c) for c in row] for row in (refined.get("rows") or [])]
        if not (header or rows):
            return
        tables = ocr.setdefault("tables", [])
        existing_headers = {tuple(t.get("header") or []) for t in tables}
        if tuple(header) not in existing_headers:
            tables.append({"title": "", "header": header, "rows": rows})
    elif region_type in {"chart", "legend"}:
        labels = [str(l) for l in (refined.get("labels") or [])]
        if not labels:
            return
        charts = ocr.setdefault("charts", [])
        if region_type == "legend" and charts:
            # legend 영역: 가장 최근 차트의 legend 를 보강(어느 차트 소속인지 좌표 없이
            # 단정 불가하므로 last chart 에 attach)
            existing = charts[-1].setdefault("legend_labels", [])
            for lab in labels:
                if lab not in existing:
                    existing.append(lab)
        else:
            existing_label_sets = [set(c.get("legend_labels") or []) for c in charts]
            if set(labels) not in existing_label_sets:
                charts.append({"title": "", "legend_labels": labels})


def extract_folder(
    input_image_dir: Path,
    output_dir: Path,
    *,
    collection_id: str,
    enable_crop_refine: bool,
    synthesis_mode: str,
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
            synthesis_mode=synthesis_mode,
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
            synthesis_mode=SYNTHESIS_MODE,
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
