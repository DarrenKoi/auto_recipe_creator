"""Step 3 runner — 페이지별 raw JSON 을 머지해 Markdown + JSON 으로 정리한다."""

import json
from datetime import datetime
from pathlib import Path

from poc.work2.logger import log_work2_event

from pipeline import state_ledger
from pipeline.settings import (
    EXTRACTED_DIR,
    ORGANIZED_DIR,
    PIPELINE_LOG_NAME,
    RETRY_FAILED,
    ensure_directories,
)


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _doc_organized_dir(doc_id: str) -> Path:
    """doc 의 organize 출력 디렉터리. 없으면 만든다."""
    out = ORGANIZED_DIR / doc_id
    (out / "pages").mkdir(parents=True, exist_ok=True)
    return out


def _load_raw_pages(doc_id: str) -> list[dict]:
    """extracted/<doc_id>/page_*.raw.json 을 페이지 순서대로 로드한다."""
    src_dir = EXTRACTED_DIR / doc_id
    if not src_dir.exists():
        return []
    items: list[tuple[int, dict]] = []
    for path in sorted(src_dir.glob("page_*.raw.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARNING] raw.json 로드 실패 {path.name}: {exc}")
            continue
        page_index = int(payload.get("page_index", 0))
        items.append((page_index, payload))
    items.sort(key=lambda item: item[0])
    return [payload for _, payload in items]


def _region_summary_lines(regions: list[dict]) -> list[str]:
    """ui-venus region 목록을 사람이 읽을 수 있는 bullet 줄로 변환한다."""
    lines: list[str] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_type = str(region.get("type", "other")).strip() or "other"
        text = str(region.get("text", "")).strip().replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        bbox = region.get("bbox") or {}
        if isinstance(bbox, dict) and all(k in bbox for k in ("left", "top", "right", "bottom")):
            bbox_label = (
                f"({int(bbox['left'])},{int(bbox['top'])})-"
                f"({int(bbox['right'])},{int(bbox['bottom'])})"
            )
        else:
            bbox_label = ""
        confidence = region.get("confidence")
        confidence_label = ""
        if isinstance(confidence, (int, float)):
            confidence_label = f" conf={float(confidence):.2f}"
        suffix = f" bbox={bbox_label}{confidence_label}".rstrip()
        if not text:
            lines.append(f"- {region_type}{suffix}")
        else:
            lines.append(f"- **{region_type}**: {text}{suffix}")
    return lines


def _render_markdown(doc_id: str, doc_meta: dict, pages: list[dict]) -> str:
    """페이지 머지본을 Markdown 으로 렌더링한다."""
    source_path = doc_meta.get("source_path", "(unknown)")
    source_type = doc_meta.get("source_type", "unknown")
    page_count = len(pages)
    organized_at = _now_iso()
    header = [
        f"# {Path(source_path).name or doc_id}",
        "",
        f"- doc_id: `{doc_id}`",
        f"- source_path: `{source_path}`",
        f"- source_type: `{source_type}`",
        f"- page_count: {page_count}",
        f"- organized_at: {organized_at}",
        "",
    ]

    body: list[str] = []
    for payload in pages:
        page_index = payload.get("page_index", "?")
        body.append(f"## Page {page_index}")
        body.append("")

        ocr_text = ""
        paddle = payload.get("paddleocr") or {}
        if isinstance(paddle, dict):
            ocr_text = (paddle.get("text") or "").strip()
        if ocr_text:
            body.append("### OCR 텍스트 (paddleocr-vl-1.5)")
            body.append("")
            body.append("```")
            body.append(ocr_text)
            body.append("```")
            body.append("")
        else:
            body.append("_OCR 텍스트 없음_")
            body.append("")

        uivenus = payload.get("uivenus") or {}
        parsed = uivenus.get("parsed") if isinstance(uivenus, dict) else {}
        regions = parsed.get("regions") if isinstance(parsed, dict) else None
        if isinstance(regions, list) and regions:
            body.append("### 영역 (ui-venus)")
            body.append("")
            body.extend(_region_summary_lines(regions))
            body.append("")
        elif isinstance(parsed, dict) and parsed.get("parse_error"):
            body.append("_ui-venus 응답 JSON 파싱 실패; raw_text 만 보존됨._")
            body.append("")

        source_image = payload.get("source_image")
        if source_image:
            body.append(f"_source image: `{source_image}`_")
            body.append("")

    return "\n".join(header + body).rstrip() + "\n"


def _build_document_json(doc_id: str, doc_meta: dict, pages: list[dict]) -> dict:
    """document.json payload 를 만든다."""
    page_records: list[dict] = []
    for payload in pages:
        paddle = payload.get("paddleocr") or {}
        uivenus = payload.get("uivenus") or {}
        parsed = uivenus.get("parsed") if isinstance(uivenus, dict) else {}
        page_records.append(
            {
                "page_index": payload.get("page_index"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "source_image": payload.get("source_image"),
                "ocr_text": (paddle.get("text") or "") if isinstance(paddle, dict) else "",
                "regions": parsed.get("regions") if isinstance(parsed, dict) else None,
                "page_type": parsed.get("page_type") if isinstance(parsed, dict) else None,
                "notes": parsed.get("notes") if isinstance(parsed, dict) else None,
            }
        )
    return {
        "doc_id": doc_id,
        "source_path": doc_meta.get("source_path"),
        "source_type": doc_meta.get("source_type"),
        "page_count": len(page_records),
        "organized_at": _now_iso(),
        "pages": page_records,
    }


def _write_page_sidecars(out_dir: Path, document_payload: dict) -> None:
    """pages/page_NNN.json 을 페이지마다 저장한다."""
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page in document_payload["pages"]:
        index = page.get("page_index")
        if index is None:
            continue
        path = pages_dir / f"page_{int(index):03d}.json"
        path.write_text(
            json.dumps(
                {
                    "doc_id": document_payload["doc_id"],
                    "source_path": document_payload["source_path"],
                    "source_type": document_payload["source_type"],
                    "page": page,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _organize_one(doc_id: str, doc: dict, state: dict) -> None:
    """doc 하나의 organize 단계를 수행한다."""
    pages = _load_raw_pages(doc_id)
    if not pages:
        message = f"extracted 폴더에 raw.json 이 없다: {EXTRACTED_DIR / doc_id}"
        print(f"[ERROR] doc_id={doc_id} {message}")
        state_ledger.mark_stage(state, doc_id, "organize", "failed", error=message)
        state_ledger.save(state)
        return

    state_ledger.mark_stage(state, doc_id, "organize", "in_progress")
    state_ledger.save(state)
    print(f"[INFO] organize 시작: doc_id={doc_id} pages={len(pages)}")

    out_dir = _doc_organized_dir(doc_id)
    try:
        markdown_text = _render_markdown(doc_id, doc, pages)
        (out_dir / "document.md").write_text(markdown_text, encoding="utf-8")

        document_payload = _build_document_json(doc_id, doc, pages)
        (out_dir / "document.json").write_text(
            json.dumps(document_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_page_sidecars(out_dir, document_payload)
    except Exception as exc:
        message = f"organize 실패: {exc}"
        print(f"[ERROR] doc_id={doc_id} {message}")
        state_ledger.mark_stage(state, doc_id, "organize", "failed", error=str(exc))
        state_ledger.save(state)
        log_work2_event(
            component="organize_runner",
            message="organize_failed",
            level="error",
            log_name=PIPELINE_LOG_NAME,
            doc_id=doc_id,
            error=str(exc),
        )
        return

    state_ledger.mark_stage(state, doc_id, "organize", "done")
    state_ledger.save(state)
    print(f"[INFO] organize 완료: doc_id={doc_id} -> {out_dir}")
    log_work2_event(
        component="organize_runner",
        message="organize_finished",
        log_name=PIPELINE_LOG_NAME,
        doc_id=doc_id,
        page_count=len(pages),
    )


def run() -> None:
    """extract 가 끝난 모든 doc 에 대해 organize 단계를 실행한다."""
    ensure_directories()

    state = state_ledger.load()
    pending = state_ledger.select_pending_documents(
        state, stage="organize", retry_failed=RETRY_FAILED
    )
    if not pending:
        print("[INFO] organize 단계에서 처리할 항목이 없다.")
        return

    print(f"[INFO] organize 대상 doc 수: {len(pending)}")
    for doc_id, doc in pending:
        _organize_one(doc_id, doc, state)
    print("[INFO] organize 단계 종료")
