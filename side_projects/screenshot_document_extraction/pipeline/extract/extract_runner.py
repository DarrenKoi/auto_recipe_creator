"""Step 2 runner — capture 된 페이지 JPEG 를 paddleocr-vl + ui-venus 로 추출한다."""

from pathlib import Path

from poc.work2.logger import log_work2_event

from pipeline import state_ledger
from pipeline.capture.common import doc_capture_dir, page_jpeg_path
from pipeline.extract import extract_one
from pipeline.settings import (
    CAPTURES_DIR,
    EXTRACTED_DIR,
    PIPELINE_LOG_NAME,
    RETRY_FAILED,
    SAFE_MODE,
    ensure_directories,
)


def _doc_extract_dir(doc_id: str) -> Path:
    """doc 의 extract 출력 디렉터리. 없으면 만든다."""
    out = EXTRACTED_DIR / doc_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _captured_pages(doc_id: str) -> list[int]:
    """디스크에 실제로 존재하는 capture 페이지 번호 목록."""
    capture_dir = doc_capture_dir(doc_id)
    pages: list[int] = []
    for path in sorted(capture_dir.glob("page_*.jpg")):
        try:
            stem = path.stem  # page_NNN
            number = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pages.append(number)
    return pages


def _extract_one_doc(doc_id: str, doc: dict, state: dict) -> None:
    """doc 하나의 모든 페이지를 순차 추출한다."""
    out_dir = _doc_extract_dir(doc_id)
    existing_pages = _captured_pages(doc_id)
    if not existing_pages:
        message = f"capture 폴더가 비어 있다: {CAPTURES_DIR / doc_id}"
        print(f"[ERROR] doc_id={doc_id} {message}")
        state_ledger.mark_stage(state, doc_id, "extract", "failed", error=message)
        state_ledger.save(state)
        return

    completed_pages: set[int] = set(doc["extract"].get("completed_pages", []))
    state_ledger.mark_stage(state, doc_id, "extract", "in_progress")
    state_ledger.save(state)

    print(
        f"[INFO] extract 시작: doc_id={doc_id} pages={len(existing_pages)} "
        f"이미완료={len(completed_pages)}"
    )
    log_work2_event(
        component="extract_runner",
        message="extract_started",
        log_name=PIPELINE_LOG_NAME,
        doc_id=doc_id,
        total_pages=len(existing_pages),
    )

    for page_index in existing_pages:
        if page_index in completed_pages:
            print(f"  [SKIP] doc_id={doc_id} page={page_index} (이미 추출됨)")
            continue
        jpeg_path = page_jpeg_path(doc_id, page_index)
        try:
            result = extract_one.process(
                jpeg_path,
                doc_id=doc_id,
                page_index=page_index,
                out_dir=out_dir,
            )
        except Exception as exc:
            message = f"page={page_index} 추출 실패: {exc}"
            print(f"[ERROR] doc_id={doc_id} {message}")
            state_ledger.mark_stage(state, doc_id, "extract", "failed", error=str(exc))
            state_ledger.save(state)
            log_work2_event(
                component="extract_runner",
                message="extract_failed",
                level="error",
                log_name=PIPELINE_LOG_NAME,
                doc_id=doc_id,
                page_index=page_index,
                error=str(exc),
            )
            return

        state_ledger.mark_page_done(state, doc_id, "extract", page_index)
        state_ledger.save(state)
        print(
            f"  [OK]   doc_id={doc_id} page={page_index} -> {result.raw_path.name}"
        )

    state_ledger.mark_stage(state, doc_id, "extract", "done")
    state_ledger.save(state)
    print(f"[INFO] extract 완료: doc_id={doc_id}")
    log_work2_event(
        component="extract_runner",
        message="extract_finished",
        log_name=PIPELINE_LOG_NAME,
        doc_id=doc_id,
    )


def run() -> None:
    """capture 가 끝난 모든 doc 에 대해 extract 단계를 실행한다."""
    ensure_directories()

    if SAFE_MODE:
        print("[INFO] SAFE_MODE 가 활성화돼 있어 VLM 호출을 실제 실행하지 않는다.")
        return

    state = state_ledger.load()
    pending = state_ledger.select_pending_documents(
        state, stage="extract", retry_failed=RETRY_FAILED
    )
    if not pending:
        print("[INFO] extract 단계에서 처리할 항목이 없다.")
        return

    print(f"[INFO] extract 대상 doc 수: {len(pending)}")
    for doc_id, doc in pending:
        _extract_one_doc(doc_id, doc, state)
    print("[INFO] extract 단계 종료")
