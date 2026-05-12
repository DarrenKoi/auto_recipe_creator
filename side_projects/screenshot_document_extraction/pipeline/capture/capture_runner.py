"""Step 1 runner — `data/inputs/` 를 순회하며 페이지별 JPEG 를 만든다.

각 doc 의 capture 단계 상태는 `logs/pipeline_state.json` 원장에 기록되며,
페이지 하나가 끝날 때마다 ledger 가 즉시 갱신되므로 도중에 중단해도 재실행 시
완료된 페이지는 건너뛴다.
"""

from pathlib import Path
from typing import Callable, Iterator

from poc.work2.logger import log_work2_event

from pipeline import state_ledger
from pipeline.ids import compute_doc_id, infer_source_type
from pipeline.settings import (
    EXTENSION_TO_SOURCE_TYPE,
    INPUTS_DIR,
    PIPELINE_LOG_NAME,
    RETRY_FAILED,
    SAFE_MODE,
    ensure_directories,
)
from pipeline.capture.common import (
    PageArtifact,
    doc_capture_dir,
    now_iso,
    page_jpeg_path,
    page_meta_path,
    save_page_jpeg,
    save_page_meta,
)


PageIter = Callable[[Path], Iterator[tuple[int, bytes, int, int]]]


def _dispatch_handler(source_type: str) -> PageIter:
    """source_type 에 맞는 핸들러의 iter_pages 를 반환한다.

    Office 핸들러는 import 시점에 pywin32 를 요구하므로 lazy import 한다.
    """
    if source_type == "pdf":
        from pipeline.capture import pdf_handler

        return pdf_handler.iter_pages
    if source_type == "powerpoint":
        from pipeline.capture import powerpoint_handler

        return powerpoint_handler.iter_pages
    if source_type == "word":
        from pipeline.capture import word_handler

        return word_handler.iter_pages
    if source_type == "excel":
        from pipeline.capture import excel_handler

        return excel_handler.iter_pages
    raise ValueError(f"지원하지 않는 source_type: {source_type}")


def _discover_inputs(state: dict) -> int:
    """`inputs/` 에 있는 새 파일들을 원장에 추가하고 추가된 수를 반환한다."""
    added = 0
    if not INPUTS_DIR.exists():
        return 0

    for path in sorted(INPUTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSION_TO_SOURCE_TYPE:
            continue
        doc_id = compute_doc_id(path)
        source_type = infer_source_type(path)
        # 상대 경로로 저장하면 사용자 PC 간 이동이 쉽다.
        try:
            relative = path.resolve().relative_to(INPUTS_DIR.parent.parent.resolve())
            stored_path = str(relative).replace("\\", "/")
        except ValueError:
            stored_path = str(path.resolve())

        existing = state.get("documents", {}).get(doc_id)
        is_new = existing is None
        state_ledger.upsert_document(state, doc_id, stored_path, source_type)
        if is_new:
            added += 1
            print(f"[INFO] 새 입력 발견: doc_id={doc_id} source={stored_path}")
            log_work2_event(
                component="capture_runner",
                message="input_discovered",
                log_name=PIPELINE_LOG_NAME,
                doc_id=doc_id,
                source_path=stored_path,
                source_type=source_type,
            )
    return added


def _capture_one(doc_id: str, doc: dict, state: dict) -> None:
    """한 doc 의 capture 단계를 끝까지 (또는 실패할 때까지) 진행한다."""
    source_type = doc["source_type"]
    source_path = Path(doc["source_path"])
    if not source_path.is_absolute():
        # ledger 에는 repo 루트 상대 경로로 저장돼 있다.
        source_path = (INPUTS_DIR.parent.parent / source_path).resolve()

    if not source_path.exists():
        message = f"원본 파일이 없다: {source_path}"
        print(f"[ERROR] {message}")
        state_ledger.mark_stage(state, doc_id, "capture", "failed", error=message)
        state_ledger.save(state)
        return

    try:
        iter_pages = _dispatch_handler(source_type)
    except ValueError as exc:
        state_ledger.mark_stage(state, doc_id, "capture", "failed", error=str(exc))
        state_ledger.save(state)
        return

    capture_payload = doc["capture"]
    completed_pages: set[int] = set(capture_payload.get("completed_pages", []))
    out_dir = doc_capture_dir(doc_id)

    state_ledger.mark_stage(state, doc_id, "capture", "in_progress")
    state_ledger.save(state)

    print(
        f"[INFO] capture 시작: doc_id={doc_id} source_type={source_type} "
        f"path={source_path}"
    )
    log_work2_event(
        component="capture_runner",
        message="capture_started",
        log_name=PIPELINE_LOG_NAME,
        doc_id=doc_id,
        source_path=str(source_path),
        source_type=source_type,
    )

    last_page_seen = 0
    try:
        for page_index, jpeg_bytes, width, height in iter_pages(source_path):
            last_page_seen = page_index
            if page_index in completed_pages:
                print(f"  [SKIP] doc_id={doc_id} page={page_index} (이미 완료)")
                continue

            jpeg_path = page_jpeg_path(doc_id, page_index)
            meta_path = page_meta_path(doc_id, page_index)
            save_page_jpeg(jpeg_bytes, jpeg_path)
            artifact = PageArtifact(
                page_index=page_index,
                width=width,
                height=height,
                source_path=str(source_path),
                source_type=source_type,
                capture_method="native_export",
                captured_at=now_iso(),
            )
            save_page_meta(artifact, meta_path)

            state_ledger.mark_page_done(state, doc_id, "capture", page_index)
            state_ledger.save(state)
            print(
                f"  [OK]   doc_id={doc_id} page={page_index} "
                f"size={width}x{height} -> {jpeg_path.name}"
            )

    except ImportError as exc:
        # 플랫폼 미지원 (예: macOS 에서 PowerPoint COM)
        message = str(exc)
        print(f"[WARNING] {message}")
        state_ledger.mark_stage(state, doc_id, "capture", "failed", error=message)
        state_ledger.save(state)
        log_work2_event(
            component="capture_runner",
            message="capture_unsupported",
            level="warning",
            log_name=PIPELINE_LOG_NAME,
            doc_id=doc_id,
            error=message,
        )
        return
    except Exception as exc:
        message = f"capture 실패: {exc}"
        print(f"[ERROR] doc_id={doc_id} {message}")
        state_ledger.mark_stage(state, doc_id, "capture", "failed", error=str(exc))
        state_ledger.save(state)
        log_work2_event(
            component="capture_runner",
            message="capture_failed",
            level="error",
            log_name=PIPELINE_LOG_NAME,
            doc_id=doc_id,
            error=str(exc),
        )
        return

    page_count = max(
        last_page_seen,
        max(state["documents"][doc_id]["capture"].get("completed_pages", [0]) or [0]),
    )
    state_ledger.mark_stage(
        state, doc_id, "capture", "done", page_count=page_count
    )
    state_ledger.save(state)
    print(f"[INFO] capture 완료: doc_id={doc_id} pages={page_count} dir={out_dir}")
    log_work2_event(
        component="capture_runner",
        message="capture_finished",
        log_name=PIPELINE_LOG_NAME,
        doc_id=doc_id,
        page_count=page_count,
    )


def run() -> None:
    """`data/inputs/` 안의 모든 지원 파일에 대해 capture 단계를 실행한다."""
    ensure_directories()

    if SAFE_MODE:
        print("[INFO] SAFE_MODE 가 활성화돼 있어 capture 를 실제 실행하지 않는다.")
        return

    state = state_ledger.load()
    added = _discover_inputs(state)
    state_ledger.save(state)
    print(f"[INFO] inputs 검색 완료: 새 항목 {added}개")

    pending = state_ledger.select_pending_documents(
        state, stage="capture", retry_failed=RETRY_FAILED
    )
    if not pending:
        print("[INFO] capture 단계에서 처리할 항목이 없다.")
        return

    print(f"[INFO] capture 대상 doc 수: {len(pending)}")
    for doc_id, doc in pending:
        _capture_one(doc_id, doc, state)
    print("[INFO] capture 단계 종료")
