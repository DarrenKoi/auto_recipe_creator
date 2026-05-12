"""재개 가능한 파이프라인 상태 원장(ledger).

단일 JSON 파일(`logs/pipeline_state.json`)에 모든 doc 의 단계별 진행 상황을 보관한다.
각 runner 는 시작 시 원장을 로드하고, 페이지 또는 단계가 끝날 때마다 atomic 으로 다시 쓴다.
동시 실행은 지원하지 않는다(한 번에 한 runner 만 돌린다는 가정).
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from pipeline.settings import STATE_LEDGER_PATH


LEDGER_VERSION: int = 1
VALID_STAGES: tuple[str, ...] = ("capture", "extract", "organize")
VALID_STATUSES: tuple[str, ...] = ("pending", "in_progress", "done", "failed")


def _now_iso() -> str:
    """타임존 없는 ISO-8601 시각 문자열."""
    return datetime.now().replace(microsecond=0).isoformat()


def _empty_stage(stage: str) -> dict:
    """단계 하나의 기본 상태."""
    payload: dict = {"status": "pending", "error": None, "finished_at": None}
    if stage in {"capture", "extract"}:
        payload["completed_pages"] = []
    if stage == "capture":
        payload["page_count"] = None
    return payload


def _empty_doc(source_path: str, source_type: str) -> dict:
    """새 doc 항목의 기본 모양."""
    return {
        "source_path": source_path,
        "source_type": source_type,
        "discovered_at": _now_iso(),
        "capture": _empty_stage("capture"),
        "extract": _empty_stage("extract"),
        "organize": _empty_stage("organize"),
    }


def load() -> dict:
    """원장을 읽어 dict 로 반환한다. 파일이 없으면 빈 원장을 만든다."""
    if not STATE_LEDGER_PATH.exists():
        return {"version": LEDGER_VERSION, "updated_at": _now_iso(), "documents": {}}

    try:
        text = STATE_LEDGER_PATH.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] 원장 로드 실패, 빈 원장으로 대체: {exc}")
        return {"version": LEDGER_VERSION, "updated_at": _now_iso(), "documents": {}}

    if not isinstance(payload, dict) or "documents" not in payload:
        print("[WARNING] 원장 형식이 비정상이라 새로 초기화한다.")
        return {"version": LEDGER_VERSION, "updated_at": _now_iso(), "documents": {}}

    payload.setdefault("version", LEDGER_VERSION)
    payload.setdefault("documents", {})
    return payload


def save(state: dict) -> None:
    """원장을 atomic 하게 디스크에 쓴다.

    같은 디스크 안에서 tmp 파일로 쓴 뒤 `os.replace` 로 rename 한다.
    POSIX/NT 모두 atomic rename 을 보장한다.
    """
    state["updated_at"] = _now_iso()
    STATE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".pipeline_state.",
        suffix=".tmp.json",
        dir=str(STATE_LEDGER_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, STATE_LEDGER_PATH)
    except Exception:
        # 실패 시 tmp 정리
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def upsert_document(state: dict, doc_id: str, source_path: str, source_type: str) -> dict:
    """원장에 doc 항목이 없으면 추가한다. 기존 항목은 source_path/type 만 최신화."""
    documents = state.setdefault("documents", {})
    entry = documents.get(doc_id)
    if entry is None:
        entry = _empty_doc(source_path, source_type)
        documents[doc_id] = entry
        return entry

    entry["source_path"] = source_path
    entry["source_type"] = source_type
    for stage in VALID_STAGES:
        entry.setdefault(stage, _empty_stage(stage))
    return entry


def mark_stage(
    state: dict,
    doc_id: str,
    stage: str,
    status: str,
    *,
    error: str | None = None,
    page_count: int | None = None,
) -> None:
    """원장에서 doc 의 단계 상태를 갱신한다."""
    if stage not in VALID_STAGES:
        raise ValueError(f"알 수 없는 stage: {stage}")
    if status not in VALID_STATUSES:
        raise ValueError(f"알 수 없는 status: {status}")

    doc = state["documents"].get(doc_id)
    if doc is None:
        raise KeyError(f"원장에 doc_id 가 없다: {doc_id}")

    stage_payload = doc.setdefault(stage, _empty_stage(stage))
    stage_payload["status"] = status
    stage_payload["error"] = error
    if status == "done":
        stage_payload["finished_at"] = _now_iso()
    if stage == "capture" and page_count is not None:
        stage_payload["page_count"] = int(page_count)


def mark_page_done(state: dict, doc_id: str, stage: str, page_number: int) -> None:
    """capture/extract 단계의 페이지 단위 완료 표시."""
    if stage not in {"capture", "extract"}:
        raise ValueError(f"페이지 단위 stage 는 capture/extract 만 가능: {stage}")

    doc = state["documents"].get(doc_id)
    if doc is None:
        raise KeyError(f"원장에 doc_id 가 없다: {doc_id}")

    stage_payload = doc.setdefault(stage, _empty_stage(stage))
    completed = stage_payload.setdefault("completed_pages", [])
    if page_number not in completed:
        completed.append(int(page_number))
        completed.sort()


def select_pending_documents(state: dict, stage: str, retry_failed: bool) -> list[tuple[str, dict]]:
    """주어진 단계에서 처리해야 할 doc 목록을 반환한다.

    Returns: [(doc_id, doc_payload), ...]
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"알 수 없는 stage: {stage}")

    pending: list[tuple[str, dict]] = []
    for doc_id, doc in state.get("documents", {}).items():
        stage_payload = doc.get(stage, _empty_stage(stage))
        status = stage_payload.get("status", "pending")
        if status == "done":
            continue
        if status == "failed" and not retry_failed:
            print(
                f"[WARNING] doc_id={doc_id} stage={stage} failed 상태이므로 건너뜀 "
                f"(RETRY_FAILED 환경변수 또는 settings 로 켤 수 있음)"
            )
            continue
        # capture 가 끝나기 전에 extract 를 시도하지 않는다.
        if stage == "extract" and doc.get("capture", {}).get("status") != "done":
            continue
        if stage == "organize" and doc.get("extract", {}).get("status") != "done":
            continue
        pending.append((doc_id, doc))

    return pending


def smoke_test() -> None:
    """간단한 원장 동작 확인용 self-test."""
    state = load()
    sample_id = "test_smoke_doc"
    upsert_document(state, sample_id, "data/inputs/sample.pdf", "pdf")
    mark_page_done(state, sample_id, "capture", 1)
    mark_stage(state, sample_id, "capture", "in_progress")
    save(state)

    reloaded = load()
    assert sample_id in reloaded["documents"], "doc 항목이 저장되지 않았다"
    assert 1 in reloaded["documents"][sample_id]["capture"]["completed_pages"], (
        "completed_pages 가 저장되지 않았다"
    )

    # smoke test 항목 정리
    reloaded["documents"].pop(sample_id, None)
    save(reloaded)
    print("[INFO] state_ledger smoke test 성공")
