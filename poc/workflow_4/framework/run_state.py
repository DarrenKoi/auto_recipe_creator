"""런 상태 모델 — 실행 추적 기록 + 명시적 JSON 직렬화."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


def now_iso() -> str:
    """로컬 시각 ISO8601 문자열 (seconds 정밀도)."""
    return datetime.now().isoformat(timespec="seconds")


class RunStatus(Enum):
    """워크플로 전체 실행 상태."""

    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ESCALATED = "escalated"


@dataclass
class TransitionRecord:
    """단일 전이 기록 (history 의 한 줄)."""

    seq: int
    ts: str
    from_node: str
    to_node: str
    event: str  # "success" | "retry" | "fallback" | "escalate" | "abort"
    failure_class: str | None = None
    attempt: int = 1
    note: str | None = None

    def to_json_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "event": self.event,
            "failure_class": self.failure_class,
            "attempt": self.attempt,
            "note": self.note,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "TransitionRecord":
        return cls(
            seq=int(d["seq"]),
            ts=str(d["ts"]),
            from_node=str(d["from_node"]),
            to_node=str(d["to_node"]),
            event=str(d["event"]),
            failure_class=d.get("failure_class"),
            attempt=int(d.get("attempt", 1)),
            note=d.get("note"),
        )


@dataclass
class RunState:
    """워크플로 실행 전체 상태 (persist 대상)."""

    run_id: str
    graph_name: str
    current_node: str
    status: RunStatus = RunStatus.RUNNING
    attempt: int = 1
    node_retries: dict[str, int] = field(default_factory=dict)  # 노드별 실패 횟수
    fallback_visits: dict[str, int] = field(default_factory=dict)  # fallback 노드로 라우팅된 횟수
    history: list[TransitionRecord] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    failure_class: str | None = None
    note: str | None = None

    def to_json_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "graph_name": self.graph_name,
            "status": self.status.value,
            "current_node": self.current_node,
            "attempt": self.attempt,
            "node_retries": dict(self.node_retries),
            "fallback_visits": dict(self.fallback_visits),
            "history": [r.to_json_dict() for r in self.history],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure_class": self.failure_class,
            "note": self.note,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "RunState":
        return cls(
            run_id=str(d["run_id"]),
            graph_name=str(d["graph_name"]),
            current_node=str(d["current_node"]),
            status=RunStatus(str(d["status"])),
            attempt=int(d.get("attempt", 1)),
            node_retries={
                str(k): int(v) for k, v in (d.get("node_retries") or {}).items()
            },
            fallback_visits={
                str(k): int(v) for k, v in (d.get("fallback_visits") or {}).items()
            },
            history=[
                TransitionRecord.from_json_dict(r) for r in (d.get("history") or [])
            ],
            started_at=str(d.get("started_at") or now_iso()),
            finished_at=d.get("finished_at"),
            failure_class=d.get("failure_class"),
            note=d.get("note"),
        )