"""워크플로 그래프 정의 — 노드/간선 구조와 validate()."""

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(Enum):
    """노드의 역할 분류. TERMINAL 은 성공/중단 종착점(succeeded, aborted 등)."""

    NORMAL = "normal"
    FALLBACK = "fallback"
    TERMINAL = "terminal"


@dataclass
class WorkflowNode:
    """워크플로 단일 노드 정의.

    failure_routes 는 failure_class -> target node_id 라우팅 테이블이다
    (실패 유형별로 가야 할 fallback 노드를 지정한다). default_next 는
    성공 시 기본 이동 대상이며, non-terminal 노드는 반드시 가져야 한다.
    """

    node_id: str
    description: str
    kind: NodeKind = NodeKind.NORMAL
    max_retries: int = 3
    retry_cooldown_sec: float = 1.0
    failure_routes: dict[str, str] = field(default_factory=dict)
    default_next: str | None = None


@dataclass
class WorkflowGraph:
    """노드 집합 + 진입점으로 이루어진 워크플로 그래프."""

    name: str
    entry_node_id: str
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)

    def add_node(self, node: WorkflowNode) -> "WorkflowGraph":
        """노드를 추가하고 self 를 반환한다 (체이닝 편의)."""
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id!r}")
        self.nodes[node.node_id] = node
        return self

    def validate(self) -> list[str]:
        """그래프 일관성을 검사해 문제 목록을 반환한다 (문제 없으면 []).

        확인 항목: 노드 존재 / entry 존재 / 미지의 default_next 와 failure
        route target / non-terminal 의 default_next 누락 / entry 에서 도달
        가능한 terminal 존재.
        """
        problems: list[str] = []
        if not self.nodes:
            problems.append(f"graph {self.name!r} has no nodes")
        if self.entry_node_id not in self.nodes:
            problems.append(f"entry node {self.entry_node_id!r} is not in nodes")

        for node in self.nodes.values():
            for target in node.failure_routes.values():
                if target not in self.nodes:
                    problems.append(
                        f"node {node.node_id!r} failure route target {target!r} is unknown"
                    )
            if node.kind is not NodeKind.TERMINAL and node.default_next is None:
                problems.append(f"non-terminal node {node.node_id!r} has no default_next")
            elif node.default_next is not None and node.default_next not in self.nodes:
                problems.append(
                    f"node {node.node_id!r} default_next {node.default_next!r} is unknown"
                )

        # entry 에서 success 간선 + failure route 간선으로 도달 가능한 집합.
        reachable: set[str] = set()
        stack = [self.entry_node_id]
        while stack:
            node_id = stack.pop()
            if node_id in reachable:
                continue
            node = self.nodes.get(node_id)
            if node is None:
                continue  # 미지 target 은 reachable 에 넣지 않는다 (별도 문제로 보고됨)
            reachable.add(node_id)
            for target in [node.default_next, *node.failure_routes.values()]:
                if target and target not in reachable and target in self.nodes:
                    stack.append(target)

        if not any(self.nodes[nid].kind is NodeKind.TERMINAL for nid in reachable):
            problems.append(
                f"no terminal node reachable from entry {self.entry_node_id!r}"
            )
        return problems