"""워크플로 그래프 시각화 — mermaid / ascii / md snapshot / HTML live view (0 deps).

HTML 뷰는 vendored `assets/mermaid.min.js` 를 인라인 임베드해 단일 파일로 완전히
self-contained 가 된다 — IDE/VS Code 없이 Edge/Chrome 등 아무 브라우저로 열린다.
asset 이 없으면 CDN fallback(오프라인에서는 vendor 파일이 필요하다는 배너 표기).
"""

import html
import os
import platform
import webbrowser
from pathlib import Path

from poc.workflow_4.framework.run_state import RunState
from poc.workflow_4.framework.state_graph import NodeKind, WorkflowGraph

# vendored mermaid asset — 없으면 CDN fallback. 파일 내용은 모듈 레벨에서 한 번만 읽는다.
_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "mermaid.min.js"
_MMA_CACHE: str | None = None


def _mermaid_asset_content() -> str:
    """vendored mermaid.min.js 를 한 번 읽어 캐시한다. 없으면 빈 문자열."""
    global _MMA_CACHE
    if _MMA_CACHE is None:
        try:
            _MMA_CACHE = _ASSET_PATH.read_text(encoding="utf-8")
        except OSError:
            print(
                f"[WARNING] vendored mermaid asset 없음: {_ASSET_PATH} "
                "- HTML 은 CDN fallback 으로 동작합니다(오프라인은 vendor 필요)."
            )
            _MMA_CACHE = ""
    return _MMA_CACHE


def write_text_atomic(path: Path, text: str) -> None:
    """tmp 에 쓰고 os.replace 로 바꿔치기한다 - 브라우저가 1s 마다 다시 읽는 3.6MB
    HTML 을 쓰는 도중에 읽히면 빈 화면/깨진 페이지가 한 틱 보이므로(torn read),
    같은 볼륨 안 rename 으로 '완성본 아니면 직전본' 만 보이게 한다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sanitize_label(text: str) -> str:
    """mermaid 라벨을 깨뜨릴 수 있는 문자를 정리한다."""
    return str(text).replace("\n", " ").replace(":", " ").replace("|", "/")


def _cell(text: str) -> str:
    """markdown 테이블 셀에서 깨지는 문자를 이스케이프한다."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _html_cell(text: str) -> str:
    """HTML 테이블 셀용 이스케이프."""
    return html.escape(str(text), quote=False).replace("\n", " ")


def render_mermaid(graph: WorkflowGraph, run_state: RunState) -> str:
    """stateDiagram-v2 문자열. 현재 노드는 active 클래스로 강조한다."""
    lines = [
        "stateDiagram-v2",
        (
            f"    %% graph={graph.name} status={run_state.status.value} "
            f"current={run_state.current_node}"
        ),
        f"    [*] --> {graph.entry_node_id}",
    ]
    for node in graph.nodes.values():
        lines.append(f"    {node.node_id}: {_sanitize_label(node.description)}")
    for node in graph.nodes.values():
        if node.default_next:
            lines.append(f"    {node.node_id} --> {node.default_next}: success")
        for failure_class, target in node.failure_routes.items():
            lines.append(
                f"    {node.node_id} --> {target}: {_sanitize_label(failure_class)}"
            )
    lines.append("    classDef active fill:#ffd54f,stroke:#f57f17")
    lines.append("    classDef term fill:#e8f5e9,stroke:#2e7d32")
    for node in graph.nodes.values():
        if node.kind is NodeKind.TERMINAL:
            lines.append(f"    class {node.node_id} term")
    lines.append(f"    class {run_state.current_node} active")
    return "\n".join(lines)


def render_ascii(graph: WorkflowGraph, run_state: RunState) -> str:
    """노드당 한 줄 지도. 현재 노드는 * 마커 + attempt 카운터를 단다."""
    lines = [
        f"{graph.name} [status={run_state.status.value}] current={run_state.current_node}"
    ]
    for node in graph.nodes.values():
        marker = "*" if node.node_id == run_state.current_node else " "
        line = f"{marker} {node.node_id}"
        if node.node_id == run_state.current_node:
            line += f" (attempt {run_state.attempt}/{node.max_retries})"
        if node.description:
            line += f" - {node.description}"
        if node.kind is NodeKind.TERMINAL:
            line += " [terminal]"
        lines.append(line)
    return "\n".join(lines)


def render_html(
    graph: WorkflowGraph, run_state: RunState, refresh_sec: int = 1
) -> str:
    """전체 HTML 문서 (self-contained). mermaid 라이브러리를 인라인 임베드한다.

    `refresh_sec` 마다 브라우저가 자동 새로고침해 live view 가 된다.
    """
    asset = _mermaid_asset_content()
    if asset:
        mermaid_script = f"<script>\n{asset}\n</script>"
        banner = ""
    else:
        mermaid_script = (
            '<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js">'
            "</script>"
        )
        banner = (
            '<div class="warn">[WARNING] vendored mermaid asset 이 없어 CDN 을 '
            "사용합니다. 오프라인에서는 <code>framework/assets/mermaid.min.js</code> 를 "
            "vendor 해야 그래프가 그려집니다.</div>"
        )

    diagram = html.escape(render_mermaid(graph, run_state), quote=False)
    rows = [
        "<tr>"
        f"<td>{r.seq}</td>"
        f"<td>{_html_cell(r.ts)}</td>"
        f"<td>{_html_cell(r.from_node)}</td>"
        f"<td>{_html_cell(r.to_node)}</td>"
        f"<td>{_html_cell(r.event)}</td>"
        f"<td>{_html_cell(r.failure_class or '')}</td>"
        f"<td>{r.attempt}</td>"
        "</tr>"
        for r in run_state.history
    ]
    history_rows = (
        "\n".join(rows)
        if rows
        else '<tr><td colspan="7" class="muted">아직 기록된 전이가 없습니다.</td></tr>'
    )

    refresh = max(1, int(refresh_sec))
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>WF4 — {graph.name} ({run_state.status.value})</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 16px; color: #222; }}
  h1 {{ font-size: 18px; }}
  .status {{ display: inline-block; padding: 2px 10px; border-radius: 10px; color: #fff;
             background: #0288d1; font-size: 13px; vertical-align: middle; }}
  .status.completed {{ background: #2e7d32; }}
  .status.aborted {{ background: #c62828; }}
  .status.escalated {{ background: #ef6c00; }}
  .muted {{ color: #888; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffe69c; border-radius: 6px;
           padding: 8px 12px; margin: 8px 0; font-size: 13px; }}
  .mermaid {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px;
             padding: 12px; margin: 12px 0; overflow: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>{graph.name}
  <span class="status {run_state.status.value}">{run_state.status.value}</span>
  <span class="muted">current: {run_state.current_node} · auto-refresh {refresh}s</span>
</h1>
<div class="mermaid">
{diagram}
</div>
<table>
<thead><tr><th>seq</th><th>ts</th><th>from</th><th>to</th><th>event</th><th>failure_class</th><th>attempt</th></tr></thead>
<tbody>
{history_rows}
</tbody>
</table>
{banner}
{mermaid_script}
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
</body>
</html>
"""


def write_graph_snapshot(
    persist_dir: Path,
    graph: WorkflowGraph,
    run_state: RunState,
    refresh_sec: int = 1,
) -> Path:
    """persist_dir / workflow_graph.md + workflow_graph.html 을 덮어쓴다 (live view).

    overwrite-only — 항상 최신 상태가 파일에 남는다.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "workflow_graph.md"

    parts = [
        f"# Workflow Graph — {graph.name} (status: {run_state.status.value})",
        "",
        "```mermaid",
        render_mermaid(graph, run_state),
        "```",
        "",
        "## History",
        "",
        "| seq | ts | from | to | event | failure_class | attempt |",
        "|-----|----|------|----|-------|---------------|---------|",
    ]
    for r in run_state.history:
        parts.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                r.seq,
                _cell(r.ts),
                _cell(r.from_node),
                _cell(r.to_node),
                _cell(r.event),
                _cell(r.failure_class or ""),
                r.attempt,
            )
        )
    parts.append("")
    write_text_atomic(path, "\n".join(parts))

    write_graph_html(persist_dir, graph, run_state, refresh_sec=refresh_sec)
    return path


def write_graph_html(
    persist_dir: Path,
    graph: WorkflowGraph,
    run_state: RunState,
    refresh_sec: int = 1,
) -> Path:
    """persist_dir / workflow_graph.html 을 덮어쓴다 (overwrite-only)."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "workflow_graph.html"
    write_text_atomic(path, render_html(graph, run_state, refresh_sec=refresh_sec))
    return path


def open_graph_view(html_path: Path) -> None:
    """HTML 을 기본 브라우저로 연다. Windows 는 os.startfile, 그 외 webbrowser.

    절대 예외를 올리지 않는다 (실패 시 [WARNING] 로그만 남기고 무시).
    """
    path = Path(html_path)
    if not path.is_file():
        print(f"[WARNING] graph view 파일 없음: {path}")
        return
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
            print(f"[INFO] 기본 브라우저로 열기: {path}")
            return
        opened = webbrowser.open(path.resolve().as_uri())
        if not opened:
            print(f"[WARNING] 브라우저 열기 실패: {path}")
    except Exception as exc:
        print(f"[WARNING] graph view 자동 열기 실패: {exc}")


def print_ascii(graph: WorkflowGraph, run_state: RunState) -> None:
    """ascii 지도를 [INFO] prefix 로 출력한다."""
    for line in render_ascii(graph, run_state).splitlines():
        print(f"[INFO] {line}")