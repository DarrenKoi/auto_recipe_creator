"""workflow_graph.json -> workflow_graph.html (오프라인, 필요할 때만).

사이클은 `workflow_graph.json` 만 남긴다. 그래프를 눈으로 보고 싶을 때 아래 상수에
경로를 적고 실행하면 같은 폴더에 self-contained HTML 을 만든다.

  uv run python poc/workflow_4/adapters/render_graph_html.py
"""

import os
from pathlib import Path

from poc.workflow_4.adapters.workflow3_cycle import GRAPH_JSON_NAME, load_graph_json
from poc.workflow_4.framework.graph_view import write_graph_html

# workflow_graph.json 파일, 또는 그 아래 어딘가에 그 파일이 있는 폴더(이벤트 폴더 등).
# 폴더면 가장 최근에 쓰인 것 하나를 고른다. 셸 env WF4_GRAPH_JSON 이 1회성 override.
GRAPH_JSON = ""


def render(target: Path) -> Path:
    """target(파일 또는 폴더)의 workflow_graph.json 을 HTML 로 렌더하고 그 경로를 돌려준다."""
    if target.is_dir():
        found = sorted(target.rglob(GRAPH_JSON_NAME), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"{GRAPH_JSON_NAME} 없음: {target}")
        target = found[-1]
    graph, run_state = load_graph_json(target)
    return write_graph_html(target.parent, graph, run_state)


if __name__ == "__main__":
    raw = os.environ.get("WF4_GRAPH_JSON") or GRAPH_JSON
    if not raw:
        print("[ERROR] 파일 상단 GRAPH_JSON 에 workflow_graph.json 경로(또는 폴더)를 적을 것")
    else:
        print(f"[INFO] graph html: {render(Path(raw))}")
