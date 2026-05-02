---
tags: [wiki, log]
level: beginner
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: []
---

# Wiki Log

> 모든 ingest, query-saved, lint, manual-edit 작업의 이력.
> 매 작업 후 LLM 또는 사람이 1 항목을 prepend (최신이 위) 한다.

형식: `## [2026-05-02] action | target`

action: `ingest` | `query-saved` | `lint` | `manual-edit`

---

<!-- 새 항목은 이 위에 prepend -->

## [2026-05-02] ingest | raw/journals/ (by 대영)

- Added wiki/components/deploy-vlms-runtime.md, rcs-login-automation.md, rcs-tool-selection.md, work2-vlm-routing.md, workflow-runner.md
- Added wiki/concepts/gui-coordinate-and-window-focus.md, model-and-retrieval-options.md, ocr-task-keyword-strategy.md
- Updated wiki/overview.md and wiki/index.md with journal-derived structure and relative cross-links
- Open question: `poc/work2/action_login.py` currently imports `poc.work2.workflow_login`, but this ingest found the implementation under `poc/workflow_1/workflow_login.py`; package boundary needs confirmation before workflow docs are marked complete.

## [2026-05-02] manual-edit | bootstrap

- LLM Wiki 부트스트랩. 폴더 구조와 SCHEMA 초기화.
- 첫 raw 자료 ingest 대기.
