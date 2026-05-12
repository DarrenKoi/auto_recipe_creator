"""Step 3 — extract 된 페이지 raw JSON 을 머지해 LLM 친화적 산출물을 만든다.

생성물:
- data/organized/<doc_id>/document.md         : 사람이 읽기 좋은 페이지별 요약
- data/organized/<doc_id>/document.json       : 문서 전체 메타 + 페이지 배열
- data/organized/<doc_id>/pages/page_NNN.json : 페이지 단위 sidecar
"""
