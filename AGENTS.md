# AGENTS.md

**`CLAUDE.md` 가 이 저장소의 유일한 에이전트 지침 정본이다 - 작업 전에 그 파일을 읽을 것.**
Codex / opencode 등 이 파일만 자동 로드하는 에이전트도 같은 규칙을 따른다.

요점 (정본과 어긋나면 `CLAUDE.md` 가 이긴다):

- 운영 패키지는 `poc/workflow_3/` 다. `workflow_1` 은 동결, `workflow_2` 는 오프라인 CV 벤치,
  `workflow_4` 는 FSM 프레임워크 + 순수 도메인(`playbook/`).
- 스크립트는 CLI 인자 없이 `uv run python <script>.py`. 실행 knob 은 진입점 파일 상단 상수,
  `.env` 는 비밀값 전용.
- 전체 테스트: `uv run pytest poc/workflow_3 poc/workflow_4` (Mac 에서 VLM/오피스 없이 돈다).
- 한국어 docstring, `[INFO]`/`[WARNING]`/`[ERROR]` print 로깅, `__future__` import 금지,
  `print()` 문자열에 em-dash 금지(오피스 콘솔 cp949).
