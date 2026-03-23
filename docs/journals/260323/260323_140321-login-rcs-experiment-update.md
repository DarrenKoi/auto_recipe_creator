## 1. 진행 사항
- `git status --short`, `git log --stat --oneline -n 8 -- poc/work2/login_rcs*.py` 기준으로 `login_rcs_*` 파일군의 최근 작업 흐름을 정리했다.
- `poc/work2/login_rcs.py`, `poc/work2/login_rcs_common.py`를 기준으로 로그인 창 탐색 로직이 공용 헬퍼로 분리되었고, `find_login_window()`가 `poc/work2/logs/open_rcs_state.json` 기반 PID 우선 탐색 + desktop scan fallback 구조로 정리된 상태를 확인했다.
- `poc/work2/login_rcs.py`는 `resolve_service_slugs_from_env()` + `run_login_benchmark()` 조합으로 로그인 화면 11개 타겟 요소를 공통 벤치마크하는 읽기 전용 진입점으로 유지되고 있음을 확인했다.
- `poc/work2/login_rcs_paddleocr.py`, `poc/work2/login_rcs_got_ocr.py`를 통해 로그인 창 텍스트 전용 OCR 실험 축이 `paddleocr-vl-1.5`와 `got-ocr`로 분리되어 있는 것을 확인했다.
- `poc/work2/login_rcs_ui_venus_rev2.py`, `poc/work2/login_rcs_ui_venus_mai.py`를 통해 UI-Venus 단일 요소 bbox grounding 실험과 `ui-venus -> mai-ui` 2단계 타겟팅 파이프라인이 추가된 흐름을 확인했다.
- `poc/work2/login_rcs_ui_tars.py`, `poc/work2/login_rcs_ui_tars_rev2.py`를 통해 UI-TARS 공식 click action 계약, smart-resize 좌표 역변환, per-element grounding 체크 실험이 별도 스크립트로 유지되고 있음을 정리했다.

## 2. 수정 내용
- 수정 파일: `poc/work2/login_rcs_common.py`
  `login_rcs_Rev2.py`에 있던 로그인 창 탐색/프로세스 검증 로직을 공용화했고, `find_login_window()`를 재사용 가능한 엔트리로 분리했다.
- 수정 파일: `poc/work2/login_rcs.py`
  공용 `find_login_window()`를 사용하도록 정리했고, 서비스 slug를 환경변수에서 받아 `run_login_benchmark()`에 넘기는 구조로 유지했다.
- 수정 파일: `poc/work2/login_rcs_paddleocr.py`
  로그인 창 텍스트만 읽는 OCR 전용 경로를 추가했고, `LOGIN_RCS_PADDLEOCR_MAX_TOKENS` 가드와 debug JPEG/WebP, raw response/result 저장 흐름을 넣었다.
- 수정 파일: `poc/work2/login_rcs_got_ocr.py`
  GOT-OCR `/v1/ocr` 호출 경로와 `LOGIN_RCS_GOT_OCR_BOX` 파싱, OCR raw/summary 저장 흐름을 추가했다.
- 수정 파일: `poc/work2/login_rcs_ui_venus_rev2.py`
  단일 요소별 bbox grounding 실험 스크립트로 정리했고, 결과 JSON/overlay 저장 흐름을 유지했다.
- 수정 파일: `poc/work2/login_rcs_ui_venus_mai.py`
  `TargetConfig`, `PREDEFINED_TARGETS`, coarse bbox 후 zoom crop을 다시 `mai-ui`로 refine하는 2단계 타겟팅 파이프라인을 추가했다.
- 수정 파일: `poc/work2/login_rcs_ui_tars.py`, `poc/work2/login_rcs_ui_tars_rev2.py`
  UI-TARS action 파서, smart-resize 좌표 역변환, batch/per-element/diagnostic 흐름과 rev2 grounding 체크 스크립트를 유지했다.
- 관찰된 상태: `poc/work2/login_rcs_Rev2.py`
  현재 working tree 에서 삭제 상태(`git status --short` 기준)이며, 일부 스크립트가 아직 이 파일의 `_find_login_window()`에 의존하고 있어 후속 정리가 필요하다.
- 신규 파일: `docs/journals/260323/260323_140321-login-rcs-experiment-update.md`
  이번 세션의 `login_rcs_*` 진행 현황을 정리하는 저널 파일을 추가했다.

## 3. 다음 단계
- `poc/work2/login_rcs_ui_venus.py`, `poc/work2/login_rcs_ui_venus_rev2.py`, `poc/work2/login_rcs_ui_tars.py`, `poc/work2/login_rcs_ui_tars_rev2.py`가 `poc/work2/login_rcs_Rev2.py`를 계속 참조할지, 아니면 `poc.work2.login_rcs_common.find_login_window`로 완전히 전환할지 정리한다.
- 사무실 Windows 환경에서 `uv run python poc/work2/connection_check.py` -> `uv run python poc/work2/open_rcs.py` -> `uv run python poc/work2/login_rcs.py` 순서로 기본 로그인 창 탐색 경로를 재검증한다.
- OCR 경로(`poc/work2/login_rcs_paddleocr.py`, `poc/work2/login_rcs_got_ocr.py`)와 grounding 경로(`poc/work2/login_rcs_ui_venus_mai.py`, `poc/work2/login_rcs_ui_tars_rev2.py`) 중 어떤 축을 주력 유지 경로로 가져갈지 비교 기준을 정한다.

## 4. 메모리 업데이트
- 변경 없음
