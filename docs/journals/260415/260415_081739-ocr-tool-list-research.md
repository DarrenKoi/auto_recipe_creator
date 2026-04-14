## 1. 진행 사항
- `poc/workflow_1/monitor_align_fail.py`를 확인해 `filter_align_fail`가 `pd.DataFrame`을 반환하는지 추적했다.
- `poc/workflow_1/office_align_fail_alarm.py`에서 `filter_align_fail(df: pd.DataFrame) -> pd.DataFrame` 구현을 확인했고, `fails.empty` 사용이 유효하다는 점을 정리했다.
- `poc/workflow_1/monitor_align_fail.py`의 row iteration을 `iterrows()` 대신 `itertuples(index=False)`로 바꾸는 수정 작업을 한 뒤, 후속 대화에서 최종 워킹트리 상태가 clean인 것을 다시 확인했다.
- `poc/workflow_1/workflow_select_tool.py`의 tool 선택 흐름을 읽고, OCR 검증과 `UI-Venus + MAI-UI` grounding 경로를 분석했다.
- `poc/workflow_1/vlm_client.py`, `flask_api/vlm_serve/service_template.py`를 읽어 `paddleocr-vl-1.5` 요청에 `system_message`와 `user_text`가 실제 payload로 포함되고 프록시가 바디를 그대로 upstream에 전달하는지 확인했다.
- `docs/setup_vlms/03-ocr-and-parser-services.md`, `docs/gui_control/03-ui-venus-ocr-crop-retry.md`를 읽어 PaddleOCR-VL-1.5는 rich prompt보다 `OCR:` / `Spotting:` task keyword 사용이 repo 기본 전략이라는 점을 확인했다.
- `uv run python -m py_compile poc/workflow_1/workflow_select_tool.py`로 중간 작업 파일의 문법을 확인했다.
- `git status --short`와 `git branch --show-current`로 최종 워킹트리 상태가 clean이며 현재 브랜치가 `main`인 것을 확인했다.

## 2. 수정 내용
- 영구 반영된 실코드 변경은 없음. 최종 확인 시점 기준 워킹트리에 남아 있는 수정 파일은 없었다.
- OCR 관련 결론만 정리했다:
- `poc/workflow_1/vlm_client.py`: `system_message`와 `user_text`를 OpenAI-compatible `messages` payload로 전송함.
- `flask_api/vlm_serve/service_template.py`: 프록시가 요청 바디를 변경하지 않고 전달하며, 필요 시 `stream=true`만 보정함.
- `poc/workflow_1/prompts/prompt_ocr_assist.py`: 현재 구현은 사실상 `OCR:`만 반환하는 단순 builder임.
- 이번 세션에서 새로 생성된 파일: `docs/journals/260415/260415_081739-ocr-tool-list-research.md`

## 3. 다음 단계
- `poc/workflow_1/workflow_select_tool.py`에서 OCR prompt를 복잡하게 늘리기보다 left-side focused crop을 먼저 적용하고, crop 이미지를 업스케일한 뒤 `OCR:`로 재시도하는 경로를 추가한다.
- 동일 흐름에서 필요 시 `Spotting:` 실험 pass를 별도로 두어 tool list 좌표 힌트를 비교한다.
- 작은 crop 재판독이 계속 실패하면 `got-ocr` fallback을 붙여 text reread 전용 경로를 만든다.
- 위 변경은 office Windows 환경에서 실제 RCS List 탭 캡처로 검증해야 한다.

## 4. 메모리 업데이트
- 변경 없음
