## 1. 진행 사항
- `poc/work2/scan_tools_from_view.py` 파일을 새로 추가해 RCS 메인 창에서 `View` 탭을 연 뒤 visible Tool ID 를 스캔하는 흐름을 구현했다.
- `poc/work2/view_list_tab_rcs.py` 의 `PREDEFINED_TARGETS["view_tab"]` 를 재사용하도록 구성해 기존 View 탭 locator 기준과 맞췄다.
- green box 후보를 색상 기반으로 탐지한 뒤 각 box 를 개별 OCR 하는 방식으로 Tool ID 수집 로직을 만들었다.
- wheel scroll 기반 페이지 순회와 Tool ID dedupe, step별 summary JSON 저장 로직을 추가했다.
- `uv run python -m py_compile poc/work2/scan_tools_from_view.py` 로 새 스크립트 문법 검증을 완료했다.

## 2. 수정 내용
- 신규 파일 추가: `poc/work2/scan_tools_from_view.py`
- `scan_tools_from_view.py` 에 다음 로직을 구현했다.
- `wait_for_rcs_main_window()` 로 메인 RCS 창 탐색
- `analyze_window_target()` + `PREDEFINED_TARGETS["view_tab"]` 로 View 탭 좌표 탐지
- `pynput.mouse` 기반 View 클릭 및 wheel scroll
- `cv2`/`numpy` 기반 green box 탐지와 fallback full-crop 스캔
- `Work2VLMClient(service_slug="paddleocr-vl-1.5")` 기반 box 단위 OCR
- Tool ID 정규화, 후보 추출, 중복 제거, step signature 기반 조기 종료
- 디버그 산출물 저장: 전체 캡처, content crop, green box overlay, box OCR 원문, summary JSON

## 3. 다음 단계
- 사무실 Windows 환경에서 실제 RCS View 화면으로 green box 검출 임계값과 content crop 비율이 맞는지 확인한다.
- 실제 Tool ID 형식에 맞춰 `.env` 의 `SCAN_VIEW_TOOL_ID_REGEX`, `SCAN_VIEW_CONTENT_*`, `SCAN_VIEW_GREEN_*` 값을 조정한다.
- 실제 자동 스크롤을 사용할 때는 `SAFE_MODE=false` 또는 `SCAN_VIEW_ACTION_ENABLED=true` 로 실행해 View 클릭과 wheel scroll 동작을 검증한다.

## 4. 메모리 업데이트
- 변경 없음
