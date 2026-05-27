# RCS 툴 선택: Spotting 단독 → VLM-우선 + Spotting 검증 전환

날짜: 2026-05-27 09:06
대상 파일: `poc/workflow_1/workflow_select_tool.py`

## 1. 진행 사항

- **문제 진단**: `select_tool_from_main_window` 의 장비 접속이 느리고 자주 실패하는 원인 분석.
  - 클릭 경로(`select_tool_from_main_window`)가 VLM grounding(`analyze_window_target`, ui-venus→mai-ui)을 전혀 쓰지 않고 PaddleOCR `Spotting` 만으로 동작하고 있었음. VLM 경로는 `verify_tool_visible_in_list` 에만 연결되어 있었음.
  - **느림 원인**: 매 캡처마다 4개 crop 시도(`focused_left`/`default`/`wide`/`full`)에 각각 Spotting 호출 → 최대화 재시도 ×4 → 스크롤 루프 최대 8회 ×4 = 최악 ~40 OCR 호출.
  - **실패 원인**: `tool_name_match.best_match` 가 canonicalize + 정확일치 + 행 모호성 가드를 써서, 작은 list 텍스트의 OCR 오인식이나 혼동맵(O→0, S→5 등) 충돌 시 곧바로 `tool_name_not_visible` 반환.
- **설계 결정 (사용자 선택)**: "VLM region + Spotting verify" 방식 채택. VLM 이 행 영역을 먼저 잡고, 그 좁은 strip 에 Spotting 1회로 텍스트 검증 후 클릭.
- **RCS 창 레이아웃 정보 반영**: 장비 ID 는 최좌측 MC ID 컬럼에 세로 나열, 각 ID 좌측에 신호등 박스(녹=On/검=Off). 유용 컬럼은 MC ID·Model·Count·DVR·Connection User. crop 은 왼쪽 집중.
- **커밋/푸시 완료**: auto-commit 훅이 `2cbe28a`, `b086653` 로 커밋, `origin/main` 에 반영됨.

## 2. 수정 내용

`poc/workflow_1/workflow_select_tool.py`:

- **신규 `_build_verify_strip(main_image, point)`**: VLM 이 찾은 point 주변에 한 행 높이(`pad_y = max(36, 0.022·H)`)의 가로 strip crop box 생성. strip 이 한 행 높이라 `best_match` 의 행 모호성 가드가 자연히 같은 행 안으로만 제한 → 잘못된 행 클릭 방지.
- **신규 `_locate_tool_via_vlm_then_verify(...)`**: 새 1차 로케이터.
  1. 입력 이미지를 `LIST_REGION_*` 비율로 왼쪽 list 영역(≈left 0→0.42·W)으로 crop 후 VLM 입력으로 사용 (우측 Connection User 컬럼 혼동 방지).
  2. `analyze_window_target`(ui-venus→mai-ui) 로 region 내 tool row point 검출 → region 좌표를 full image 좌표로 복원.
  3. `_build_verify_strip` 으로 strip crop → `_run_list_spotting` 1회 → `best_match` 로 tool ID 텍스트 검증.
  4. 매칭 성공 시 그 bbox 중심을 클릭점으로(텍스트 검증됨, `detection_source="vlm_then_spotting"`). 실패 시 mai-ui refined point 로 fallback(`detection_source="vlm_only"`).
- **`select_tool_from_main_window` 본문 재작성**: 기존 `_spotting_locate`(4-crop 팬아웃) 제거, `_locate` 클로저가 `_locate_tool_via_vlm_then_verify` 호출. 최대화/스크롤 재시도 루프는 유지하되 새 로케이터를 구동. summary JSON 필드를 `detection_source`/`verify_crop_box`/`vlm_point`/`locate_attempts` 로 갱신.
- **`_tool_row_target` 설명 강화**: "leftmost MC ID column", 좌측 신호등 박스(녹=On/검=Off), 우측 컬럼(RCS IP/Location/Model/Status/Count/DVR/Connection User) 무시 앵커 추가.
- **유지**: `verify_tool_visible_in_list` 및 그 의존 함수(`_locate_tool_via_spotting`, `_build_list_crop_attempts`, `_run_tool_list_ocr_attempts`, `_locate_tool_on_attempts`)는 기존 Spotting 경로 그대로 유지 (dead code 없음).

호출당 API 호출이 ~40 → ~2-3회로 감소.

## 3. 다음 단계

- **오피스(Windows)에서 실측 검증 필요** (Mac 에선 RCS 구동 불가). `uv run python poc/workflow_1/connect_tool.py` 또는 align-fail 플로우 실행.
  - 확인할 디버그 artifact: `*_vlm_*` (ui-venus/mai-ui 오버레이, 이제 왼쪽 region crop 기준), `tool_verify_*_spotting_overlay.jpg` (strip + 매칭 박스 gold), summary JSON 의 `detection_source`/`region_box`/`vlm_point`.
- **검토 대기 항목**: `verify_tool_visible_in_list` 도 VLM-우선으로 전환할지 여부 (현재는 접속 경로가 아니라 미변경). 실측 후 사용자 판단 필요.
- strip 가로폭/`pad_y` 등 파라미터는 실데이터로 캘리브레이션 여지 있음.

## 4. 메모리 업데이트

- 신규 메모리 추가: `project_rcs_tool_list_layout.md` (RCS List 탭 컬럼 레이아웃 + 신호등 On/Off, crop 왼쪽 집중 규칙). `MEMORY.md` 인덱스에 등록 완료.
