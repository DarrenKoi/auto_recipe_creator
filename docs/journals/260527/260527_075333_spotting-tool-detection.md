# 세션 저널 — Spotting 기반 Tool 선택 강건화

- 날짜: 2026-05-27 07:53
- 대상 모듈: `poc/workflow_1/workflow_select_tool.py` 및 신규 보조 모듈
- 주제: RCS List 탭에서 tool(EQP_ID) 접근 실패(잘못된 행 클릭 / crop 클리핑) 해결

---

## 1. 진행 사항

이번 세션은 "일부 tool 접근 실패 — 프롬프트와 이미지 캡처가 충분히 강건하지 않다"는
문제 제기에서 출발했다.

- **현황 진단**: `workflow_select_tool.py` 의 tool 탐지 구조를 분석.
  - OCR 패스(`paddleocr-vl-1.5` + `OCR:` 프롬프트)는 "보이는지"만 확인하고,
    실제 클릭 좌표는 별도의 ui-venus→mai-ui grounding(`analyze_window_target`)이
    독립적으로 산출 → 두 결과가 서로 다른 행을 가리켜 **잘못된 행 클릭** 발생.
  - crop 비율이 `LEFT=0.00, RIGHT=0.42` 로 고정되어 list 패널을 **클리핑**.
  - `build_ocr_assist_prompt` 는 `focus_words`/`context_label` 을 무시하고 `"OCR:"` 만 반환.
- **사용자 확인 실패 모드**: "잘못된 행 클릭" + "crop 클리핑", tool ID 는 영숫자 혼합
  (스크롤은 현재 범위 외).
- **핵심 발견**: `paddleocr-vl-1.5` 의 `Spotting:` 태스크는 검출 텍스트마다 bbox 를
  함께 반환 → 매칭된 텍스트 박스 중심이 곧 클릭 좌표가 되어 "텍스트 매칭"과
  "클릭 좌표"가 분리될 수 없음. `poc/work2/tool_screen_spotting.py` 에 검증된 파서 존재.
- **설계 결정(사용자 승인)**: ① Spotting 우선, ui-venus→mai-ui fallback 유지.
  ② 매칭은 canonicalize(혼동 문자 정규화) + exact.
- **구현 후 코드 리뷰**(3개 finder 앵글 병렬) 수행 → critical 버그 1건 발견·수정.

## 2. 수정 내용

### 신규 파일
- `poc/workflow_1/ocr_spotting.py`
  - `parse_spotting_items(raw_text)` — Spotting JSON 의 다양한 형태(dict bbox,
    `[x1,y1,x2,y2]`, polygon, 중첩 wrapper)를 공통 `{text, bbox}` 로 정규화.
  - `poc/work2/tool_screen_spotting.py` 의 파서를 work2 의존성 없이 자립 이식.
- `poc/workflow_1/tool_name_match.py`
  - `canonicalize(text)` — 대문자/영숫자화 + 혼동 문자 매핑(O,Q→0 / I,L→1 / B→8 /
    S→5 / Z→2 / G→6). **D→0, T→7 은 의도적으로 제외**(MCD 계열 접두 보존).
  - `best_match(items, target)` — canonical 토큰 정확 일치, bbox 최소 면적 tie-break.
- `poc/workflow_1/test_tool_name_match.py`
  - 오프라인 테스트(RCS 불필요) 7케이스. `MCD63O`/`MCDG30` 은 `MCD630` 매칭,
    `MCD680` 은 비매칭 등. **7/7 통과**.

### 변경 파일
- `poc/workflow_1/prompts/prompt_ocr_assist.py`
  - `build_spotting_prompt()` 추가(`("", "Spotting:")`), `__all__` 갱신.
- `poc/workflow_1/prompts/__init__.py`
  - `build_spotting_prompt` 재노출.
- `poc/workflow_1/workflow_select_tool.py`
  - crop 시도 확장: 기존 `focused_left`/`default` + `wide`(right≈0.55) +
    `full`(전체 창) fallback.
  - `_run_list_spotting(...)` 추가 — Spotting 호출, webp/raw/json 디버그 저장,
    `(items, model_name)` 반환.
  - `_locate_tool_via_spotting(...)` 추가 — crop별 Spotting→`best_match`→bbox 중심을
    `_map_point_from_working_image` 로 base crop 좌표 복원→클릭 좌표 산출.
  - `_save_spotting_overlay(...)` 추가 — crop별 검출 박스 오버레이(lime=전체,
    gold+십자=매칭 행) 저장: `..._tool_list_<attempt>_spotting_overlay.jpg`.
  - `select_tool_from_main_window` / `verify_tool_visible_in_list` 를 Spotting 우선
    → OCR/VLM fallback 구조로 재구성. summary JSON 에 `detection_source`,
    `spotting_attempts` 추가.

### 버그 픽스(코드 리뷰 결과)
- **[critical] `workflow_select_tool.py` 성공 경로 KeyError** — 최종 성공 return 의
  `ocr_target_visible=selected_attempt["ocr_result"]["target_visible"]` 가 Spotting
  경로에서 `ocr_result` 키 없는 crop-attempt dict 를 참조해 `KeyError` 발생.
  (이전 일괄치환이 12-space 들여쓰기만 잡고 8-space 최종 return 을 놓침.)
  → 지역변수 `ocr_target_visible` 사용으로 수정. import 정상 + 테스트 7/7 통과 확인.

## 3. 다음 단계

- **오피스(Windows/실 RCS) 검증 필요** — Mac 에서는 RCS 구동 불가.
  - `uv run python poc/workflow_1/workflow_select_tool.py` 실행 후 summary JSON 의
    `detection_source="spotting"` 확인, `*_spotting_overlay.jpg` 에서 gold 박스가
    올바른 행에 위치하는지, 실제로 맞는 tool 이 열리는지 확인.
  - 과거 오클릭하던 영숫자 ID 로 회귀 테스트.
- **미해결 리뷰 지적(사용자 결정 대기)** — 자동 수정하지 않고 보고만 함:
  - 혼동 매핑 충돌: 실 tool 군에 혼동 슬롯만 다른 ID 가 있으면 오매칭 가능.
    → 동일 target 으로 canonical 일치하는 텍스트가 2개 이상이면 매칭 거부+VLM fallback 가드 옵션.
  - `full` 전체창 crop 이 list 그리드 밖(타이틀바/툴팁)의 동일 텍스트를 매칭할 위험.
    → `full` 매칭을 창 좌측 영역으로 제한하는 가드 옵션.
  - `ocr_spotting.py` 의 hinted_text 라벨 누수 / 단일키 dict 의 4-숫자 강제 bbox 변환
    — work2 검증 파서 유래의 저확률 엣지.
- **확인 질문**: 위 두 가드(혼동 충돌 / full 영역 제한)를 적용할지 사용자 결정 필요.

## 4. 메모리 업데이트

`MEMORY.md` 에 새 도메인 패턴 1건 추가 권장(미반영 상태):
- PaddleOCR-VL `Spotting:` 태스크 = 텍스트+bbox 동시 반환 → GUI 클릭 좌표 직접 산출.
  workflow_1 tool 선택은 Spotting 우선 + canonicalize-exact 매칭 + VLM grounding fallback 구조.
  (이는 코드/CLAUDE.md 에서 직접 도출 가능한 부분도 있으나, "왜 Spotting 우선인가"
  = 텍스트 매칭과 클릭 좌표의 분리 방지라는 의도는 메모리 가치가 있음.)

→ 사용자 승인 시 `MEMORY.md` 의 Domain Constraints 섹션에 항목 추가 예정.
