# UI-Venus 공식 grounding 프롬프트로 전환

**날짜:** 2026-03-18
**세션 요약:** UI-Venus 1.5 모델의 공식 grounding 프롬프트 형식을 조사하고, 기존 커스텀 batch 프롬프트 방식에서 공식 단일 요소 프롬프트 방식으로 전환

---

## 1. 진행 사항

### 리서치
- UI-Venus 1.5 모델의 공식 GitHub repo, HuggingFace 모델카드, 기술 보고서(arXiv) 조사
- 공식 grounding 프롬프트 형식 확인:
  - **단일 요소 프롬프트**: `"Output the center point of the position corresponding to the following instruction: {instruction}. The output should just be the coordinates of a point, in the format [x,y]."`
  - 좌표계: `[0, 1000]` 정규화
  - Refusal: `[-1, -1]` (요소가 보이지 않을 때)
- 기존 방식(11개 요소를 한 번에 요청하는 커스텀 JSON 프롬프트)과 공식 방식(단일 요소씩) 비교 분석
- UI-Venus 1.5 벤치마크 성능 확인: ScreenSpot-Pro 68.4% (UI-TARS 35.7% 대비 약 2배)

### 코드 전환
- `poc/work2/prompts/prompt_login_rcs_ui_venus.py` — 공식 단일 요소 프롬프트 빌더 추가
- `poc/work2/login_rcs_ui_venus_rev2.py` — 공식 형식 기반으로 전면 재작성
- `userid_input` 요소 하나로 시작하여 동작 확인 완료

---

## 2. 수정 내용

### `poc/work2/prompts/prompt_login_rcs_ui_venus.py`
- `build_ui_venus_single_element_prompt(instruction)` 추가 — 공식 UI-Venus 1.5 템플릿 사용, system message 없이 user message 만 전송
- `build_ui_venus_single_element_prompt_by_key(element_key)` 추가 — element key 로 description dict 에서 instruction 자동 조회
- `userid_input` description 을 클릭 의도가 명확하도록 수정
- 기존 `build_login_rcs_ui_venus_prompt()` 는 레거시로 보존 (deprecation 주석 추가)

### `poc/work2/prompts/__init__.py`
- `build_ui_venus_single_element_prompt`, `build_ui_venus_single_element_prompt_by_key` export 추가

### `poc/work2/login_rcs_ui_venus_rev2.py` (전면 재작성)
- ~700줄 → ~230줄로 대폭 축소
- **핵심 변경:**
  - 한 번에 11개 요소 요청 → **요소당 1회 VLM 호출** (공식 형식)
  - 복잡한 JSON 파싱(`extract_json` + `parse_coords`) → **`[x, y]` 정규식 파싱** (`_parse_point_response`)
  - `relative_1000` 좌표 변환 → **`_to_pixel()` 직접 변환** (0-1000 → pixel)
  - `[-1, -1]` refusal 을 None 으로 처리 (hallucination 방지)
- **제거된 로직:**
  - dual prompt 비교 (loose vs anchored)
  - visible-first 자유형식 요소 탐색
  - role normalization, slug 생성, ORIGIN_MARKER 등
- `TARGET_KEYS = ("userid_input",)` 로 시작 — 추후 확장 시 튜플에 추가만 하면 됨

---

## 3. 다음 단계

- `TARGET_KEYS` 에 나머지 요소 추가 (`server_input`, `password_input`, `login_button` 등) 및 정확도 검증
- 여러 요소를 순차 호출할 때 latency 측정 → 필요시 병렬 호출 고려
- 다른 화면(main tabs, measurement 등)에도 공식 단일 요소 프롬프트 적용 검토
- Navigation 모드(`<think>` / `<action>` / `<conclusion>` XML 출력) 활용 가능성 조사 — 수동 step-by-step 스크립팅을 모델 주도 agent loop 으로 대체 가능

---

## 4. 메모리 업데이트

UI-Venus 공식 grounding 형식 도입은 프로젝트의 VLM 프롬프트 전략에 중요한 변경이므로 메모리에 기록한다.
