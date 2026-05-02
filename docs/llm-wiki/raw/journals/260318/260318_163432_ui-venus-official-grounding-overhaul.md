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
- 2026-03-18 기준 공개 README 재확인:
  - grounding 과 별도로 navigation evaluation entrypoint(`scripts/run_navi.sh`)가 공개되어 있음
  - navigation 입력 JSON 은 최소 `task`, `image_path` 를 포함해야 함
  - navigation 결과는 execution history JSON 으로 저장됨
  - 공개 README 에서 노출한 `PROMPT_TYPE` 는 `mobile` / `web` 이며, Windows desktop live executor 는 별도 제공되지 않음
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

### Navigation 모드 / agent loop 적용 메모
- 이 문맥에서 navigation 모드는 grounding 처럼 screenshot 1장에 대해 좌표 1개를 받고 끝나는 방식이 아니라, `task + current_screenshot + recent_history` 를 넣고 매 step 마다 **다음 action 1개** 를 받는 루프를 뜻한다.
- 이 repo 에서는 공개 UI-Venus navigation 평가 스크립트를 그대로 Windows RCS 자동화에 붙이기보다, **모델은 다음 액션 제안**, **로컬 executor 는 실제 입력 1 step 실행** 으로 분리하는 구조가 안전하다.
- 권장 루프는 `observe -> decide -> normalize_action -> act -> verify -> append_history -> repeat` 이다.
- 실제 입력 연결 경로는 기존 구현을 재사용하면 된다:
  - mouse click / double click / scroll: `poc/work/mouse_control.py`
  - keyboard typing / hotkey: `poc/work/keyboard_control.py`
  - end-to-end 실행 분기 참고: `poc/work/vlm_rcs_agent.py` 의 `_execute_action()`
- `type` 액션은 바로 문자열만 보내지 말고, `click target -> focus 확인 -> 필요시 ctrl+a -> type_text(text) -> post screenshot 확인` 순서로 실행하는 편이 안전하다. 기존 `VLMRCSAgent._execute_action()` 도 이 패턴을 사용한다.
- 모델 출력이 XML 이든 자유형식이든 그대로 실행하지 말고, 내부 action schema 로 정규화한 뒤 executor 에 전달해야 한다. 최소 스키마 예시는 아래와 같다.

```json
{
  "action_type": "click|double_click|type|hotkey|scroll|wait|done",
  "x": 0,
  "y": 0,
  "text": "",
  "keys": [],
  "confidence": 0.0,
  "expected_effect": ""
}
```

- 좌표 처리 규칙도 모델별로 분리해야 한다:
  - UI-Venus grounding 은 `[0,1000]` 정규화 좌표를 현재 screenshot pixel 로 환산한 뒤 클릭
  - UI-TARS action 좌표는 smart-resize 공간 기준이므로 `poc/work2/login_rcs_ui_tars.py` 처럼 역변환이 필요
- real action 으로 연결하려면 blind loop 를 피해야 한다. 최소한 아래 3개는 매 step 유지해야 한다:
  - pre-action screenshot
  - post-action screenshot
  - verification result (rule/OCR/VLM)
- safety 기본값은 유지한다:
  - `SAFE_MODE=true` 기본
  - low confidence action 은 실행하지 않음
  - destructive action 은 human checkpoint 로 승격
  - 연속 실패 시 중단
- 즉, desktop RCS 에서의 navigation mode 최소 구현은 "모델이 OS 를 직접 조작"하는 구조가 아니라, "모델이 다음 액션을 제안하고 로컬 Python executor 가 1 step 씩 안전하게 실행하는 구조" 로 보는 것이 맞다.

```python
while step < max_steps:
    pre = capture_rcs_window()
    raw_action = plan_next_action(task, pre, history)
    action = normalize_action(raw_action)

    if not is_safe(action):
        ask_human(action)
        break

    execute_action(action)
    post = capture_rcs_window()
    verification = verify_action(post, action)
    history.append({"action": action, "verification": verification})

    if action["action_type"] == "done":
        break
```

---

## 3. 다음 단계

- `TARGET_KEYS` 에 나머지 요소 추가 (`server_input`, `password_input`, `login_button` 등) 및 정확도 검증
- 여러 요소를 순차 호출할 때 latency 측정 → 필요시 병렬 호출 고려
- 다른 화면(main tabs, measurement 등)에도 공식 단일 요소 프롬프트 적용 검토
- `poc/work2` 에 `agent_loop.py` / `action_executor.py` 수준의 분리 구현 검토
- `poc/work` 의 `MouseController`, `KeyboardController`, `VLMRCSAgent._execute_action()` 를 `poc/work2` 기준으로 재사용 또는 이관 검토
- live action 범위는 먼저 `click`, `type`, `hotkey`, `wait` 만 열고, scroll / drag / destructive action 은 나중에 확대
- Navigation 모드 실험은 desktop RCS 전용 prompt + local executor + post-action verification 조합으로 제한해서 검증
- 공식 공개 자료 확인 링크:
  - GitHub: <https://github.com/inclusionAI/UI-Venus>

---

## 4. 메모리 업데이트

UI-Venus 공식 grounding 형식 도입은 프로젝트의 VLM 프롬프트 전략에 중요한 변경이므로 메모리에 기록한다.
이번 navigation mode / agent loop 설명 보강으로 인한 추가 메모리 변경은 없음.
