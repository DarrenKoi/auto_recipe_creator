# VLM 기반 후행 검증 (Post-Action Verification)

## 3.1 현재 유일한 검증 패턴

현재 `action_login.py`의 유일한 후행 검증:

```python
# 4. 로그인 성공 확인 — 메인 RCS 창 대기
rcs_window, rcs_title, _ = wait_for_rcs_main_window()
login_verified = rcs_window is not None
```

이것은 **pywinauto 윈도우 타이틀 체크**입니다.
VLM을 사용한 시각적 검증은 없습니다.

## 3.2 Before/After 비교 전략

핵심 아이디어: **액션 전후 스크린샷을 VLM에 보내서 기대한 변화가 일어났는지 확인한다.**

```
[before 캡처] → [액션 수행] → [poll-until-stable] → [after 캡처] → [VLM 검증]
```

중요 규칙:

- after 캡처는 반드시 **액션 수행 후 새로 찍어야** 합니다 (stale 방지, doc 04 참조)
- before/after 둘 다 VLM에 전송하되, 질문은 **구체적**이어야 합니다
- "무엇이 바뀌었나?"가 아니라 "이 특정 변화가 일어났는가?"로 질문합니다
- before/after 비교 전에 frame 이 stable 한지 먼저 확인합니다
- unstable frame 이면 verify 실패가 아니라 `window_unstable`로 분류하고 recapture 먼저 시도합니다

### Poll-Until-Stable 안정화 대기

고정 sleep (`POST_ACTION_SETTLE`) 대신 **poll-until-stable** 방식을 사용합니다.
RCS 같은 레거시 Windows 앱은 렌더링 속도가 가변적이므로 고정 sleep은 너무 짧거나(불안정) 너무 길어(느림) 문제가 됩니다.

```python
def _wait_until_stable(
    self,
    step: WorkflowStep,
    max_wait_sec: float = 3.0,
    poll_interval_sec: float = 0.3,
    similarity_threshold: float = 0.98,
) -> bool:
    """연속 캡처 2장의 유사도가 threshold를 넘을 때까지 대기한다.

    Returns:
        True: 화면이 안정됨
        False: max_wait_sec 내에 안정화되지 않음
    """
    prev_hash = self._capture_and_hash(self.window)
    deadline = time.monotonic() + max_wait_sec

    while time.monotonic() < deadline:
        time.sleep(poll_interval_sec)

        curr_hash = self._capture_and_hash(self.window)
        similarity = self._compare_hashes(prev_hash, curr_hash)

        if similarity >= similarity_threshold:
            return True
        prev_hash = curr_hash

    return False
```

구현 옵션:
1. **Pixel histogram 비교**: `PIL.Image.histogram()` 차이. 현재 의존성만으로 바로 구현 가능
2. **pHash (perceptual hash)**: `imagehash` 라이브러리, 해밍 거리 비교. 선택적 정확도 향상
3. **SSIM**: `skimage.metrics.structural_similarity`. 정확하지만 느림. v1에서는 불필요

v1 권장:
- 기본값은 `PIL.Image.histogram()` 기반 안정화 비교
- `imagehash`가 설치된 환경에서는 pHash를 선택적으로 활성화
- optional dependency 가 없다고 step 자체가 실패해서는 안 되며, 가능한 verifier 로 degrade 해야 합니다

규칙:
- `max_wait_sec`는 step별로 조정 가능합니다 (scroll step은 길게, click step은 짧게)
- 안정화 실패 시 `window_unstable`로 분류하고, 재캡처 후 재시도합니다
- 최소 1회는 반드시 대기합니다 (poll_interval_sec 만큼)

### Optional Dependency / Capability Guard

워크플로 검증기는 환경마다 사용할 수 있는 능력이 다를 수 있습니다.
v1에서는 다음 capability guard 를 명시합니다:

- `PIL` 기반 histogram 비교: **필수 경로**
- `imagehash`: 있으면 사용, 없으면 histogram 으로 fallback
- Windows UIA / pywinauto: 있으면 우선 사용, 없으면 OCR 또는 window title 검증으로 fallback
- 어떤 optional verifier 가 없더라도 `unsafe_to_retry` 상황이 아닌 한, 즉시 crash 하지 않고 capability 로그를 남겨야 합니다

## 3.3 Step Type별 검증 프롬프트

| step_type | 검증 질문 | 검증 수단 |
|-----------|----------|----------|
| `click` (버튼) | "이전에 보이던 dialog가 사라졌는가?" 또는 "버튼이 pressed 상태인가?" | VLM before/after |
| `click` (필드) | "입력 준비 상태가 되었는가?" | UIA/pywinauto 우선, 보조로 OCR/VLM |
| `type` | "입력 필드에 기대한 텍스트가 보이는가?" | OCR (가장 확실) |
| `double_click` | "새로운 화면이 열렸는가?" | VLM + 윈도우 타이틀 체크 |
| `scroll` | "리스트 내용이 변경되었는가?" | VLM before/after |

주의:

- `field_focused`를 VLM 단독으로 판정하는 것은 권장하지 않습니다
- Windows UIA 정보가 있으면 먼저 사용하고, 없을 때만 OCR/VLM을 보조 증거로 씁니다
- password field 는 plaintext OCR 검증 대신 "입력 마스크 개수 증가", "dialog 상태 변화", "submit 후 성공 여부"를 더 신뢰합니다

새 프롬프트 빌더가 필요합니다:

```python
# poc/work2/prompts/prompt_action_verify.py

def build_action_verification_prompt(
    step_type: str,
    target_description: str,
    success_criteria: ConditionGroup,
) -> tuple[str, str]:
    """액션 후행 검증 프롬프트를 생성한다.

    Returns:
        (system_message, user_message) 튜플
    """
    # VLM 응답 형식: {"verified": true/false, "confidence": 0.0-1.0, "reason": "..."}
```

## 3.4 검증 판정 규칙

```
confidence >= 0.8 AND verified == true   → 성공
confidence >= 0.6 AND verified == true   → 약한 성공, 로그 경고, 1회 재검증 가능
verified == false OR confidence < 0.6    → 실패, 기본 동작은 HALT / escalation
타임아웃 (응답 없음)                       → 실패 (failure_class = "verify_timeout")
JSON 파싱 실패 (malformed VLM 응답)       → 실패 (failure_class = "verify_parse_error"), 1회 재검증 후 escalation
```

보강 규칙:

- `type` step 에서 입력 후 검증이 약하면 먼저 **액션 재실행이 아닌** `verify_only` 재캡처를 1회 수행합니다
- 입력이 실제로 반영되었는지 불확실할 때는 곧바로 같은 문자열을 다시 타이핑하지 않습니다
- `verify_failed` / `verify_timeout` 에서 자동으로 허용되는 것은 **verifier 교체 또는 재캡처 기반 재검증** 뿐입니다
- `idempotent=False` step 은 검증 실패 시 기본값이 재실행이 아니라 `halt_non_idempotent` 입니다 ([04-retry-strategy.md](04-retry-strategy.md) Section 4.8 참조)

## 3.5 VLM 실현 가능성 분석

**VLM이 잘 판단하는 것:**

- dialog 출현/소멸 (큰 시각적 변화)
- 새 윈도우 출현 (타이틀 바 + 콘텐츠 변화)
- 버튼/탭의 selected vs unselected 상태
- 입력 필드의 텍스트 존재 여부 (특히 OCR sidecar 병행 시)
- 에러 팝업, 모달 dialog
- 스크롤 후 리스트 내용 변화

**VLM이 어려워하는 것:**

- 커서 깜빡임 (focus 여부 판단)
- 미세한 hover 효과
- 비시각적 상태 변화 (네트워크 상태, 백엔드 데이터)
- 애니메이션 중간 프레임
- 밀도 높은 engineering UI의 작은 체크박스 상태

**완화 전략:**

- poll-until-stable 대기 후 캡처 (고정 sleep 대신)
- OCR cross-check로 텍스트 검증 보강
- 반복 캡처로 일시적 상태 필터링
- structured JSON 응답 + confidence 강제로 hallucination 감소
- 검증이 어려운 step은 `verify_method: "window_title"` 또는 `"skip"`으로 대체 가능

**기존 precedent:**
`poc/work2/prompts/prompt_screen_analysis.py`의 `build_measurement_judgment_prompt()`가 이미 VLM 기반 성공/실패 판정을 수행하고 있습니다. 같은 패턴을 워크플로 검증에 적용 가능합니다.

## 3.6 Hybrid 검증 — 비용 대비 효과

모든 step에 VLM 검증을 걸면 비용과 시간이 과합니다. Hybrid 접근:

| safety_tier | 검증 수단 | 이유 |
|-------------|----------|------|
| Tier 0 | pywinauto 윈도우 타이틀 체크 | VLM 불필요, 빠름 |
| Tier 1 | OCR only (텍스트 필드) | 정확하고 빠름 |
| Tier 2 | VLM before/after 비교 | UI 상태 변화 판단 필요 |
| Tier 3 | VLM + OCR + 사람 확인 | 장비 영향 있는 액션 |

실무 규칙:

- 가능한 경우 "가장 싼 검증"을 먼저 씁니다
- window title / UIA / OCR 로 충분한 step 에는 VLM을 붙이지 않습니다
- VLM 검증은 "시각적 상태 변화가 핵심인 step"에만 집중합니다
