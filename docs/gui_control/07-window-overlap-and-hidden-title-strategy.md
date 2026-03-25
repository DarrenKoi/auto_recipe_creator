# Overlapped Window / Hidden Title 대응 전략

이 문서는 tool screen 안에서 여러 child window 가 서로 겹쳐 보여서 일부 창이 잘 보이지 않거나,
window title text 일부가 다른 창 뒤에 가려져 VLM/OCR 이 정확히 읽지 못하는 경우를 다루는 계획입니다.

핵심 목표는 다음 두 가지입니다:

1. 가려짐(occlusion)을 먼저 감지한다.
2. read-only 추정만으로 충분하지 않으면, low-risk z-order recovery action 으로 가시성을 복구한다.

중요한 전제:

- 현재 `poc/work2/` mainline 은 여전히 `observe -> decide -> act -> verify` 기준을 유지한다.
- hidden text 추정은 "임시 판단" 이며, 실제 click/action 의 근거로 바로 승격하지 않는다.
- overlap 해소 action 은 office Windows 환경에서 검증된 입력만 허용한다.

## 1. 문제 정의

이번 문제는 단순 OCR 품질 문제가 아닙니다.
실제 root cause 는 다음 세 부류가 섞여 있을 가능성이 큽니다:

- child window 여러 개가 tool 내부에서 겹쳐짐
- title bar 또는 title text 일부가 다른 panel/dialog 뒤에 들어감
- focus 전환 또는 refresh 중간 상태에서 반쯤 가려진 화면이 캡처됨

즉, "VLM 이 못 읽는다" 가 아니라 먼저 "지금 캡처가 읽을 수 있는 상태인가" 를 판단해야 합니다.

## 2. 실패 모드

주요 failure mode:

- title bar 전체는 보이지만 text 중간이 다른 window edge 에 가려짐
- child window 몸체만 보이고 title 은 거의 안 보임
- 앞 window 가 반투명/비반투명 overlay 로 뒤 window 의 핵심 text 를 덮음
- tool 내부 window 가 focus 이동 때마다 z-order 가 바뀜
- scroll/click 직후 transition frame 이 저장되어 실제 stable layout 이 아님

이 경우 OCR/VLM 은 아래와 같은 오류를 낼 수 있습니다:

- title 오독
- 다른 window 의 title 을 target 으로 착각
- 존재하지 않는 control 을 hallucination
- partially hidden text 를 과도하게 확신

## 3. 기본 전략

권장 기본 루프:

`capture -> occlusion detect -> visibility score -> infer or recover -> recapture -> analyze`

여기서 중요한 점:

- 먼저 "읽을 가치가 있는 화면인가" 를 판단한다.
- visibility 가 낮으면 바로 full-screen VLM 해석으로 가지 않는다.
- recover 가능하면 먼저 창 가시성을 회복한 뒤 다시 읽는다.

## 4. Occlusion 감지 전략

### 4.1 geometry 기반 신호

tool screen 내부 child window 마다 가능한 경우 다음 정보를 유지합니다:

- `window_bbox`
- `title_bar_bbox`
- `visible_ratio`
- `overlap_ratio`
- `z_order_guess`

초기 신호:

- title bar 예상 높이는 보이는데 text region 이 끊겨 있음
- 창 외곽선(rectangle)은 있지만 내부 content 가 다른 창 경계로 잘림
- 서로 다른 창 edge 가 비정상적으로 가까운 거리로 겹침
- 이전 frame 대비 동일 window 위치인데 visible area 만 급감

### 4.2 text 기반 신호

hidden-title 후보 신호:

- title OCR 결과가 너무 짧음
- 좌우 prefix/suffix 만 읽히고 중간 token 이 비어 있음
- 반복 프레임에서 읽히는 조각들이 서로 보완 관계임
- VLM 이 "partially visible", "covered", "obscured" 류 설명을 반복함

예:

- `Recipe Mon...`
- `...Monitor`
- `Rec pe Monitor`

이런 경우는 "text 자체가 없는 것" 보다 "가려져 있는 것" 으로 분류해야 합니다.

### 4.3 temporal 기반 신호

single frame 으로 판단하지 않습니다.

같은 tool screen 에 대해 최근 `2~4` frame 을 유지하고:

- 이전 frame 에서는 title 이 정상이었는지
- 이번 frame 에서 overlap 이 새로 생겼는지
- overlap 위치가 고정인지
- text fragment 가 시간이 지나며 합쳐지는지

를 같이 봅니다.

## 5. Hidden Title 추정 규칙

추정은 가능하지만, confidence class 를 분리해야 합니다.

### 5.1 추정 입력

다음 입력을 같이 사용합니다:

- 현재 frame OCR 조각
- 이전 frame OCR 조각
- VLM 설명
- tool/window 후보 사전
- 해당 화면의 기대 title 목록

후보 사전 예시:

- `Recipe Monitor`
- `Queue Manager`
- `File Manager`
- `Port-C`

이 사전은 repo 의 실제 tool/control 명칭에서 가져옵니다.

### 5.2 추정 규칙

권장 규칙:

1. 현재 frame 단독으로는 완성 title 확정 금지
2. 이전 frame 과 합쳐서 유일 후보가 될 때만 `inferred_title` 생성
3. 후보가 2개 이상이면 `ambiguous` 로 유지
4. inferred result 는 action target 이 아니라 recovery guide 로만 사용

예시 JSON:

```json
{
  "visibility": "partial",
  "occluded": true,
  "ocr_fragments": ["Recipe", "Monitor"],
  "inferred_title": "Recipe Monitor",
  "inference_confidence": 0.78,
  "actionable": false,
  "reason": "Current frame is partially hidden, but recent frames and known title list strongly suggest the title."
}
```

### 5.3 금지 규칙

다음 경우는 추정 사용 금지:

- 현재 frame 과 이전 frame 이 서로 다른 child window 로 보일 때
- 복수 후보가 비슷한 score 일 때
- 추정 결과가 high-risk action 으로 바로 이어질 때
- title 이 아니라 body text 일부만 읽고 window identity 를 확정하려 할 때

## 6. Recovery Action Ladder

read-only 추정만으로 충분하지 않으면, 아래 순서로 low-risk recovery 를 시도합니다.

### 6.1 Level 0: recapture only

- 짧은 settle 후 재캡처
- focus 흔들림/transition frame 제거
- visibility score 가 회복되면 여기서 종료

### 6.2 Level 1: main window foreground

- `poc/work2/util/window_utils.py` 의 foreground/activate 계열을 사용해 main RCS 창을 다시 앞으로 가져옴
- foreground 이후 재캡처

이 단계는 tool 내부 child z-order 문제를 직접 해결하지는 못하지만,
top-level 외부 window 가 덮고 있던 경우를 먼저 제거합니다.

### 6.3 Level 2: local child window focus switch

tool 내부에서 보이는 child window header/title 영역을 click 하여 해당 창을 앞으로 가져오는 방식입니다.

조건:

- click target 이 title/header 안전 영역에 있어야 함
- destructive control 근처가 아니어야 함
- action 후 recapture 로 실제 z-order 변화가 확인되어야 함

### 6.4 Level 3: app-specific z-order recovery

사용자 제안의 `Alt+click` 류 입력은 여기로 둡니다.

중요:

- 이것은 Windows 일반 규칙으로 가정하지 않습니다.
- 실제 office 환경의 target app 에서 "해당 modifier + click 시 뒤로 보내기" 가 검증된 경우에만 활성화합니다.

검증되면 다음 계약으로 제한합니다:

- `.env` opt-in
- 대상 window header 영역에서만 수행
- 1회 수행 후 즉시 recapture
- visibility 개선이 없으면 반복 금지

예시 설정:

- `TOOL_OCCLUSION_ENABLE_ALT_CLICK_BACKWARD=true`
- `TOOL_OCCLUSION_MAX_RECOVERY_ACTIONS=1`

### 6.5 Level 4: operator assist fallback

다음 경우는 사람 개입으로 넘깁니다:

- overlap 이 반복되지만 안전한 recovery action 이 없음
- z-order 가 예측 불가능하게 흔들림
- hidden title 추정 confidence 가 낮음
- recover action 이 tool behavior 를 바꿀 위험이 있음

## 7. Visibility Score

분석 파이프라인 앞단에 간단한 visibility score 를 둡니다.

예시 등급:

- `good`: title 과 body 주요 영역이 충분히 보임
- `partial`: title/body 일부가 가려졌지만 추정 가능
- `poor`: text identity 가 불확실하고 recovery 필요

판정 신호 예시:

- title OCR 길이
- text box 연속성
- overlap ratio
- edge truncation 여부
- 최근 frame 대비 visibility 변화

운영 규칙:

- `good` 만 일반 분석 파이프라인으로 보냄
- `partial` 은 hidden-title inference 후 recovery 여부 결정
- `poor` 는 바로 recovery ladder 로 보냄

## 8. VLM/OCR 프롬프트 전략

질문을 더 좁혀야 합니다.
전체 screen 에 "무슨 창이 보이냐" 를 묻기보다 아래처럼 묻습니다:

- 이 title bar text 가 일부 가려져 보이는가
- visible text fragment 는 무엇인가
- 다른 window 가 앞을 가리고 있는가
- 현재 읽기보다 z-order recovery 가 우선인가

strict JSON 예시:

```json
{
  "occluded": true,
  "visibility": "partial",
  "visible_fragments": ["Recipe", "Monitor"],
  "covered_side": "center",
  "should_recover_before_read": true,
  "confidence": 0.86
}
```

핵심:

- VLM 에 완성 title 을 억지로 guess 하게 하지 않는다.
- fragment, occlusion side, recovery 필요 여부를 먼저 받는다.

## 9. Repo 기준 구현 제안

새 구현은 바로 full automation 으로 가지 말고 단계적으로 나눕니다.

### 9.1 문서/실험 단계

- `docs/gui_control/07-window-overlap-and-hidden-title-strategy.md`
- hidden-title failure screenshot 세트 수집
- overlap 유형별 evidence 정리

### 9.2 read-only detector 단계

제안 파일:

- `poc/work2/tool_screen_occlusion.py`

역할:

- tool screen screenshot 입력
- visible score 계산
- hidden-title fragment 추출
- inferred title 후보 생성
- recovery recommendation 반환

출력 예시:

- `occlusion_report.json`
- marked overlay JPEG
- OCR fragment text dump

### 9.3 safe recovery 단계

확장 후보:

- `poc/work2/util/window_utils.py`
- `poc/work2/util/mouse_utils.py`

추가 책임:

- header safe-zone click helper
- app-specific modifier click helper
- recovery action 전후 capture 비교

### 9.4 integrated flow 단계

기존 연결 후보:

- `poc/work2/tool_screen_read.py`
- `poc/work2/select_tool.py`
- `poc/work2/scan_tools_from_view.py`

통합 순서:

`capture -> occlusion report -> recover if allowed -> recapture -> OCR/VLM read`

## 10. 검증 계획

최소 검증 세트:

1. title 완전 노출 baseline
2. title 좌측 일부 가림
3. title 중앙 일부 가림
4. title 우측 일부 가림
5. child window body 만 가림
6. 다른 top-level window 가 전체 일부를 덮는 경우
7. `Alt+click` recovery 성공/실패 케이스

각 케이스마다 확인할 것:

- visibility class
- OCR fragment 품질
- inferred title 정확도
- recovery 전후 visibility 변화
- false action 발생 여부

## 11. Safety Rule

이 문서의 핵심 safety rule:

- hidden text guess 는 control action 근거가 아니다
- recovery action 은 low-risk header 영역에만 제한한다
- office Windows 실측으로 검증되지 않은 modifier action 은 기본 비활성화다
- recover 후에는 반드시 recapture 한다
- recapture 에서 visibility 가 개선되지 않으면 분석을 중단하거나 operator assist 로 넘긴다

## 12. 권장 v1 범위

v1 에서 먼저 할 일:

1. overlap / hidden-title detection
2. fragment 기반 inferred title 생성
3. visibility score 산출
4. main window foreground + safe header click 까지만 recovery 허용

v1 에서 아직 하지 않을 일:

- high-risk automated control
- 확신 낮은 title guess 기반 클릭
- 반복적인 modifier action
- z-order recovery 와 business action 을 한 루프에 혼합

즉, 첫 단계는 "가려진 상태를 읽을 수 있게 복구하는 보조 계층" 까지만 구현하는 것이 맞습니다.
