# Phase 3 (work3): 동적 화면 자동화 — 프로브 모니터 & 실시간 SEM 이미지 대응

## 배경

CD-SEM/VeritySEM 계측 장비의 화면은 **정적이지 않다**. 장비가 자동 레시피를 실행하여 웨이퍼 샘플을 연속 측정하는 동안, 화면의 프로브 모니터 영역은 ~200-500ms 간격으로 실시간 SEM 이미지가 갱신된다. 마우스 조작이 없어도 프로브의 이동, 측정 상태, 획득 이미지가 계속 변한다.

현재 Phase 2 (`poc/work2/`) 자동화는 **단발성(single-shot)** 방식이다: 스크린샷 캡처 → VLM 분석 → 액션 실행. 이는 로그인 화면, 메뉴, 탭 등 **정적 화면**에서는 잘 동작하지만, 동적 화면에서는 다음 문제가 발생한다:

| 문제 | 설명 |
|------|------|
| **Stale frame** | 캡처와 클릭 사이의 1-3초 VLM 지연 동안 화면이 변경됨 |
| **전이 중 캡처** | 상태 전이 도중 캡처하면 VLM이 모호한 프레임을 분석 |
| **프로브 오조작** | 프로브 모니터 영역에 더블클릭 시 프로브가 물리적으로 이동 — 오클릭 위험 |

### 핵심 사실: 프로브 더블클릭

프로브 모니터(실시간 SEM 이미지 표시 영역)에서 **더블클릭**하면, 프로브가 클릭된 지점으로 **물리적으로 이동**한다. 이것은:
- 엔지니어가 측정 위치를 수동 조정할 때 사용하는 기능
- 잘못된 좌표로 이동 시 웨이퍼 손상 가능 → **고위험 조작**
- 자동화 시 반드시 다중 안전 장치 필요

---

## Phase 3 목표: 수동 작업의 10% 자동화

모든 수동 작업을 한 번에 자동화하는 것이 아니라, **3단계 티어**로 위험도에 따라 점진적으로 확대한다.

### 티어 구조

```
Tier 1 (안전)       Tier 2 (중간)         Tier 3 (고위험)
상태 모니터링    →   에러 복구         →   프로브 재위치 지정
읽기 전용           정적 UI 영역 클릭      SEM 영역 더블클릭
VLM 관찰만          VLM 판단 + 버튼 클릭   VLM 좌표 + 물리 이동
```

| 티어 | 대상 작업 | 위험도 | VLM 상호작용 |
|------|-----------|--------|-------------|
| **Tier 1** | 상태 모니터링 | 없음 (읽기 전용) | 주기적 VLM 분석, 클릭 없음 |
| **Tier 2** | 에러 복구 | 낮음 (정적 UI 영역) | VLM이 에러 감지 → 재시도/건너뛰기 버튼 클릭 |
| **Tier 3** | 프로브 재위치 | 높음 (물리적 이동) | VLM이 SEM 이미지 내 타겟 식별 → 안전 가드 거친 더블클릭 |

---

## 핵심 기술 설계

### 1. 프레임 안정성 검출 (`frame_stability`)

동적 화면에서 "지금 안전하게 분석/조작 가능한가?"를 판단하는 핵심 메커니즘.

#### 원리

연속 캡처된 두 프레임의 픽셀 차이를 비교하여 화면이 "안정" 상태인지 판단한다.

```
Frame A (t=0.0s) ──┐
                    ├── ImageChops.difference() → diff_ratio
Frame B (t=0.3s) ──┘

diff_ratio < 0.02 (2%) → "안정" (stable)
diff_ratio ≥ 0.02       → "변동 중" (changing)
```

#### 핵심 개념: 영역별 안정성 (Region-based Stability)

화면 전체가 아니라 **특정 영역만** 안정성을 확인한다. 이것이 동적 화면 대응의 핵심이다.

- **정적 UI 조작 시** (메뉴, 버튼 클릭): SEM 영역을 **제외**하고 나머지만 안정성 확인
  - SEM 이미지는 항상 변하므로, 전체 화면으로 안정성을 판단하면 영원히 "불안정"
  - 툴바/메뉴/상태 패널은 SEM과 무관하게 안정적
- **프로브 조작 시** (더블클릭 후 검증): SEM 영역**만** 안정성 확인
  - 프로브 이동 후 SEM 이미지가 안정 = 프로브가 목표 위치에 도달

```
┌──────────────────────────────────────────────┐
│  [메뉴바]                          (정적)    │
├──────────┬───────────────────────────────────┤
│          │                                   │
│  제어    │     프로브 모니터 (SEM 이미지)     │
│  패널    │     ← 200-500ms 간격 갱신         │
│  (정적)  │     ← 더블클릭 = 프로브 물리 이동  │
│          │                                   │
├──────────┴───────────────────────────────────┤
│  [상태바 / 측정 결과]              (동적)    │
└──────────────────────────────────────────────┘

정적 UI 클릭 → stability_region = 메뉴바 + 제어패널 (SEM 제외)
프로브 이동  → stability_region = 프로브 모니터 (SEM 영역만)
```

#### `wait_for_stability()` 동작 흐름

```
1. capture_func()로 Frame A 캡처
2. poll_interval_sec (0.3s) 대기
3. Frame B 캡처
4. compare_frames(A, B, region=stability_region) → diff_ratio 계산
5. is_stable이면 stable_count += 1, 아니면 stable_count = 0
6. stable_count >= min_stable_count (2) → 안정 확인, 최종 프레임 반환
7. timeout_sec (10s) 초과 시 None 반환 + [WARNING] 로그
```

#### 구현 시 PIL 사용 근거

| 방법 | 의존성 | 속도 (2560x1440) | 선택 |
|------|--------|------------------|------|
| PIL `ImageChops.difference()` | Pillow (이미 사용 중) | ~2-5ms | **채택** |
| numpy 배열 연산 | numpy (work2에 미사용) | ~1-3ms | 불필요한 의존성 |

---

### 2. 화면 영역 정의 (`screen_zones`)

화면을 정적/동적/위험 영역으로 구분하여 클릭 안전성을 검증한다.

#### 영역 타입

| 타입 | 의미 | 클릭 허용 | 예시 |
|------|------|-----------|------|
| `static` | 변하지 않는 UI 요소 | 항상 허용 | 메뉴바, 툴바, 제어 패널 |
| `dynamic` | 내용이 변하지만 클릭 대상 아님 | 경고 후 허용 | 측정 상태 바, 로그 영역 |
| `danger` | 클릭 시 물리적 영향 | 명시적 오버라이드 필요 | 프로브 모니터 (SEM 이미지) |

#### 영역 좌표: 비율 기반 + 환경변수 보정

실제 좌표는 장비마다 다를 수 있으므로, 창 크기 대비 **비율(%)로 기본값**을 설정하고, 사무실에서 환경변수로 미세 조정한다.

```
기본값 예시 (calibration 전 — 실제 스크린샷으로 조정 필요):

메뉴바:       x=0%, y=0%, w=100%, h=3%     → static
툴바:         x=0%, y=3%, w=100%, h=5%     → static
제어 패널:    x=0%, y=8%, w=20%, h=77%     → static
프로브 모니터: x=20%, y=8%, w=80%, h=77%    → danger
상태바:       x=0%, y=85%, w=100%, h=15%   → dynamic

환경변수 오버라이드:
  RCS_ZONE_PROBE_X_PCT=20
  RCS_ZONE_PROBE_Y_PCT=8
  RCS_ZONE_PROBE_W_PCT=80
  RCS_ZONE_PROBE_H_PCT=77
```

#### 영역 보정 워크플로우

```
1. 사무실에서 도구 화면 스크린샷 캡처
2. save_zone_overlay() 실행 → 영역 경계선이 그려진 디버그 이미지 생성
3. 디버그 이미지 확인 → 비율 조정 → 재실행
4. 정확한 비율 확정 후 환경변수에 설정
```

---

### 3. Capture-Verify-Act 패턴

기존 단발성 분석의 한계를 극복하는 3단계 패턴.

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐     ┌─────────┐
│ CAPTURE │ ──→ │ STABILIZE│ ──→ │ ANALYZE │ ──→ │  VERIFY  │ ──→ │   ACT   │
│ (캡처)   │     │ (안정 대기)│     │ (VLM)   │     │ (재캡처)  │     │ (실행)   │
└─────────┘     └──────────┘     └─────────┘     └──────────┘     └─────────┘
                                                       │
                                                  diff_ratio >    YES → [WARNING]
                                                  10%? ────────────────→ 좌표 신뢰
                                                       │                 불가, 재시도
                                                       NO → 안전, 진행
```

1. **CAPTURE**: `capture_window()`로 스크린샷 캡처
2. **STABILIZE**: `wait_for_stability()`로 안정 프레임 확보 (region 기반)
3. **ANALYZE**: 안정 프레임을 VLM에 전송, 상태/좌표 추출
4. **VERIFY**: VLM 응답 수신 후 한 번 더 캡처, 안정 프레임과 비교
   - `diff_ratio > 10%` → 화면이 크게 변함, VLM 좌표 신뢰 불가
   - `diff_ratio ≤ 10%` → 화면 동일, 좌표 유효
5. **ACT**: 검증 통과 시에만 클릭 실행

#### VLM 지연과 sub-second 갱신의 공존

VLM 응답까지 500ms-2000ms 소요되는 동안 SEM 이미지는 1-4회 갱신된다. 이 문제의 해결 핵심:

- **정적 UI 조작**: verify 시 SEM 영역 제외. 메뉴/버튼 영역만 비교하면 diff_ratio ≈ 0
- **프로브 조작**: verify 불가 (SEM 항상 변함). 대신 조작 후 `wait_for_stability()`로 결과 확인

---

### 4. 더블클릭 지원 확장

#### 현황

| 위치 | 더블클릭 지원 | 방식 |
|------|-------------|------|
| `test/vlm_input_control/mouse_control.py` | **있음** | pynput `controller.click(Button.left, 2)` |
| `poc/work2/rcs_utils.py:click_at()` | **없음** | pywinauto single click만 |

#### 확장 방향

`click_at()`에 `click_count` 파라미터를 추가한다. `click_count=2`일 때:

```python
# 1차 시도: pywinauto
window.click_input(coords=(rel_x, rel_y), double=True)

# 2차 시도 (fallback): pynput
from pynput.mouse import Button, Controller
controller = Controller()
controller.position = (abs_x, abs_y)
controller.click(Button.left, 2)
```

#### 3중 안전 장치

프로브 더블클릭은 물리적 결과를 초래하므로 3개 레이어의 안전 장치가 필요하다:

```
Layer 1: 모듈 수준 SAFE_MODE (환경변수, 기본값 True)
    ↓ SAFE_MODE=false일 때만 통과
Layer 2: 영역 안전성 검증 (screen_zones)
    ↓ danger zone이면 차단, 명시적 override 필요
Layer 3: click_at() 파라미터 검증
    ↓ click_count=2이면 danger_zone_override=True 필요

세 레이어 모두 통과해야 더블클릭 실행
```

실행 차단 시 `[SAFE_MODE]` 접두사로 로그를 남겨, 사무실에서 무엇이 차단되었는지 확인 가능.

---

### 5. 프로브 모니터 전용 VLM 프롬프트

SEM 이미지 특성에 맞춘 전용 프롬프트 빌더가 필요하다.

#### 기존 프롬프트 빌더와의 차이

| 항목 | 기존 (RCS 로그인 등) | 프로브 모니터 전용 |
|------|---------------------|-------------------|
| 이미지 특성 | 정적 GUI (텍스트, 버튼) | 실시간 SEM (전자현미경 이미지) |
| 모션 블러 | 없음 | 있을 수 있음 |
| 타겟 요소 | 버튼, 입력필드 좌표 | 프로브 위치 마커, 측정 타겟 |
| 클릭 결과 | 소프트웨어 동작 | **물리적 프로브 이동** |
| 좌표 검증 | 창 경계 내 확인 | 프로브 모니터 경계 내 확인 |

#### 추출 대상 요소

```json
{
  "probe_current_position": {"x": 450, "y": 320},
  "probe_target_position": {"x": 600, "y": 280},
  "probe_monitor_boundary": {"x": 400, "y": 200}
}
```

- `probe_current_position`: 현재 프로브 크로스헤어/마커 위치
- `probe_target_position`: 더블클릭으로 이동할 목표 위치
- `probe_monitor_boundary`: SEM 표시 영역 중심점 (영역 검증용)

#### 프롬프트 설계 시 주의사항

- SEM 이미지는 실시간 캡처로 모션 블러 가능성을 명시
- `probe_target_position`이 물리적 프로브 이동을 유발함을 강조
- 프로브 모니터 경계 외부의 좌표는 무효임을 명시
- OCR 보조 힌트 (`extra_instructions`) 수용 — 측정 파라미터 텍스트 활용

---

### 6. 연속 모니터링 루프 (Tier 1 자동화)

#### 목적

엔지니어가 화면을 지속 주시하지 않아도, 시스템이 상태 변화를 감시하고 알려준다.

#### 동작 흐름

```
[시작] → 상태: "unknown"
  │
  ├── 반복 (poll_interval_sec=2.0s마다)
  │     │
  │     ├── capture_verify_act(stability_region=정적 영역만)
  │     │     → ScreenAnalysisResult (state_id, confidence, ui_elements)
  │     │
  │     ├── 상태 변경 감지?
  │     │     YES → StateTransition 기록
  │     │           [STATE] measuring → idle (confidence=0.92)
  │     │
  │     ├── target_states에 해당?
  │     │     YES → [ALERT] 에러 상태 감지!
  │     │
  │     └── auto_act=True이고 suggested_actions 있으면?
  │           → 영역 안전성 검증 후 실행 (정적 영역만)
  │
  └── stop() 호출 또는 max_iterations 도달 시 종료
```

#### 상태 전이 추적

```python
StateTransition(
    timestamp=1710000000.0,
    from_state="measuring",
    to_state="error",
    confidence=0.85,
    frame_stable=True,    # 안정된 프레임에서 감지했는지 여부
)
```

상태 이력은 리스트로 누적되어, 사후 분석 가능:
- 측정 사이클 패턴 분석
- 에러 발생 빈도/시점 파악
- 프로브 동작과 상태 변화 상관관계

#### Tier 2 확장: 에러 자동 복구

상태가 `"error"`로 감지되면, VLM의 `suggested_actions`에 "retry 버튼 클릭" 또는 "skip 버튼 클릭"이 포함될 수 있다. 이 버튼들은 정적 UI 영역 (메뉴/툴바)에 위치하므로:

```
에러 감지 → suggested_action = "click retry button"
         → retry 버튼 좌표: (150, 45) → 영역 확인: toolbar (static) → 안전
         → click_at("retry_button", ...) 실행
```

`auto_act=False` (기본값)에서는 알림만 출력하고, 엔지니어가 `auto_act=True`로 전환하여 자동 복구를 활성화한다.

---

## 프로브 재위치 지정 (Tier 3, 별도 스크립트)

가장 고위험 조작이므로 연속 모니터링과 분리하여 **독립 스크립트**로 실행한다.

### 실행 흐름

```
1. 도구 창 탐색 (check_tool_screen 패턴 재사용)
2. wait_for_stability() — 전체 화면 안정성 확인 (도구가 "준비" 상태인지)
3. 스크린샷 캡처 + build_probe_monitor_prompt()로 VLM 분석
   → probe_current_position, probe_target_position 추출
4. 영역 검증: target 좌표가 probe_monitor_boundary 내부인지 확인
5. ── SAFE_MODE=true (기본값) ──
   │  [SAFE_MODE] 좌표 출력 + 디버그 이미지 저장 → 종료
   │  엔지니어가 디버그 이미지 확인
   │
   └── SAFE_MODE=false ──
      6. click_at(click_count=2, danger_zone_override=True) → 더블클릭
      7. wait_for_stability(region=프로브_모니터_영역) → SEM 안정 대기
         → 안정 = 프로브가 목표에 도달
      8. 재캡처 + VLM 분석 → probe_current_position이 target 근처인지 검증
      9. before/after 디버그 이미지 저장
```

### 검증 기준

```
이동 전 probe_current_position: (450, 320)
이동 목표 probe_target_position: (600, 280)
이동 후 probe_current_position: (595, 283)  ← target과 ±10px 이내 → 성공

허용 오차: ±10px (환경변수 RCS_PROBE_MOVE_TOLERANCE_PX로 조정)
```

---

## 구현 대상 파일 목록

| 파일 | 작업 | 목적 |
|------|------|------|
| `poc/work2/frame_stability.py` | 신규 | 프레임 비교, 안정성 검출 |
| `poc/work2/screen_zones.py` | 신규 | 영역 정의, 클릭 안전성 검증, 디버그 오버레이 |
| `poc/work2/continuous_monitor.py` | 신규 | 상태 모니터링 루프, 에러 복구 오케스트레이션 |
| `poc/work2/automate_probe_move.py` | 신규 | 프로브 재위치 (안전 가드 포함) |
| `poc/work2/prompts/probe_monitor.py` | 신규 | SEM 이미지 분석용 VLM 프롬프트 |
| `poc/work2/prompts/__init__.py` | 수정 | 신규 프롬프트 빌더 re-export |
| `poc/work2/rcs_utils.py` | 수정 | `click_at()`에 `click_count`, `safe_mode` 추가 |
| `poc/work2/vlm_screen_analysis.py` | 수정 | `capture_verify_act()` 메서드 추가 |

---

## 테스트 계획

### macOS (Claude Code) — 단위 테스트

코드 작성 직후 macOS에서 실행 가능한 테스트. 실제 장비 불필요.

#### `frame_stability.py` 테스트

| 테스트 케이스 | 입력 | 기대 결과 |
|-------------|------|-----------|
| 동일 프레임 비교 | 같은 흰색 이미지 2장 | `is_stable=True`, `diff_ratio=0.0` |
| 완전 상이 프레임 | 흰색 vs 검정 이미지 | `is_stable=False`, `diff_ratio>0.9` |
| 부분 변경 (임계값 이내) | 1% 픽셀만 변경 | `is_stable=True` (기본 threshold 2%) |
| 부분 변경 (임계값 초과) | 5% 픽셀 변경 | `is_stable=False` |
| 영역 지정 비교 | 좌측만 변경, region=우측 | `is_stable=True` (관심 영역은 변하지 않음) |
| 노이즈 필터링 | 모든 픽셀 ±5 변경 (threshold=30) | `is_stable=True` (노이즈 수준 변경 무시) |
| timeout 동작 | 매 프레임 다른 이미지 | `None` 반환, 비교 이력 포함 |

```python
# 테스트 예시
def test_identical_frames_are_stable():
    """동일 프레임 비교 시 안정으로 판단."""
    img = Image.new("RGB", (100, 100), color="white")
    result = compare_frames(img, img)
    assert result.is_stable is True
    assert result.diff_ratio == 0.0

def test_region_excludes_dynamic_area():
    """관심 영역 외부의 변경은 무시."""
    img_a = Image.new("RGB", (200, 200), color="white")
    img_b = img_a.copy()
    # 좌측 100px만 검정으로 변경
    for x in range(100):
        for y in range(200):
            img_b.putpixel((x, y), (0, 0, 0))
    # region=(100, 0, 100, 200) = 우측만 비교 → 변경 없음
    result = compare_frames(img_a, img_b, region=(100, 0, 100, 200))
    assert result.is_stable is True
```

#### `screen_zones.py` 테스트

| 테스트 케이스 | 입력 | 기대 결과 |
|-------------|------|-----------|
| 정적 영역 내 좌표 | (50, 10) in 메뉴바 | `is_safe_for_click=True` |
| 위험 영역 내 좌표 | (500, 300) in 프로브 모니터 | `is_safe_for_click=False`, `is_danger_zone=True` |
| 영역 외부 좌표 | 어느 영역에도 미해당 | `is_safe_for_click=True` (경고 로그) |
| 위험 영역 + override | 위험 영역 + allow override | `is_safe_for_click=True` |
| 영역 생성 비율 검증 | 1920x1080 창 | 각 영역의 절대 좌표가 비율에 맞는지 |

#### `prompts/probe_monitor.py` 테스트

| 테스트 케이스 | 검증 항목 |
|-------------|-----------|
| 반환 타입 | `(str, str)` 튜플 |
| system 메시지 | 이미지 해상도 포함 |
| user 메시지 | 타겟 요소 설명 포함 |
| extra_instructions 반영 | OCR 힌트가 user 메시지에 삽입 |
| JSON 출력 형식 | 응답 형식 예시에 target_keys 포함 |

### Windows (사무실) — 통합 테스트

실제 장비 화면에서 실행하며, Claude Code에서는 실행 불가.

#### Phase 3-A: 영역 보정 (Tier 1 준비)

```
1. 도구 화면 스크린샷 캡처
2. save_zone_overlay() 실행
3. 디버그 이미지에서 영역 경계선 확인
   → 프로브 모니터 영역이 실제 SEM 표시 영역과 일치하는가?
   → 메뉴바/툴바 영역이 실제 버튼 위치를 포함하는가?
4. 비율 조정 → 재실행 → 반복
5. 확정된 비율을 환경변수에 기록
```

**성공 기준**: 디버그 이미지에서 모든 영역 경계가 실제 UI 요소와 ±5px 이내 일치

#### Phase 3-B: 안정성 검출 검증 (Tier 1)

```
1. 장비가 자동 측정 실행 중인 상태에서:
2. wait_for_stability(region=정적_영역) 실행
   → 정적 영역은 안정으로 판단되는가? (기대: True)
3. wait_for_stability(region=프로브_모니터) 실행
   → SEM 영역은 불안정으로 판단되는가? (기대: 측정 중이면 True)
4. threshold/stability_ratio 파라미터 튜닝
```

**성공 기준**: 정적 영역은 1초 이내 안정 판정, SEM 영역은 측정 중 불안정 판정

#### Phase 3-C: 상태 모니터링 (Tier 1)

```
1. continuous_monitor.py를 SAFE_MODE=true로 실행
2. 장비가 측정 중 → idle → 에러 등 상태 전이 관찰
3. 콘솔 출력 확인:
   [STATE] unknown → measuring (confidence=0.88)
   [STATE] measuring → idle (confidence=0.92)
   [STATE] idle → error (confidence=0.79)
4. 상태 분류가 실제와 일치하는지 확인
```

**성공 기준**: 주요 상태 전이 (measuring ↔ idle, error 감지)를 80% 이상 정확하게 감지

#### Phase 3-D: 에러 복구 (Tier 2)

```
1. auto_act=True로 연속 모니터링 실행
2. 의도적으로 에러 상태 유발 (또는 자연 발생 대기)
3. VLM이 에러를 감지하고 retry/skip 버튼 좌표를 추출하는지 확인
4. 좌표가 정적 영역 내인지 영역 검증 통과하는지 확인
5. 실제 버튼 클릭이 에러 복구로 이어지는지 확인
```

**성공 기준**: 에러 감지 후 올바른 복구 버튼을 클릭하여 정상 상태로 복귀

#### Phase 3-E: 프로브 재위치 (Tier 3)

```
1. SAFE_MODE=true로 automate_probe_move.py 실행
2. 디버그 이미지 확인:
   → probe_current_position 마커가 실제 프로브 위치와 일치?
   → probe_target_position이 의도한 목표와 일치?
   → probe_monitor_boundary가 SEM 영역을 정확히 포괄?
3. 여러 번 반복 실행하여 좌표 일관성 확인
4. 충분히 신뢰 확보 후:
5. SAFE_MODE=false로 실행 — 실제 프로브 이동
6. before/after 이미지로 이동 정확도 검증
```

**성공 기준**:
- SAFE_MODE 단계: 좌표가 5회 연속 ±10px 이내 일관성
- 실제 이동 단계: 프로브가 목표 지점 ±10px 이내 도달

---

## 기존 Phase 2와의 관계

| 측면 | Phase 2 (`poc/work2/`) | Phase 3 확장 |
|------|----------------------|--------------|
| 화면 특성 | 정적 (로그인, 메뉴, 탭) | 동적 (실시간 SEM, 연속 갱신) |
| 캡처 방식 | 단발성 (1회 캡처) | 안정성 기반 (연속 캡처 + 비교) |
| VLM 분석 | capture → analyze → act | capture → stabilize → analyze → verify → act |
| 클릭 종류 | 단일 클릭만 | 단일 + 더블클릭 |
| 안전 장치 | SAFE_MODE 1단 | SAFE_MODE + 영역 검증 + 파라미터 검증 3단 |
| 상태 추적 | 없음 (단발성) | 연속 모니터링 + 상태 전이 이력 |
| 프로브 제어 | 없음 | 더블클릭 기반 프로브 이동 + 이동 검증 |

Phase 3 코드는 `poc/work2/` 디렉토리 내에 추가되며, 기존 모듈 (`vlm_screen_analysis.py`, `rcs_utils.py`)을 확장한다. Phase 2의 기존 기능은 **하위 호환**을 유지한다 — 기존 스크립트는 새 파라미터 없이 동일하게 동작.

---

## 리스크 요약

| 리스크 | 완화 방법 |
|--------|----------|
| 프로브 오이동 → 웨이퍼 손상 | 3중 안전 장치, SAFE_MODE 기본 활성화 |
| VLM 좌표 환각 (hallucination) | 조작 후 재캡처 + VLM 재분석으로 검증 |
| SEM 200-500ms 갱신으로 stale frame | stability_region으로 정적/동적 영역 분리 |
| 영역 비율 오보정 | save_zone_overlay() 디버그 이미지 + 환경변수 미세조정 |
| VLM 지연 (500-2000ms) 동안 화면 변경 | verify 단계에서 diff_ratio 확인, 초과 시 재시도 |
| pynput 더블클릭 타이밍 | Windows 시스템 더블클릭 속도 설정 자동 적용 |
