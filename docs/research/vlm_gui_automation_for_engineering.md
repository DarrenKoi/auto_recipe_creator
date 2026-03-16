# 리서치: 엔지니어링 워크플로우를 위한 VLM 기반 GUI 자동화

## 배경

본 프로젝트는 CD-SEM (VeritySEM/RCS) 레시피 설정을 자동화한다 — 반도체 계측용 Windows 데스크톱 애플리케이션이다. 현재 구현은 스크린샷 → VLM 좌표 추출 → pynput 클릭/타이핑 방식의 2단계 파이프라인(PaddleOCR-VL + ui-venus)을 사용하고 있다. 본 리서치는 **엔지니어링/산업용 애플리케이션**에 특화된 견고한 멀티스텝 워크플로우 자동화 시스템으로의 발전 방향을 탐색한다.

---

## 1. 마우스 & 키보드 제어 — 고려사항

### 현재 접근법 (자체 코드베이스)
- 마우스/키보드 제어에 **pynput** 사용 (pywinauto는 RCS 레거시 컨트롤에서 실패)
- `click_at()`: VLM 좌표 → 화면 좌표 → 2회 재시도 fallback 클릭
- 선형 보간(60fps)을 통한 부드러운 마우스 이동
- 단축키 조합, 텍스트 타이핑, 드래그 작업 지원

### 학계/업계의 권장 기법

| 기법 | 설명 | 관련성 |
|------|------|--------|
| **정규화 좌표 (0-1)** | UI-TARS는 절대 픽셀 대신 `(x/width, y/height)`를 사용. 해상도 간 전이 가능 | 높음 — RCS는 장비별로 다른 해상도에서 실행될 수 있음 |
| **좌표 비의존 그라운딩** | GUI-Actor (Microsoft)는 좌표를 텍스트로 생성하는 대신 attention 기반 patch alignment 사용. 7B 모델로 UI-TARS-72B 능가 | 향후 과제 — 커스텀 모델 학습 필요 |
| **Set-of-Mark (SoM)** | OmniParser가 스크린샷에 번호가 매겨진 bounding box를 오버레이한 후 VLM에 전송. 그라운딩 정확도 70.5% → 93.8% 향상 | **높은 우선순위** — 모델 변경 없이 파이프라인에 추가 가능 |
| **Action space 표준화** | UI-TARS 정의: Click, Drag, Scroll, Type, Hotkey, Wait, Finished, CallUser | 도입 권장 — 자체 액션 타입 표준화 필요 |
| **실행 전 검증** | VeriSafe Agent: 안전 제약을 DSL로 인코딩, 실행 전 액션 검증. 94-98% 정확도 | **엔지니어링에 필수** — 잘못된 클릭 = 웨이퍼 손상 |
| **Readback 검증** | 값 입력 후: 스크린샷 → OCR → 표시된 값이 의도한 값과 일치하는지 확인 | **레시피 파라미터에 필수** |

### 엔지니어링 특화 고려사항
- **밀집된 수치 UI**: VLM은 밀집된 파라미터 필드의 작은 텍스트에서 어려움을 겪음. 자체 OCR 보조 파이프라인은 연구 결과와 일치 (복잡한 레이아웃에서 2단계 > 단일 단계)
- **레거시 Win32 컨트롤**: accessibility tree 없음 → 순수 비전 접근법이 적합. OmniParser V2 (YOLOv8 + Florence-2 + PaddleOCR)는 accessibility API 없이도 인터랙티브 요소 감지 가능
- **정밀한 필드 타겟팅**: 작은 입력 필드의 경우 대상 영역 주변의 **확대 크롭**을 사용하여 정확도 향상 (RegionFocus: ScreenSpot-Pro에서 +28%)

---

## 2. 의사결정 — VLM 에이전트의 행동 결정 방식

### 패러다임

#### a) ReAct (Reason + Act) — 가장 일반적
```
반복:
  1. 관찰: 스크린샷 + OCR
  2. 사고: "로그인 화면이 보인다. Server 필드를 클릭해야 한다."
  3. 행동: click(x=150, y=200)
  4. 관찰: 새 스크린샷
  5. 사고: "Server 필드에 포커스가 맞춰졌다. 서버 주소를 입력해야 한다."
  ...
```
- 사용 사례: UI-TARS, Anthropic Computer Use, 대부분의 GUI 에이전트
- 장점: 유연함, 예상치 못한 상태 처리 가능
- 단점: 매 스텝마다 VLM 호출 필요, 긴 워크플로우에서 느림

#### b) Plan-then-Execute — 알려진 워크플로우에 더 적합
```
1. 계획: "레시피 설정 순서: 로그인 → 레시피 편집기 이동 → 파라미터 설정 → 저장"
2. 각 단계 실행, 필요 시에만 VLM으로 검증
3. 예상치 못한 상태 감지 시 재계획
```
- 사용 사례: Agent-S (Manager + Worker), UFO2 (HostAgent + AppAgent)
- 장점: 빠름, VLM 호출 감소, 예측 가능
- **RCS에 적합**: 레시피 설정은 알려진 워크플로우 — 계획은 고정이고 요소 위치만 변동

#### c) State Machine + 사전 학습된 지식 — 산업용에 최적
```
상태: {login_screen, main_menu, recipe_editor, parameter_dialog, ...}
전이: {login_screen --[login 클릭]--> main_menu, ...}
각 상태에서: 사전 정의된 액션 시퀀스 실행
```
- 사용 사례: **ActionEngine** (95% 성공률, ReAct 대비 11.8배 비용 절감), **InfraMind** (산업용 GUI에서 83%)
- **RCS에 권장** — 레시피 설정은 고정된 상태 그래프를 가짐

### InfraMind — 자체 유즈케이스에 가장 관련된 프레임워크

InfraMind (2025년 9월)은 RCS와 동일한 과제를 가진 산업용 관리 GUI를 다룸:

| InfraMind 과제 | RCS 대응 상황 |
|---|---|
| 커스텀 개발 컨트롤, accessibility 없음 | RCS 레거시 Win32 컨트롤, pywinauto 실패 |
| 데스크톱 앱에 URL 식별자 없음 | 수십 개의 RCS 다이얼로그가 유사하게 보임 |
| 폐쇄망 환경 | 반도체 팹 네트워크는 격리됨 |
| 안전 중요 컨트롤 | 잘못된 레시피 파라미터 = 웨이퍼 손상 |
| 정밀도 + 효율성 요구 | 레시피 설정에 정확한 파라미터 값 필요 |

**InfraMind의 접근법:**
1. **탐색 단계**: 모든 GUI 요소를 BFS/DFS로 순회 (VM 스냅샷으로 안전한 롤백). 완전한 요소 기능 맵 구축
2. **지식 증류**: 대형 모델 (GPT-4o)이 탐색 → 구조화된 지식 베이스 → 소형 7B 모델로 배포
3. **상태 식별**: CLIP 시각 임베딩 + 텍스트 설명의 이중 상태 표현
4. **메모리 기반 계획**: 성공한 액션 플로우 트리를 저장, 재사용
5. **안전**: CLIP 기반 블랙리스트 필터링, 확인 다이얼로그, 리스크 평가

**결과**: OpenDCIM에서 83.3%, 상용 플랫폼에서 76.7% — 범용 에이전트 (UI-TARS: 43.3%/20.0%) 대비 월등히 앞섬

### ActionEngine — State Machine 메모리

ActionEngine (2025년 2월)은 GUI를 state machine M=(S,O,T)으로 모델링:
- **상태(States)**: 구조적 뷰 (상태 폭발 방지를 위해 atom으로 구성)
- **연산(Operations)**: 엣지 (UI 조작 또는 데이터 수집)
- **전이(Transitions)**: 실제 실행으로 검증
- 반자동 크롤러가 그래프 구축 (보통 20-30개 상태, 100-150개 전이)
- **단일 추론 단계**로 워크플로우를 실행 가능한 계획으로 컴파일
- **95% 성공률** vs 베이스라인 66%, 11.8배 비용 절감

---

## 3. 프로세스 추출 — 워크플로우 기록 및 재생

### 핵심 질문
엔지니어의 레시피 설정 과정을 어떻게 캡처하여 재현 가능한 자동화로 변환할 것인가?

### 접근법 A: 사람의 시연 녹화 (ShowUI-Aloha)

**ShowUI-Aloha** (2025년 12월)는 가장 완전한 오픈소스 시스템을 제공:

1. **레코더 앱** (Windows .exe): 30fps 화면 영상 + 마우스/키보드 이벤트를 타임스탬프와 함께 캡처
2. **Raw Log Parser**: 연속 키 입력 병합, 드래그 재구성, 중복 클릭 제거
3. **Screenshot Marker**: 액션별 주석이 달린 스크린샷 생성 (클릭은 빨간 X, 드래그는 폴리라인)
4. **Trace Generator**: VLM이 스텝별 구조화된 JSON 생성:
   ```json
   {
     "observation": "Server, UserID, Password 필드가 있는 로그인 다이얼로그",
     "think": "Server 필드에 서버 주소를 입력해야 함",
     "action": "click(x=150, y=200) then type('SEM-SERVER-01')",
     "expectation": "Server 필드에 'SEM-SERVER-01'이 표시되어야 함"
   }
   ```
5. **일반화**를 위해 설계된 출력 — 단일 시연으로 UI 레이아웃 변형에 대응 가능

### 접근법 B: Record-and-Replay + 적응 (AgentRR)

**AgentRR** (2025년 5월) — 자체 케이스에 가장 실용적:

1. **녹화 단계**: 엔지니어가 레시피 설정을 한 번 수행. 시스템이 캡처:
   - 각 액션의 스크린샷
   - 마우스/키보드 이벤트
   - VLM이 생성한 각 스텝의 추론
2. **경험 추상화** (3단계):
   - **Low**: 정확한 재생 (동일 좌표, 동일 값)
   - **Medium**: 파라미터화된 재생 (동일 워크플로우, 다른 값 — 예: 다른 레시피 파라미터)
   - **High**: 개념적 재생 (동일 목표, UI 변경에 적응)
3. **재생 단계**: Low 레벨에서 시작, 환경 변경 시 상위 레벨로 에스컬레이션
4. 312개의 사람 궤적만으로 기본 모델 대비 **141% 성능 향상** (PC Agent-E)

### 접근법 C: 체계적 탐색 (InfraMind)

사람을 녹화하지 않고 — VLM이 애플리케이션을 체계적으로 탐색:

1. 모든 인터랙티브 요소를 BFS/DFS로 순회
2. 각 액션 전 VM 스냅샷 → 원치 않은 결과 시 롤백
3. 액션 전후 스크린샷 비교로 요소 기능 학습
4. 완전한 상태 그래프 자동 구축
5. **장점**: 사람이 시연한 경로뿐 아니라 모든 경로를 발견

### RCS 권장 전략

**하이브리드: 탐색 + 녹화 + 파라미터화**

```
Phase 1: 탐색 (EXPLORE)
  - VLM이 RCS 화면을 체계적으로 탐색
  - 상태 그래프 구축: {screen_name → [elements, transitions]}
  - 화면별 모든 인터랙티브 요소 식별

Phase 2: 녹화 (RECORD)
  - 엔지니어가 녹화하면서 레시피 설정 한 번 수행
  - ShowUI-Aloha 방식: 영상 + 이벤트 + 스크린샷
  - VLM이 녹화를 구조화된 trace로 변환

Phase 3: 파라미터화 (PARAMETERIZE)
  - trace에서 레시피 파라미터 추출 (서버, 측정 사이트, 임계값)
  - 레시피 템플릿 생성: 고정 워크플로우 + 변수 파라미터
  - 템플릿 형식:
    {
      "workflow": "rcs_recipe_setup_v1",
      "steps": [
        {"state": "login_screen", "action": "type", "target": "server_field", "value": "${recipe.server}"},
        {"state": "login_screen", "action": "click", "target": "login_button"},
        {"state": "main_menu", "action": "click", "target": "recipe_editor"},
        ...
      ],
      "parameters": {
        "server": "SEM-SERVER-01",
        "recipe_name": "LINE_CD_45nm",
        "measurement_sites": [...],
        ...
      }
    }

Phase 4: 재생 (REPLAY)
  - 레시피 템플릿 + 파라미터 값 로드
  - 각 상태 전이마다 VLM 검증을 수행하며 스텝 실행
  - 모든 입력 값에 대해 readback 검증
```

---

## 4. 상태 추적 — VLM에게 "현재 위치"를 알려주는 방법

### 문제
RCS 레시피 설정에는 수십 개의 화면/다이얼로그가 관여한다. VLM은 다음을 알아야 한다:
- 지금 어떤 화면을 보고 있는가?
- 워크플로우의 어떤 단계에 있는가?
- 지금까지 완료된 것은 무엇인가?
- 다음에 무엇을 해야 하는가?

### 접근법 A: 스크린샷 히스토리 (UI-TARS 방식)

```
VLM에 전달되는 컨텍스트:
  - 최근 5개 스크린샷 (FIFO 슬라이딩 윈도우)
  - 전체 사고 + 액션 텍스트 히스토리
  - 현재 스텝 번호: "23개 중 7번째 스텝"
```

**히스토리 관리에 대한 연구 결과:**
- UI-TARS: 최대 5개 스크린샷 + 전체 텍스트 히스토리
- Agent-S: 8개 이미지 턴
- JetBrains Research (2025): **Observation masking** (최근 10턴은 상세 유지, 이전 스크린샷은 텍스트 placeholder로 대체)이 LLM 요약보다 **동등하거나 더 나은 성능** — 비용 50%+ 절감
- AgentProg: 히스토리를 변수가 있는 프로그램으로 재구성, 활성 실행 경로만 유지 (~9k 토큰 vs 베이스라인 17k+)

### 접근법 B: State Machine (RCS에 권장)

```python
# 예상 상태 정의
STATES = {
    "login_screen": {
        "visual_cues": ["Server", "User ID", "Password", "Log In"],
        "clip_embedding": "<사전 계산>",
        "expected_elements": ["server_field", "userid_field", "password_field", "login_button"],
        "transitions": {"login_success": "main_menu", "login_error": "login_screen"}
    },
    "main_menu": {
        "visual_cues": ["View", "List", "Recipe", "Tool"],
        # ...
    },
    "recipe_editor": {
        # ...
    },
}

# 각 스텝에서:
# 1. 스크린샷 캡처
# 2. 현재 상태 식별 (CLIP 유사도 + OCR 키워드 매칭)
# 3. 해당 상태의 예상 액션 조회
# 4. 액션 실행
# 5. 다음 예상 상태로의 전이 검증
```

**이중 상태 식별** (InfraMind):
- **시맨틱**: VLM이 보이는 것을 설명 → 알려진 상태와 매칭
- **시각적**: CLIP 임베딩 유사도로 알려진 상태 스크린샷과 비교
- 결합 스코어링으로 시각적으로 유사한 화면의 오식별 방지

### 접근법 C: 진행 상황 추적 프롬프트

모든 VLM 호출에 워크플로우 컨텍스트 포함:

```
당신은 RCS 소프트웨어에서 CD-SEM 레시피 설정을 자동화하고 있습니다.

워크플로우 진행 상황:
  [x] 스텝 1: RCS 로그인 (완료)
  [x] 스텝 2: 레시피 편집기로 이동 (완료)
  [ ] 스텝 3: 새 레시피 생성 (현재)
  [ ] 스텝 4: 측정 파라미터 설정
  [ ] 스텝 5: 측정 사이트 정의
  [ ] 스텝 6: 레시피 저장

현재 상태: Recipe Editor - 메인 뷰
마지막 액션: 메인 메뉴에서 "Recipe" 탭 클릭
예상: "New Recipe" 버튼이 보여야 함

스크린샷을 기반으로 "New Recipe" 버튼을 식별하고 좌표를 반환하세요.
```

### 권장: B + C 조합

- **state machine**으로 자동 상태 식별 (빠름, 알려진 상태에서 VLM 호출 불필요)
- VLM 지원이 필요할 때 **진행 상황 추적 프롬프트** 사용 (요소 위치, 예상치 못한 상태)
- 상태를 인식할 수 없을 때만 **스크린샷 히스토리**로 fallback

---

## 5. 프로젝트 아키텍처 권장사항

### 현재 vs 제안 아키텍처

```
현재 (스크립트별, 하드코딩된 시퀀스):
  automate_rcs_login.py → click_rcs_view_mode.py → check_tool_screen.py
  각 스크립트: 스크린샷 → VLM → 파싱 → 클릭 (반복)

제안 (state machine 기반 워크플로우 엔진):
  ┌─────────────────────────────────────────────┐
  │           Recipe Workflow Engine             │
  │                                              │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
  │  │  State   │  │ Action   │  │ Verifier  │  │
  │  │Recognizer│  │ Executor │  │           │  │
  │  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
  │       │              │              │         │
  │  ┌────▼──────────────▼──────────────▼─────┐  │
  │  │         State Machine Graph            │  │
  │  │  (탐색/시연에서 사전 학습)              │  │
  │  └────────────────────────────────────────┘  │
  │                                              │
  │  ┌────────────┐  ┌───────────────────────┐   │
  │  │ Recipe     │  │ VLM Pipeline          │   │
  │  │ Template   │  │ (OCR + Primary VLM)   │   │
  │  │ (JSON)     │  │                       │   │
  │  └────────────┘  └───────────────────────┘   │
  └─────────────────────────────────────────────┘
```

### 구축해야 할 핵심 컴포넌트

1. **State Recognizer**: CLIP 임베딩 + OCR 키워드 매칭 → VLM 호출 없이 현재 화면 식별
2. **Action Executor**: 표준화된 액션 타입 (click, type, hotkey, wait, scroll, verify)
3. **Verifier**: 액션 후 스크린샷 → OCR → 예상 결과 확인 (입력 값 readback)
4. **State Machine Graph**: 탐색 또는 녹화로 사전 구축, JSON으로 저장
5. **Recipe Template**: 파라미터화된 워크플로우 정의 (고정 스텝 + 변수 값)
6. **Safety Layer**: 실행 전 제약 조건 검사, 위험 요소 블랙리스트

### 우선순위

| 우선순위 | 항목 | 이유 |
|----------|------|------|
| **P0** | 더 나은 그라운딩을 위한 Set-of-Mark (SoM) 오버레이 | +23% 정확도, 모델 변경 불필요 |
| **P0** | 타이핑 후 readback 검증 | 레시피 파라미터 정확도에 필수 |
| **P1** | State recognizer (CLIP + OCR) | 중복 VLM 호출 제거, state machine 구현 기반 |
| **P1** | 표준화된 액션 타입 + 워크플로우 JSON | 스크립트별 하드코딩 대체 |
| **P2** | 녹화 시스템 (ShowUI-Aloha 방식) | 새 워크플로우를 위한 엔지니어 시연 캡처 |
| **P2** | 실행 전 안전 제약 조건 | 위험한 액션 방지 |
| **P3** | 탐색 크롤러 | 모든 RCS 화면과 요소 자동 발견 |
| **P3** | 소형 모델로의 지식 증류 | 팹 내 프로덕션 배포 |

---

## 6. 핵심 논문 & 도구 레퍼런스

### 필독 (엔지니어링 유즈케이스에 직접 적용 가능)

| 논문/도구 | 핵심 기여 | 링크 |
|---|---|---|
| **InfraMind** (2025년 9월) | 산업용 GUI 프레임워크: 탐색, 안전, 상태 식별, 지식 증류 | arxiv:2509.13704 |
| **ActionEngine** (2025년 2월) | GUI 에이전트를 위한 state machine 메모리, 95% 성공률, 11.8배 비용 절감 | arxiv:2602.20502 |
| **AgentRR** (2025년 5월) | 다단계 경험 추상화를 통한 Record-and-Replay | arxiv:2505.17716 |
| **ShowUI-Aloha** (2025년 12월) | 완전한 시연 녹화 → trace 생성 파이프라인 | github:showlab/ShowUI-Aloha |
| **VeriSafe Agent** (2025년 3월) | DSL 기반 실행 전 안전 검증 | arxiv:2503.18492 |
| **OmniParser V2** (2025년 2월) | 순수 비전 화면 파싱 (YOLOv8 + Florence-2 + PaddleOCR) | github:microsoft/OmniParser |
| **WorldGUI** (2025년 2월) | 데스크톱 GUI 벤치마크, 에이전트 실패 모드 분석 | arxiv:2502.08047 |

### 주요 참고 자료

| 논문/도구 | 핵심 기여 | 링크 |
|---|---|---|
| **UI-TARS 2** (2025) | System-2 추론을 갖춘 SOTA 엔드투엔드 GUI 에이전트 | arxiv:2509.02544 |
| **Agent-S3** (2025) | 2계층 아키텍처, OSWorld에서 사람 능가 | github:simular-ai/Agent-S |
| **UFO2** (2025년 4월) | 듀얼 에이전트 + 하이브리드 인식 (accessibility + vision 융합) | github:microsoft/UFO |
| **GUI-Actor** (2025) | 좌표 비의존 그라운딩, 7B로 UI-TARS-72B 능가 | microsoft.github.io/GUI-Actor |
| **RegionFocus** (2025) | 테스트 타임 스케일링을 위한 visual zoom, +28% 그라운딩 | arxiv:2505.00684 |
| **PC Agent-E** (2025) | 312개 사람 궤적으로 141% 성능 향상 | arxiv:2505.13909 |
| **VAGEN** (2025) | 3단계 검증: static → retrospective → probing | arxiv:2602.00575 |

### 반도체 특화

| 자료 | 관련성 |
|---|---|
| **GUIDE-X** (SPIE 2025) | 반도체 X-ray 검사 가이던스를 위한 VLM+LLM |
| **Canopus AI / Siemens** (2026년 1월) | AI 기반 계측 워크플로우 자동화 |
| **Applied Materials AIx** | 실시간 프로세스 레시피 최적화 |
| **Design-Based Metrology** | 기존 CAD-to-recipe 자동화 (GUI 레벨과는 다른 계층) |

---

## 7. 요약: 프로젝트 핵심 시사점

1. **알려진 워크플로우에서는 State machine > ReAct**: 레시피 설정은 예측 가능하다 — 워크플로우가 고정된 상황에서 추론에 VLM 호출을 낭비하지 않는다. VLM은 요소 위치 파악과 검증에만 사용.

2. **Set-of-Mark이 가장 쉬운 정확도 향상 방법**: 스크린샷에 번호가 매겨진 bounding box를 오버레이한 후 VLM에 전송. OmniParser V2는 순수 비전으로 인터랙티브 요소를 감지 가능.

3. **한 번 녹화, 여러 번 재생**: ShowUI-Aloha 또는 커스텀 녹화로 하나의 시연을 캡처, 파라미터화한 후 다른 레시피 값으로 재생.

4. **이중 상태 식별이 단일보다 우수**: CLIP 임베딩 유사도 + OCR 키워드 매칭은 단독 사용보다 더 견고 (InfraMind 접근법).

5. **엔지니어링에서 readback 검증은 필수**: 모든 입력 값 후 스크린샷 → OCR → 검증. 이것이 산업용 자동화를 소비자 앱 에이전트와 구분 짓는 핵심.

6. **히스토리 관리에서 observation masking > 요약**: 최근 10턴은 상세 유지, 이전 것은 텍스트 placeholder로 대체. 더 단순하고, 저렴하며, LLM 요약보다 종종 더 효과적.

7. **안전 제약 조건은 명시적이어야 함**: 위험한 액션에 대한 DSL 또는 블랙리스트 정의. 엔지니어링 환경에서 VLM의 "상식"에 안전을 의존하지 않는다.

8. **자체 2단계 OCR 파이프라인은 연구에 의해 검증됨**: 학계에서도 밀집/복잡 레이아웃에서 파이프라인 OCR이 엔드투엔드 VLM보다 우수함을 확인. 이 접근법을 유지하고 강화할 것.
