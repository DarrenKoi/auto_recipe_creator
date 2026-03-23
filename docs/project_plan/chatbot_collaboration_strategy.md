# 챗봇 협업 전략 — 엔지니어-AI 대화형 CD-SEM 자동화

> Phase 3 (UI 개발) 착수를 위한 챗봇 아키텍처 및 접근 전략 문서
>
> 작성일: 2026-02-20 | 관련: `docs/project_plan/project_planning_ppt.md` Slide 9 (Supervised Autonomy), Slide 10 (Phase 3 UI)

---

## 1. 현재 상태 평가

### 구현 완료된 기능

| 모듈 | 파일 | 상태 | 비고 |
|------|------|------|------|
| 화면 캡처 | `vlm_click_demo.py`, `screen_capture.py` | ✅ 완료 | mss 기반, DPI/멀티모니터 보정 포함 |
| VLM 화면 분석 | `vlm_screen_analysis.py` | ✅ 완료 | 상태 인식, 측정 판단, 자유 질의 지원 |
| Observe-Think-Act 루프 | `vlm_rcs_agent.py` | ✅ 완료 | 최대 30스텝, 히스토리 3턴, 토큰 버킷 레이트 리미터 |
| 마우스/키보드 제어 | `mouse_control.py`, `keyboard_control.py` | ✅ 완료 | pynput 기반, safe_mode 기본값 |
| RCS 로그인 자동화 | `automate_rcs_login.py`, `rcs_launcher.py` | ✅ 완료 | pywinauto UIA/Win32, 재시도 로직 |
| 좌표 변환 체인 | `vlm_click_demo.py` | ✅ 완료 | VLM→물리→논리→절대 좌표 4단계 |
| OpenSearch 벡터 검색 | `opensearch_handler.py` | ⏸ 비활성 | import-guard, 재활성화 대기 중 |
| 설정 관리 | `config.py` (PocConfig) | ✅ 완료 | .env 기반 통합 설정 |

### 챗봇에 필요하지만 미구현인 기능

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| **대화 세션 관리** | 멀티턴 메시지 히스토리, 세션 ID, 컨텍스트 윈도우 관리 | 높음 |
| **Human-in-the-Loop 체크포인트** | 비가역 작업 전 확인 게이트, 신뢰도 기반 에스컬레이션 | 높음 |
| **Tool-Use (Function Calling) 구조** | 모놀리식 에이전트 루프 → 개별 함수 도구 분리 | 높음 |
| **채팅 UI** | 웹 기반 대화 인터페이스 + 화면 미러링 | 중간 |
| **스트리밍 출력** | 토큰 단위 점진적 응답 표시 | 중간 |
| **레시피 도메인 지식** | 파라미터 스키마, 유효 범위, 장비별 제약 조건 | 중간 |
| **세션 메모리** | 장기 작업 이력 저장, 이전 세션 참조 | 낮음 |

---

## 2. 인터랙션 모델 설계

### 역할 분담: 엔지니어 vs. AI

```
┌─────────────────────────────────────────────────────────┐
│                  엔지니어 (주도권)                         │
│  • 작업 목표 지시 ("Pattern A의 CD 측정 레시피 만들어줘")     │
│  • 비가역 작업 승인 (레시피 저장, 측정 실행)                  │
│  • 이상 상황 판단 및 개입                                  │
│  • 모드 전환 (AUTO ↔ SUPERVISED ↔ MANUAL)                │
└────────────────────────┬────────────────────────────────┘
                         │ 자연어 대화
┌────────────────────────┼────────────────────────────────┐
│                   AI Agent (실행)                         │
│  • 화면 상태 인식 및 보고                                  │
│  • 다음 행동 제안 + 이유 설명                              │
│  • 승인된 행동 자동 실행                                   │
│  • 실패 시 대안 제시 + 엔지니어 판단 요청                    │
└─────────────────────────────────────────────────────────┘
```

### 대화 흐름 예시: 레시피 셋업

```
엔지니어: "Lot ABC의 Pattern X에 대한 CD 측정 레시피를 셋업해줘"

AI: 현재 RCS 화면을 확인했습니다.
    📍 상태: Recipe Editor 메인 화면
    📋 계획:
     1. Recipe 이름 입력 (ABC_PatternX_CD)
     2. 측정 타입 선택 (CD Measurement)
     3. 패턴 좌표 설정
     4. 측정 파라미터 조정

    ▶ Step 1을 진행할까요? [진행] [수정] [취소]

엔지니어: "진행"

AI: ✅ Recipe 이름 입력 완료: ABC_PatternX_CD
    다음: 측정 타입을 CD Measurement로 설정합니다.
    ▶ 진행할까요?

엔지니어: "측정 타입은 Line Width로 해줘"

AI: 알겠습니다. 측정 타입을 Line Width로 변경합니다.
    ⚠️ Line Width 측정은 추가 파라미터(Threshold, Edge Detection Method)가 필요합니다.
    기본값을 사용할까요, 아니면 직접 설정하시겠습니까?

엔지니어: "기본값으로"

AI: ✅ 측정 타입: Line Width (기본 파라미터) 설정 완료
    [스크린샷 첨부 — 현재 화면 상태]
    다음: 패턴 좌표를 설정합니다. ...
```

### 대화 흐름 예시: 트러블슈팅

```
AI: ⚠️ 측정 실행 결과, 위치 오프셋이 감지되었습니다.
    📊 판단: position_offset (신뢰도 82%)
    📐 제안: X 방향 +3.2μm 보정

    [측정 결과 스크린샷 첨부]

    ▶ 제안대로 보정할까요? [적용] [직접 조정] [무시]

엔지니어: "적용"

AI: ✅ X 방향 +3.2μm 보정 완료. 재측정을 시작합니다.
    ...
    ✅ 재측정 성공 — CD = 28.3nm (허용 범위 내)
```

---

## 3. 아키텍처: Tool-Use 패턴으로의 전환

### 현재 구조 (모놀리식 에이전트 루프)

```
VLMRCSAgent.run(task)
  └─ while not done:
       capture → ask_vlm(screenshot + task + history) → execute_action
```

현재 `VLMRCSAgent`는 하나의 프롬프트에 모든 판단을 위임한다. 챗봇에서는 이를 **개별 도구(Tool)**로 분리하여 LLM이 상황에 맞는 도구를 선택하도록 변경한다.

### 제안 구조 (Tool-Use / Function Calling)

```python
# LLM에 등록할 도구 목록
tools = [
    {
        "name": "capture_screen",
        "description": "현재 RCS 화면을 캡처하여 이미지를 반환합니다.",
        "parameters": {
            "region": "(선택) 특정 영역만 캡처 — x,y,width,height"
        }
    },
    {
        "name": "analyze_screen",
        "description": "캡처된 화면의 현재 상태를 분석합니다 (어떤 창/메뉴가 열려있는지).",
        "parameters": {
            "task": "state_recognition | measurement_judgment | general_query",
            "question": "(general_query일 때) 화면에 대한 질문"
        }
    },
    {
        "name": "click_at",
        "description": "화면의 지정된 좌표 또는 UI 요소를 클릭합니다.",
        "parameters": {
            "target": "클릭할 대상 설명 (예: 'Recipe Name 입력 필드')",
            "x": "X 좌표 (픽셀)",
            "y": "Y 좌표 (픽셀)",
            "double_click": "더블 클릭 여부"
        }
    },
    {
        "name": "type_text",
        "description": "키보드로 텍스트를 입력합니다.",
        "parameters": {
            "text": "입력할 텍스트",
            "press_enter": "입력 후 Enter 키를 누를지 여부"
        }
    },
    {
        "name": "press_hotkey",
        "description": "단축키 조합을 실행합니다.",
        "parameters": {
            "keys": "키 조합 (예: 'ctrl+s', 'alt+f4')"
        }
    },
    {
        "name": "scroll",
        "description": "마우스 스크롤을 실행합니다.",
        "parameters": {
            "direction": "up | down",
            "amount": "스크롤 양"
        }
    },
    {
        "name": "judge_measurement",
        "description": "측정 결과 화면을 분석하여 성공/실패를 판단하고 보정 방향을 제안합니다.",
        "parameters": {}
    },
    {
        "name": "search_similar_workflow",
        "description": "현재 상황과 유사한 과거 워크플로우를 RAG에서 검색합니다.",
        "parameters": {
            "query": "검색 질의"
        }
    },
    {
        "name": "wait",
        "description": "지정된 시간만큼 대기합니다 (화면 로딩 등).",
        "parameters": {
            "seconds": "대기 시간 (초)"
        }
    }
]
```

### 기존 코드와의 매핑

| Tool | 기존 코드 매핑 | 수정 사항 |
|------|--------------|----------|
| `capture_screen` | `vlm_click_demo.capture_screen()` | 반환값을 base64 이미지로 래핑 |
| `analyze_screen` | `VLMScreenAnalyzer.analyze_screen()` | 그대로 재사용 |
| `click_at` | `VLMRCSAgent._execute_action()` → click 분기 | 좌표 변환 로직 분리 |
| `type_text` | `VLMRCSAgent._execute_action()` → type 분기 | 그대로 분리 |
| `press_hotkey` | `VLMRCSAgent._execute_action()` → hotkey 분기 | 그대로 분리 |
| `scroll` | `VLMRCSAgent._execute_action()` → scroll 분기 | 그대로 분리 |
| `judge_measurement` | `VLMScreenAnalyzer.judge_measurement()` | 그대로 재사용 |
| `search_similar_workflow` | `OpenSearchHandler.retrieve_context()` | 활성화 필요 |
| `wait` | `time.sleep()` | 신규 래핑 |

### Tool-Use 실행 흐름

```
엔지니어 메시지
    ↓
LLM (대화 컨텍스트 + 도구 정의 + 시스템 프롬프트)
    ↓
┌─ tool_call: capture_screen() ──→ 스크린샷 반환
│   ↓
├─ tool_call: analyze_screen(task="state_recognition") ──→ 상태 분석 결과
│   ↓
├─ LLM 판단: "Recipe Name 필드를 클릭해야 합니다"
│   ↓
├─ ⚠️ CHECKPOINT: 엔지니어 승인 필요 (click 액션)
│   ↓ [승인]
├─ tool_call: click_at(target="Recipe Name 필드", x=450, y=320)
│   ↓
├─ tool_call: type_text(text="ABC_PatternX_CD", press_enter=false)
│   ↓
└─ LLM 응답: "✅ Recipe 이름을 입력했습니다. 다음 단계는..."
    ↓
엔지니어에게 표시
```

---

## 4. Human-in-the-Loop 체크포인트

### 체크포인트 분류

| 레벨 | 조건 | AI 동작 | 엔지니어 동작 |
|------|------|---------|-------------|
| **🟢 자동 실행** | 화면 캡처, 상태 분석, RAG 검색, 대기 | 바로 실행, 결과 보고 | 로그에서 확인 가능 |
| **🟡 알림 후 실행** | 클릭, 텍스트 입력, 스크롤 (SUPERVISED 모드) | 실행 예정 알림 → 3초 대기 → 실행 | 3초 내 취소 가능 |
| **🔴 승인 필수** | 레시피 저장(Ctrl+S), 측정 실행, 삭제, 모드 변경 | 실행 대기 + 승인 요청 | 명시적 승인/거부 |

### 자동 에스컬레이션 규칙

```python
ESCALATION_RULES = {
    # 신뢰도 기반
    "low_confidence": {
        "condition": "VLM confidence < 0.7",
        "action": "SUPERVISED → 승인 필수로 격상",
        "message": "⚠️ AI 판단 신뢰도가 낮습니다 ({confidence:.0%}). 확인해주세요."
    },
    # 연속 실패 기반
    "repeated_failure": {
        "condition": "같은 액션 3회 연속 실패",
        "action": "자동 중단 + MANUAL 모드 전환 제안",
        "message": "🛑 동일 작업이 3회 실패했습니다. 직접 확인이 필요합니다."
    },
    # 화면 변화 없음
    "no_change": {
        "condition": "액션 실행 후 화면 변화 없음 (2회 연속)",
        "action": "대안 전략 제시 + 엔지니어 판단 요청",
        "message": "❓ 실행했으나 화면에 변화가 없습니다. 다른 접근이 필요할 수 있습니다."
    },
    # 예상 외 화면
    "unexpected_state": {
        "condition": "에러 다이얼로그 감지 또는 알 수 없는 화면 상태",
        "action": "즉시 중단 + 스크린샷 첨부 보고",
        "message": "🚨 예상하지 못한 화면 상태입니다. [스크린샷 첨부]"
    }
}
```

### 모드별 체크포인트 동작

| 동작 유형 | AUTO | SUPERVISED | MANUAL |
|-----------|------|-----------|--------|
| capture_screen | 자동 | 자동 | 자동 |
| analyze_screen | 자동 | 자동 | 자동 |
| click_at | 자동 | 알림 후 실행 | 제안만 |
| type_text | 자동 | 알림 후 실행 | 제안만 |
| 레시피 저장 | 승인 필수 | 승인 필수 | 제안만 |
| 측정 실행 | 승인 필수 | 승인 필수 | 제안만 |
| search_similar | 자동 | 자동 | 자동 |

---

## 5. 대화 백엔드 설계

### 세션 관리자 구조

```python
@dataclass
class ChatSession:
    """단일 대화 세션"""
    session_id: str                    # UUID
    created_at: datetime               # 세션 시작 시각
    mode: str                          # AUTO | SUPERVISED | MANUAL
    messages: List[Dict]               # 대화 히스토리
    tool_results: List[Dict]           # 도구 실행 결과 이력
    current_task: Optional[str]        # 현재 진행 중인 작업 설명
    screenshot_cache: Optional[bytes]  # 최근 스크린샷 캐시

class ConversationManager:
    """멀티턴 대화 관리자 — VLMRCSAgent를 래핑"""

    def __init__(self, config: PocConfig):
        self.config = config
        self.analyzer = VLMScreenAnalyzer(...)
        self.sessions: Dict[str, ChatSession] = {}
        self.tool_registry = ToolRegistry()  # Tool-Use 도구 등록소

    def create_session(self, mode="SUPERVISED") -> str:
        """새 대화 세션 생성"""
        ...

    def process_message(self, session_id: str, user_message: str) -> AsyncGenerator:
        """
        사용자 메시지 처리 — 스트리밍 응답 반환

        1. 메시지를 히스토리에 추가
        2. 시스템 프롬프트 + 히스토리 + 도구 정의를 LLM에 전송
        3. LLM 응답 스트리밍 (텍스트 또는 tool_call)
        4. tool_call이면 → 체크포인트 확인 → 실행 → 결과를 LLM에 재전송
        5. 최종 응답을 히스토리에 저장
        """
        ...

    def approve_action(self, session_id: str, action_id: str) -> Dict:
        """체크포인트 승인"""
        ...

    def reject_action(self, session_id: str, action_id: str, reason: str) -> Dict:
        """체크포인트 거부 (사유 포함)"""
        ...

    def switch_mode(self, session_id: str, new_mode: str) -> None:
        """제어 모드 전환"""
        ...

    def get_context_window(self, session_id: str) -> List[Dict]:
        """
        컨텍스트 윈도우 관리:
        - 최근 N턴의 대화만 유지 (기본 20턴)
        - 스크린샷은 최근 3장만 포함
        - 오래된 tool_result는 요약으로 압축
        """
        ...
```

### 시스템 프롬프트 구조

```
당신은 CD-SEM/VeritySEM 장비의 Recipe 셋업을 돕는 AI 어시스턴트입니다.

## 역할
- 엔지니어의 지시에 따라 RCS GUI를 조작합니다
- 화면 상태를 분석하고 다음 행동을 제안합니다
- 모든 비가역 작업은 엔지니어의 승인을 받습니다

## 현재 상태
- 제어 모드: {mode}
- 현재 화면 상태: {last_screen_state}
- 진행 중인 작업: {current_task}

## 사용 가능한 도구
{tool_definitions}

## 행동 규칙
1. 화면 조작 전에 반드시 capture_screen + analyze_screen을 실행하세요
2. 신뢰도 70% 미만이면 엔지니어에게 확인을 요청하세요
3. 레시피 저장, 측정 실행 등 비가역 작업은 반드시 승인을 받으세요
4. 실패 시 동일 행동을 3회 이상 반복하지 마세요
5. 응답은 한국어로, 간결하게 작성하세요
```

### 컨텍스트 윈도우 관리 전략

| 항목 | 전략 | 근거 |
|------|------|------|
| 대화 턴 수 | 최근 20턴 유지 | Qwen3-VL 32K 컨텍스트 기준 |
| 스크린샷 | 최근 3장만 (WebP 85%) | 이미지가 토큰의 대부분을 차지 |
| tool_result | 10턴 이후 → 1줄 요약으로 압축 | "✅ click_at(450,320) 성공" |
| 세션 메모리 | 작업 완료 시 요약을 세션 시작에 주입 | "이전 세션: Pattern A 레시피 완성 (5단계)" |

---

## 6. 채팅 UI 옵션 비교

### 프레임워크 비교표

| 기준 | **Gradio** | **Streamlit** | **FastAPI + React** |
|------|-----------|-------------|-------------------|
| 개발 속도 | ⭐⭐⭐ 매우 빠름 | ⭐⭐⭐ 빠름 | ⭐ 느림 |
| 챗봇 UI 지원 | `gr.ChatInterface` 내장 | `st.chat_message` 내장 | 직접 구현 |
| 스트리밍 | ✅ 네이티브 지원 | ✅ 지원 (제한적) | ✅ SSE/WebSocket |
| 이미지 표시 | ✅ 인라인 이미지 | ✅ 인라인 이미지 | ✅ 자유도 높음 |
| 버튼/승인 UI | ✅ `gr.Button`, 커스텀 컴포넌트 | ⚠️ 제한적 (리렌더링 이슈) | ✅ 완전한 자유도 |
| 실시간 업데이트 | ✅ WebSocket 기반 | ⚠️ polling 기반 | ✅ WebSocket |
| 사내 배포 | ✅ 단일 Python 프로세스 | ✅ 단일 Python 프로세스 | ⚠️ 프론트+백엔드 분리 배포 |
| 프로덕션 확장성 | ⚠️ 중간 | ⚠️ 낮음 | ⭐⭐⭐ 높음 |
| 화면 미러링 | ⚠️ 이미지 갱신으로 구현 | ⚠️ 이미지 갱신으로 구현 | ✅ Canvas/WebRTC 가능 |

### 권장: Gradio (PoC) → FastAPI + React (프로덕션)

**PoC 단계 (Phase 3 전반): Gradio**
- `gr.ChatInterface`로 1일 내 채팅 UI 프로토타입 가능
- 스크린샷을 채팅 메시지에 인라인 표시
- 승인 버튼을 `gr.Button`으로 구현
- Python-only → 프론트엔드 빌드 불필요
- 사내 네트워크에서 `gradio.launch(server_name="0.0.0.0")` 배포

**프로덕션 전환 (Phase 4 이후): FastAPI + React**
- Phase 4 파일럿에서 엔지니어 피드백 반영 후 전환 검토
- WebSocket 기반 실시간 화면 미러링
- 기획 보고서 Slide 6 아키텍처의 Vue 프론트엔드와 정합
- 다중 세션, 인증, 권한 관리

### Gradio PoC 구조 예시

```python
import gradio as gr

def respond(message, history, mode):
    """챗봇 응답 — ConversationManager.process_message() 래핑"""
    session = manager.get_or_create_session(mode=mode)

    for chunk in manager.process_message(session.session_id, message):
        if chunk["type"] == "text":
            yield chunk["content"]
        elif chunk["type"] == "screenshot":
            yield gr.Image(value=chunk["image"])
        elif chunk["type"] == "checkpoint":
            yield f"⚠️ 승인 필요: {chunk['description']}\n[승인] [거부]"

with gr.Blocks(title="CD-SEM Recipe Assistant") as demo:
    gr.Markdown("# CD-SEM Recipe 셋업 어시스턴트")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=respond,
                additional_inputs=[
                    gr.Radio(
                        ["AUTO", "SUPERVISED", "MANUAL"],
                        value="SUPERVISED",
                        label="제어 모드"
                    )
                ]
            )
        with gr.Column(scale=1):
            screen_view = gr.Image(label="현재 RCS 화면")
            refresh_btn = gr.Button("화면 새로고침")

            with gr.Row():
                approve_btn = gr.Button("✅ 승인", variant="primary")
                reject_btn = gr.Button("❌ 거부", variant="stop")

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 7. 레시피 도메인 지식

### 구조화된 파라미터 스키마 (예시)

챗봇이 레시피 파라미터를 이해하고 유효성을 검증하려면, 도메인 지식을 구조화된 스키마로 정의해야 한다.

```python
RECIPE_PARAMETER_SCHEMA = {
    "recipe_name": {
        "type": "string",
        "pattern": r"^[A-Za-z0-9_\-]+$",
        "max_length": 64,
        "description": "레시피 식별 이름"
    },
    "measurement_type": {
        "type": "enum",
        "values": ["CD", "Line Width", "Space Width", "Pitch", "Edge Roughness"],
        "description": "측정 유형"
    },
    "scan_speed": {
        "type": "enum",
        "values": ["Fast", "Normal", "Fine"],
        "default": "Normal",
        "description": "스캔 속도 — Fine은 정밀도↑ 시간↑"
    },
    "magnification": {
        "type": "integer",
        "range": [1000, 500000],
        "unit": "X",
        "description": "배율"
    },
    "tilt_angle": {
        "type": "float",
        "range": [0.0, 60.0],
        "unit": "degrees",
        "default": 0.0,
        "description": "틸트 각도"
    },
    "pattern_coordinates": {
        "type": "object",
        "properties": {
            "x": {"type": "float", "unit": "μm"},
            "y": {"type": "float", "unit": "μm"}
        },
        "description": "측정 대상 패턴의 웨이퍼 좌표"
    }
}
```

### OpenSearch 연동 (RAG)

`opensearch_handler.py`가 이미 구축되어 있으며, 다음 기능을 챗봇에 연결할 수 있다:

| 기능 | 메서드 | 챗봇 활용 |
|------|--------|----------|
| 유사 워크플로우 검색 | `retrieve_context()` | "이 패턴과 비슷한 레시피가 있나요?" |
| 벡터 유사도 검색 | `search_similar()` | 현재 화면과 유사한 과거 작업 장면 검색 |
| 텍스트 검색 | `search_text()` | "CD 측정 실패 사례" 등 키워드 검색 |
| 워크플로우 저장 | `store_workflow()` | 완료된 레시피 셋업 과정을 지식으로 축적 |

**활성화 조건**: `opensearch-py` 패키지를 `requirements.txt`에 추가하고, OpenSearch 서버 접속 정보를 `.env`에 설정

---

## 8. 구현 단계 (Phase 3 세부화)

기획 보고서 Phase 3 (11~14주, UI 개발)을 챗봇 관점에서 세부화한다.

### Phase 3-A: 기반 구축 (11~12주)

| 작업 | 산출물 | 의존성 |
|------|--------|--------|
| Tool-Use 구조 리팩토링 | `tools/` 패키지 (개별 도구 모듈) | 기존 `vlm_rcs_agent.py` |
| ConversationManager 구현 | `conversation_manager.py` | PocConfig, VLMScreenAnalyzer |
| 체크포인트 시스템 구현 | `checkpoint.py` | ConversationManager |
| 시스템 프롬프트 작성 | `prompts/system_prompt.txt` | 레시피 파라미터 스키마 |

### Phase 3-B: UI 프로토타입 (12~13주)

| 작업 | 산출물 | 의존성 |
|------|--------|--------|
| Gradio 채팅 UI | `chatbot_ui.py` | ConversationManager |
| 화면 미러링 (이미지 갱신) | Gradio Image 컴포넌트 | capture_screen 도구 |
| 승인/거부 버튼 연동 | Gradio Button + 이벤트 핸들러 | 체크포인트 시스템 |
| 스트리밍 응답 | SSE 기반 점진적 출력 | LLM API 스트리밍 |

### Phase 3-C: 통합 테스트 (13~14주)

| 작업 | 산출물 | 의존성 |
|------|--------|--------|
| 엔드투엔드 시나리오 테스트 | 테스트 스크립트 + 결과 보고서 | Phase 3-A, 3-B |
| 모드 전환 테스트 (AUTO/SUPERVISED/MANUAL) | 모드별 동작 검증 | 체크포인트 시스템 |
| 엔지니어 사용성 피드백 수집 | 피드백 문서 | UI 프로토타입 |
| Safe mode 통합 테스트 | 실제 입력 없이 전체 흐름 검증 | 전체 |

---

## 9. 사전 준비 체크리스트

챗봇 구현 착수 전에 확인/준비가 필요한 항목들:

### 필수 (착수 전 완료)

- [ ] **LLM 백엔드 결정**: 챗봇의 대화 엔진으로 사용할 모델 확정
  - 현재 Qwen3-VL은 화면 분석용 — 대화 + Tool-Use 지원 모델이 별도로 필요할 수 있음
  - 옵션: Qwen3-VL 단일 모델 / 대화용 LLM + 화면용 VLM 분리 / Claude API
  - Tool-Use (Function Calling) 지원 여부 확인 필수
- [ ] **대표 워크플로우 시나리오 3~5개 문서화**
  - 예: "신규 CD 측정 레시피 생성", "기존 레시피 파라미터 수정", "측정 실패 후 보정"
  - 각 시나리오의 단계별 화면 상태 + 조작 내용 포함
- [ ] **레시피 파라미터 목록 확보**
  - 실제 CD-SEM/VeritySEM의 레시피에서 설정하는 파라미터 전체 목록
  - 각 파라미터의 유효 범위, 기본값, 단위, 설명

### 권장 (병행 준비)

- [ ] **RCS 화면 상태 정의 확대** (현재 10종 → 20종 목표)
  - 챗봇이 인식해야 할 화면 상태가 늘어남
  - 에러 다이얼로그, 확인 팝업 등 예외 상태 포함
- [ ] **테스트 환경 확보**
  - 챗봇 개발/테스트용 RCS 접속 가능한 환경
  - Safe mode에서 대부분 테스트 가능하나, 최종 검증은 실제 환경 필요
- [ ] **OpenSearch 서버 구축** (RAG 연동 시)
  - 현재 코드는 준비됨 (`opensearch_handler.py`)
  - 서버 인스턴스 + 인덱스 초기화만 필요
- [ ] **엔지니어 피드백 채널 설정**
  - 프로토타입 시연 후 피드백 수집 방법 결정
  - 사용성 테스트 참여 엔지니어 2~3명 확보

### 기술적 확인 사항

- [ ] **VLM API 스트리밍 지원 여부**: Qwen3-VL API가 `stream=True` 옵션을 지원하는지 확인
- [ ] **동시 요청 제한**: 레이트 리미터 설정 검토 (현재 5 req / 5 sec)
- [ ] **WebSocket 지원**: 자동화 PC에서 WebSocket 서버 운영 가능 여부 (방화벽 등)
- [ ] **Gradio 설치 가능 여부**: 자동화 PC의 Python 환경에서 `pip install gradio` 가능 확인

---

## 부록 A: 용어 정리

| 용어 | 설명 |
|------|------|
| **Tool-Use / Function Calling** | LLM이 대화 중 외부 함수를 호출할 수 있는 기능. LLM이 JSON 형식으로 함수 호출을 요청하면 시스템이 실행 후 결과를 반환 |
| **Human-in-the-Loop (HITL)** | AI 시스템에 사람의 판단/승인 단계를 포함하는 설계 패턴 |
| **Supervised Autonomy** | AI가 주도적으로 작업을 수행하되, 사람이 언제든 개입/중단할 수 있는 제어 방식 |
| **SSE (Server-Sent Events)** | 서버에서 클라이언트로 실시간 단방향 데이터 스트리밍. 채팅 응답의 토큰 단위 출력에 사용 |
| **컨텍스트 윈도우** | LLM이 한 번에 처리할 수 있는 입력 토큰 수 제한. 대화 히스토리 관리의 핵심 제약 |

## 부록 B: 마이그레이션 전 기존 코드 참조 경로

`poc/work` 는 `poc/work2` 전환 완료 후 제거되었으며, 아래 경로는 설계 맥락을 위한 과거 참조용이다.

| 파일 | 용도 | 챗봇 연관성 |
|------|------|-----------|
| `poc/work/vlm_rcs_agent.py` | Observe-Think-Act 루프 | Tool-Use로 분리할 원본 |
| `poc/work/vlm_screen_analysis.py` | 화면 분석 API | analyze_screen, judge_measurement 도구 |
| `poc/work/vlm_click_demo.py` | 좌표 변환 + 시각화 | capture_screen 도구, 화면 미러링 |
| `poc/work/config.py` | 통합 설정 | ConversationManager 설정 확장 |
| `poc/work/opensearch_handler.py` | 벡터/텍스트 검색 | search_similar_workflow 도구 |
| `poc/work/automate_rcs_login.py` | RCS 로그인 자동화 | 세션 시작 시 자동 로그인 |
| `test/vlm_input_control/mouse_control.py` | 마우스 제어 | click_at 도구 |
| `test/vlm_input_control/keyboard_control.py` | 키보드 제어 | type_text, press_hotkey 도구 |
