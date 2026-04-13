# Natural Language Command Orchestrator for RCS Workflows

## 1. Problem

Engineers run individual scripts manually (`workflow_login.py`, `workflow_select_tool.py`).
They want to type commands like:

- "login and go to tool MCD018"
- "check the tool MCDC10 if it is align failed or not"
- "login with this id and password and visit server R2"

...and have the system figure out which workflows to chain and execute them.

---

## 2. Why Not LangGraph?

| Factor | LangGraph | Simple Dispatcher |
|--------|-----------|-------------------|
| Dependency weight | Heavy (langgraph + langchain) | Zero new deps |
| Workflow shape | Branching graphs with cycles | Linear sequential chains |
| Step orchestration | Redundant — WorkflowRunner already does this | Builds on top of WorkflowRunner |
| Learning curve | New mental model for team | Matches existing patterns |
| Debugging | Graph trace viewer | Print logs (existing pattern) |

**Verdict**: Our `WorkflowRunner` already handles the hard part (step-level conditions, retries, dependencies).
What's missing is a thin layer above: **NLU → workflow selection → sequential chaining**.
This is a linear pipeline, not a graph. LangGraph is overkill.

---

## 3. Architecture

```
Engineer's command (text)
       │
       ▼
 CommandOrchestrator          ← NEW entry point
       │
       ├─ IntentParser        ← NEW: kimi-k2.5 LLM + regex fallback
       │
       ├─ WorkflowPlanner     ← NEW: intent → workflow chain (registry)
       │
       └─ ChainExecutor       ← NEW: runs chain with shared state
              │
              ▼
       WorkflowRunner         ← EXISTING (no change)
              │
              ▼
       workflow_login.py      ← EXISTING
       workflow_select_tool.py
       (future workflows...)
```

### Key Principle

Each layer does ONE thing:

| Layer | Responsibility |
|-------|---------------|
| IntentParser | "What does the engineer want?" |
| WorkflowPlanner | "Which workflows, in what order?" |
| ChainExecutor | "Run them, bridge state between them" |
| WorkflowRunner | "Execute steps within one workflow" |

---

## 4. Module Design

### 4.1 `command_types.py` — Shared Dataclasses

```python
@dataclass
class ParsedIntent:
    """자연어 명령에서 파싱된 의도."""
    actions: list[str]          # ["login", "select_tool"]
    tool_name: str | None       # "MCD018"
    server_name: str | None     # "R2"
    user_id: str | None
    password: str | None
    extra_query: str | None     # "check if align failed"
    raw_command: str

@dataclass
class CommandPlan:
    """실행할 워크플로 체인 계획."""
    plan_id: str
    workflow_chain: list[str]   # ["login", "select_tool"]
    parameters: dict            # {"tool_name": "MCD018", "server_name": "R2"}

@dataclass
class CommandResult:
    """명령 실행 전체 결과."""
    plan_id: str
    status: str                 # "completed" | "partial" | "failed"
    workflow_results: list      # list of WorkflowRun objects
    summary_message: str        # Korean result for engineer
    elapsed_ms: float
```

### 4.2 `intent_parser.py` — NLU (Natural Language Understanding)

**Primary path**: Send command to a **configurable text LLM** with a structured prompt that forces JSON output. No vision needed — any text-capable model works.

- NLU model is resolved from `SHARED_PIPELINE_SETTINGS["nlu_service"]` or env var `NLU_SERVICE`
- Works with any registered service: kimi-k2.5, qwen3-vl-30b, or future text models
- The service just needs an OpenAI-compatible `/v1/chat/completions` endpoint

```
System: You are a command parser for an RCS automation system.
Available actions: login, select_tool, check_alignment, check_measurement
Output JSON: {"actions": [...], "tool_name": "...", "server_name": "..."}

User: login and go to tool MCD018
→ {"actions": ["login", "select_tool"], "tool_name": "MCD018"}
```

**Fallback path**: Regex extraction when LLM is unavailable.

```python
# Tool name: uppercase+digits pattern
TOOL_PATTERN = r'[A-Z0-9]{4,}'          # MCD018, 6MCD2201, MCDC10

# Server: R1, R2, server R1
SERVER_PATTERN = r'(?:server\s*)?[Rr]\d+'

# Action keywords
ACTION_MAP = {
    "login": ["login", "로그인", "sign in"],
    "select_tool": ["tool", "go to", "visit", "select"],
    "check_alignment": ["align", "alignment", "정렬"],
    "check_measurement": ["measure", "measurement", "측정"],
}
```

### 4.3 `workflow_planner.py` — Registry + Dependency Resolution

```python
WORKFLOW_REGISTRY = {
    "login": WorkflowDef(
        name="login",
        runner_fn=run_login_workflow,
        provides=["logged_in", "main_window"],
        requires=[],
    ),
    "select_tool": WorkflowDef(
        name="select_tool",
        runner_fn=run_select_tool_workflow,
        provides=["tool_selected"],
        requires=["logged_in"],
    ),
    "check_alignment": WorkflowDef(
        name="check_alignment",
        runner_fn=run_check_alignment_workflow,   # future
        provides=["alignment_checked"],
        requires=["tool_selected"],
    ),
}
```

**Dependency resolution example**:

```
Command: "check alignment on MCD018"
  → Parsed actions: ["check_alignment"]
  → check_alignment requires ["tool_selected"]
  → tool_selected requires ["logged_in"]
  → Resolved chain: ["login", "select_tool", "check_alignment"]

  But if RCS is already logged in (detected from window state):
  → Skip login
  → Resolved chain: ["select_tool", "check_alignment"]
```

### 4.4 `chain_executor.py` — Sequential Execution with State Bridging

```python
class ChainExecutor:
    def execute(self, plan: CommandPlan, settings: WorkflowSettings) -> CommandResult:
        shared_context = build_initial_context(plan.parameters)

        for workflow_name in plan.workflow_chain:
            definition = WORKFLOW_REGISTRY[workflow_name]
            run = definition.runner_fn(settings=settings, **plan.parameters)

            if run.status != "completed":
                return CommandResult(status="partial", ...)
                break

            # Bridge state to next workflow
            self._bridge_state(shared_context, workflow_name)

        return CommandResult(status="completed", ...)
```

**State bridging** between workflows:

```
After "login" completes:
  → Call find_rcs_main_window() to get window handle
  → Store in shared_context: rcs_main_window, rcs_main_title, backend

After "select_tool" completes:
  → Store: tool_selected=True, tool_point_on_screen
```

### 4.5 `command_orchestrator.py` — Top-Level Entry Point

```python
class CommandOrchestrator:
    def __init__(self, settings=None):
        self.settings = settings or load_workflow_settings()
        self.parser = IntentParser()
        self.planner = WorkflowPlanner()
        self.executor = ChainExecutor()

    def run(self, command: str) -> CommandResult:
        intent = self.parser.parse(command)
        plan = self.planner.plan(intent, current_state=self._detect_state())
        result = self.executor.execute(plan, self.settings)
        self._report(result)
        return result

    def _detect_state(self) -> dict:
        """현재 RCS 상태 감지 (프로세스, 윈도우)."""
        # read logs/open_rcs_state.json
        # check if RCS window is visible
        ...

    def _report(self, result: CommandResult):
        print(f"[결과] {result.summary_message}")
```

**Entry point** (interactive prompt, no argparse):

```bash
uv run python -m poc.workflow_1.command_orchestrator
# > 명령을 입력하세요: login and go to MCD018
# > [결과] RCS 로그인 완료. Tool MCD018 선택 완료.
# > 명령을 입력하세요: check alignment
# > [결과] Tool MCD018 alignment 상태: aligned
# > 명령을 입력하세요: exit
```

Interactive loop keeps the orchestrator alive for multiple commands, maintaining state across them (e.g., already logged in → skip login on next command).

**NLU approach**: LLM-first (configurable model) with regex fallback when LLM is unavailable.

---

## 5. State Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ shared_context                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Initial (from NLU):                                            │
│    target_tool_name: "MCD018"                                   │
│    process_exe_name: "RcsMainHD.exe"                            │
│                                                                 │
│  After login workflow:                                          │
│    + logged_in: True                                            │
│    + rcs_main_window: <handle>                                  │
│    + rcs_main_title: "Remote Control System - R2"               │
│    + rcs_main_backend: "uia"                                    │
│                                                                 │
│  After select_tool workflow:                                    │
│    + tool_selected: True                                        │
│    + tool_point_on_screen: {"x": 450, "y": 320}                │
│                                                                 │
│  After check_alignment workflow (future):                       │
│    + alignment_status: "aligned" | "failed"                     │
│    + alignment_details: {...}                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Required Changes to Existing Code

### `vlm_client.py` — Add `chat_text()` method (~15 lines)

Currently all methods require an image. Need a text-only path for NLU:

```python
# In OpenAICompatibleVLMClient:
def chat_text(self, model, system_message, user_text, temperature=0.0, max_tokens=2048) -> str:
    """이미지 없이 텍스트 전용 chat completions 요청."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_text},
    ]
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    response = requests.post(self.endpoint, headers=self._headers(), json=payload, timeout=self.timeout_sec)
    response.raise_for_status()
    return self._extract_text_from_json_body(response.json())

# In Workflow1VLMClient:
def chat_text(self, system_message, user_text, **kwargs) -> ChatResponse:
    """텍스트 전용 LLM 호출."""
    ...
```

### No changes needed to:
- `workflow_runner.py`
- `workflow_types.py`
- `workflow_config.py`
- `workflow_login.py` (Option A: ChainExecutor re-detects state)
- `workflow_select_tool.py`

---

## 7. New File Structure

```
poc/workflow_1/
├── command_orchestrator.py     ← NEW: entry point
├── command_types.py            ← NEW: ParsedIntent, CommandPlan, CommandResult
├── intent_parser.py            ← NEW: LLM NLU + regex fallback
├── workflow_planner.py         ← NEW: registry + dependency resolver
├── chain_executor.py           ← NEW: sequential runner with state bridge
├── prompts/
│   └── prompt_intent_parser.py ← NEW: NLU prompt for kimi-k2.5
│
├── workflow_runner.py          (existing, no change)
├── workflow_types.py           (existing, no change)
├── workflow_config.py          (existing, no change)
├── workflow_login.py           (existing, no change)
├── workflow_select_tool.py     (existing, no change)
├── vlm_client.py               (existing, add chat_text method)
└── ...
```

---

## 8. Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `command_types.py` | Pure dataclasses, no deps |
| 2 | `vlm_client.py` | Add `chat_text()` to existing client |
| 3 | `prompts/prompt_intent_parser.py` | NLU prompt template |
| 4 | `intent_parser.py` | LLM parse + regex fallback |
| 5 | `workflow_planner.py` | Registry + dependency resolution |
| 6 | `chain_executor.py` | Sequential executor with state bridging |
| 7 | `command_orchestrator.py` | Wire everything, entry point |

---

## 9. Verification

1. **Intent parser unit test**: Feed various command strings, verify parsed intents
2. **Planner unit test**: Verify dependency resolution chains
3. **Chain executor integration test** (Windows):
   ```bash
   COMMAND="login and go to tool 6MCD2201" uv run python -m poc.workflow_1.command_orchestrator
   ```
4. Check `logs/workflow_runs/` for correct chaining artifacts

---

## 10. Extensibility

Adding a new workflow requires exactly 3 touches:

1. **Implement** `workflow_check_measurement.py` with `run_check_measurement_workflow()`
2. **Register** in `WORKFLOW_REGISTRY` in `workflow_planner.py`
3. **Add action** keyword to NLU prompt in `prompt_intent_parser.py`

No changes to orchestrator, chain executor, or existing workflows.
