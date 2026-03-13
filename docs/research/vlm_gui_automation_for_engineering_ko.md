# Research: VLM-Based GUI Automation for Engineering Workflows

## Context

Our project automates CD-SEM (VeritySEM/RCS) recipe setup — a Windows desktop application for semiconductor metrology. The current implementation uses screenshot → VLM coordinate extraction → pynput click/type, with a two-stage pipeline (PaddleOCR-VL + ui-venus). This research explores how to evolve toward a robust, multi-step workflow automation system specifically for **engineering/industrial applications**.

---

## 1. Controlling Mouse & Keyboard — What to Consider

### Current Approach (our codebase)
- **pynput** for mouse/keyboard (pywinauto failed on RCS legacy controls)
- `click_at()`: VLM coords → screen coords → click with 2-retry fallback
- Smooth mouse movement via linear interpolation (60fps)
- Hotkey combos, text typing, drag operations supported

### What the field recommends

| Technique | Description | Relevance |
|-----------|-------------|-----------|
| **Normalized coordinates (0-1)** | UI-TARS uses `(x/width, y/height)` instead of absolute pixels. Enables cross-resolution transfer | High — RCS may run at different resolutions on different machines |
| **Coordinate-free grounding** | GUI-Actor (Microsoft) uses attention-based patch alignment instead of generating coordinates as text. Outperforms UI-TARS-72B with 7B model | Future — requires custom model training |
| **Set-of-Mark (SoM)** | OmniParser overlays numbered bounding boxes on screenshot before sending to VLM. Improved grounding 70.5% → 93.8% | **High priority** — can be added to our pipeline without model changes |
| **Action space standardization** | UI-TARS defines: Click, Drag, Scroll, Type, Hotkey, Wait, Finished, CallUser | Adopt — standardize our action types |
| **Pre-execution verification** | VeriSafe Agent: encode safety constraints as DSL, verify action before execution. 94-98% accuracy | **Critical for engineering** — wrong click = wafer damage |
| **Readback verification** | After typing a value: screenshot → OCR → verify displayed value matches intended | **Must-have for recipe params** |

### Engineering-specific considerations
- **Dense numerical UIs**: VLMs struggle with small text in dense parameter fields. Our OCR-assist pipeline is well-aligned with research (two-stage > single-stage for complex layouts)
- **Legacy Win32 controls**: No accessibility tree → pure vision approach is correct. OmniParser V2 (YOLOv8 + Florence-2 + PaddleOCR) can detect interactive elements without accessibility APIs
- **Precise field targeting**: For small input fields, use **zoomed crop** around target area for higher accuracy (RegionFocus: +28% on ScreenSpot-Pro)

---

## 2. Decision-Making — How VLM Agents Decide What to Do

### Paradigms

#### a) ReAct (Reason + Act) — Most Common
```
Loop:
  1. Observe: screenshot + OCR
  2. Think: "I see the login screen. I need to click the Server field."
  3. Act: click(x=150, y=200)
  4. Observe: new screenshot
  5. Think: "Server field is now focused. I need to type the server address."
  ...
```
- Used by: UI-TARS, Anthropic Computer Use, most GUI agents
- Pros: Flexible, handles unexpected states
- Cons: Each step costs a VLM call, slow for long workflows

#### b) Plan-then-Execute — Better for Known Workflows
```
1. Plan: "Recipe setup requires: login → navigate to recipe editor → set parameters → save"
2. Execute each step, verify with VLM only when needed
3. Re-plan if unexpected state detected
```
- Used by: Agent-S (Manager + Worker), UFO2 (HostAgent + AppAgent)
- Pros: Faster, fewer VLM calls, predictable
- **Best fit for RCS**: Recipe setup is a known workflow — plan is fixed, only element locations vary

#### c) State Machine with Pre-learned Knowledge — Best for Industrial
```
States: {login_screen, main_menu, recipe_editor, parameter_dialog, ...}
Transitions: {login_screen --[click login]--> main_menu, ...}
At each state: execute pre-defined action sequence
```
- Used by: **ActionEngine** (95% success, 11.8x cost reduction vs ReAct), **InfraMind** (83% on industrial GUIs)
- **Recommended for RCS** — recipe setup has a fixed state graph

### InfraMind — Most Relevant Framework for Our Use Case

InfraMind (Sept 2025) addresses industrial management GUIs with the same challenges as RCS:

| InfraMind Challenge | Our RCS Equivalent |
|---|---|
| Custom-developed controls, no accessibility | RCS legacy Win32 controls, pywinauto fails |
| Desktop apps lack URL identifiers | Dozens of RCS dialogs look similar |
| Air-gapped environments | Semiconductor fab networks are isolated |
| Safety-critical controls | Wrong recipe params = wafer damage |
| Precision + efficiency needed | Recipe setup requires exact parameter values |

**InfraMind's approach:**
1. **Exploration phase**: BFS/DFS through every GUI element (with VM snapshots for safe rollback). Builds a complete element functionality map
2. **Knowledge distillation**: Large model (GPT-4o) explores → structured knowledge base → small 7B model deploys
3. **State identification**: CLIP visual embeddings + textual descriptions for dual state representation
4. **Memory-driven planning**: Stores successful action-flow trees, replays them
5. **Safety**: CLIP-based blacklist filtering, confirmation dialogs, risk assessment

**Result**: 83.3% on OpenDCIM, 76.7% on commercial platform — far ahead of general agents (UI-TARS: 43.3%/20.0%)

### ActionEngine — State Machine Memory

ActionEngine (Feb 2025) models GUIs as state machines M=(S,O,T):
- **States**: Structural views (composed of atoms to prevent state explosion)
- **Operations**: Edges (UI manipulation or data collection)
- **Transitions**: Validated by actual execution
- Semi-automated crawler builds graph (typically 20-30 states, 100-150 transitions)
- Compiles workflows into executable plans in **single inference step**
- **95% success rate** vs 66% baselines, 11.8x cost reduction

---

## 3. Process Extraction — Recording and Replaying Workflows

### The Core Question
How do we capture a human engineer's recipe setup process and convert it into a replayable automation?

### Approach A: Record Human Demonstration (ShowUI-Aloha)

**ShowUI-Aloha** (Dec 2025) provides the most complete open-source system:

1. **Recorder app** (Windows .exe): captures screen video 30fps + mouse/keyboard events with timestamps
2. **Raw Log Parser**: merges consecutive keystrokes, reconstructs drags, deduplicates clicks
3. **Screenshot Marker**: generates annotated screenshots per action (red X for clicks, polyline for drags)
4. **Trace Generator**: VLM produces structured JSON per step:
   ```json
   {
     "observation": "Login dialog with Server, UserID, Password fields",
     "think": "Need to enter server address in the Server field",
     "action": "click(x=150, y=200) then type('SEM-SERVER-01')",
     "expectation": "Server field should now show 'SEM-SERVER-01'"
   }
   ```
5. Output designed for **generalization** — single demo works across UI layout variants

### Approach B: Record-and-Replay with Adaptation (AgentRR)

**AgentRR** (May 2025) — most practical for our case:

1. **Record phase**: Engineer performs recipe setup once. System captures:
   - Screenshots at each action
   - Mouse/keyboard events
   - VLM-generated reasoning for each step
2. **Experience abstraction** (3 levels):
   - **Low**: Exact replay (same coordinates, same values)
   - **Medium**: Parameterized (same workflow, different values — e.g., different recipe params)
   - **High**: Conceptual (same goal, adapted to UI changes)
3. **Replay phase**: Start at low level, escalate to higher levels when environment changes
4. **141% improvement** over base model using only 312 human trajectories (PC Agent-E)

### Approach C: Systematic Exploration (InfraMind)

Don't record a human — let the VLM explore the application systematically:

1. BFS/DFS through every interactive element
2. VM snapshot before each action → rollback if unwanted
3. Compare before/after screenshots to learn element functions
4. Build complete state graph automatically
5. **Advantage**: Discovers all paths, not just the one the human demonstrated

### Recommended Strategy for RCS

**Hybrid: Record + Explore + Parameterize**

```
Phase 1: EXPLORE
  - Let VLM systematically explore RCS screens
  - Build state graph: {screen_name → [elements, transitions]}
  - Identify all interactive elements per screen

Phase 2: RECORD
  - Engineer performs one recipe setup while recording
  - ShowUI-Aloha style: video + events + screenshots
  - VLM converts recording to structured trace

Phase 3: PARAMETERIZE
  - Extract recipe parameters from trace (server, measurement sites, thresholds)
  - Create recipe template: fixed workflow + variable parameters
  - Template format:
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

Phase 4: REPLAY
  - Load recipe template + parameter values
  - Execute steps with VLM verification at each state transition
  - Readback verification for all typed values
```

---

## 4. State Tracking — How to Let the VLM Know "Where We Are"

### The Problem
RCS recipe setup involves dozens of screens/dialogs. The VLM needs to know:
- Which screen am I looking at?
- What step in the workflow am I on?
- What has been completed so far?
- What should I do next?

### Approach A: Screenshot History (UI-TARS Style)

```
Context to VLM:
  - Last 5 screenshots (FIFO sliding window)
  - Full text history of thoughts + actions
  - Current step number: "Step 7 of 23"
```

**Research findings on history management:**
- UI-TARS: max 5 screenshots + full text history
- Agent-S: 8 image turns
- JetBrains Research (2025): **Observation masking** (keep recent 10 turns, replace older screenshots with text placeholders) often **outperforms** LLM summarization — cuts costs 50%+ with equal or better quality
- AgentProg: Reframe history as program with variables, prune to active execution path (~9k tokens vs 17k+ baselines)

### Approach B: State Machine (Recommended for RCS)

```python
# Define expected states
STATES = {
    "login_screen": {
        "visual_cues": ["Server", "User ID", "Password", "Log In"],
        "clip_embedding": "<precomputed>",
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

# At each step:
# 1. Capture screenshot
# 2. Identify current state (CLIP similarity + OCR keyword matching)
# 3. Look up expected actions for this state
# 4. Execute actions
# 5. Verify transition to next expected state
```

**Dual state identification** (InfraMind):
- **Semantic**: VLM describes what it sees → match to known states
- **Visual**: CLIP embedding similarity to known state screenshots
- Combined scoring prevents misidentification of visually similar screens

### Approach C: Progress Tracking Prompt

Include workflow context in every VLM call:

```
You are automating CD-SEM recipe setup in RCS software.

WORKFLOW PROGRESS:
  [x] Step 1: Login to RCS (completed)
  [x] Step 2: Navigate to Recipe Editor (completed)
  [ ] Step 3: Create new recipe (CURRENT)
  [ ] Step 4: Set measurement parameters
  [ ] Step 5: Define measurement sites
  [ ] Step 6: Save recipe

CURRENT STATE: Recipe Editor - main view
LAST ACTION: Clicked "Recipe" tab in main menu
EXPECTED: "New Recipe" button should be visible

Based on the screenshot, identify the "New Recipe" button and return its coordinates.
```

### Recommended: Combine B + C

- Use **state machine** for automated state identification (fast, no VLM call needed for known states)
- Use **progress tracking prompt** when VLM assistance is needed (element location, unexpected states)
- Fall back to **screenshot history** only when state is unrecognized

---

## 5. Architecture Recommendations for Our Project

### Current vs Proposed Architecture

```
CURRENT (per-script, hardcoded sequence):
  automate_rcs_login.py → click_rcs_view_mode.py → check_tool_screen.py
  Each script: screenshot → VLM → parse → click (repeat)

PROPOSED (workflow engine with state machine):
  ┌─────────────────────────────────────────────┐
  │           Recipe Workflow Engine             │
  │                                              │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
  │  │  State   │  │ Action   │  │ Verifier  │  │
  │  │ Recognizer│  │ Executor │  │           │  │
  │  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
  │       │              │              │         │
  │  ┌────▼──────────────▼──────────────▼─────┐  │
  │  │         State Machine Graph            │  │
  │  │  (pre-learned from exploration/demo)   │  │
  │  └────────────────────────────────────────┘  │
  │                                              │
  │  ┌────────────┐  ┌───────────────────────┐   │
  │  │ Recipe     │  │ VLM Pipeline          │   │
  │  │ Template   │  │ (OCR + Primary VLM)   │   │
  │  │ (JSON)     │  │                       │   │
  │  └────────────┘  └───────────────────────┘   │
  └─────────────────────────────────────────────┘
```

### Key Components to Build

1. **State Recognizer**: CLIP embedding + OCR keyword matching → identifies current screen without VLM call
2. **Action Executor**: Standardized action types (click, type, hotkey, wait, scroll, verify)
3. **Verifier**: Post-action screenshot → OCR → confirm expected result (readback for typed values)
4. **State Machine Graph**: Pre-built from exploration or recording, stored as JSON
5. **Recipe Template**: Parameterized workflow definition (fixed steps + variable values)
6. **Safety Layer**: Pre-execution constraint checking, blacklist for dangerous elements

### Priority Order

| Priority | What | Why |
|----------|------|-----|
| **P0** | Set-of-Mark (SoM) overlay for better grounding | +23% accuracy, no model change needed |
| **P0** | Readback verification after typing | Critical for recipe parameter accuracy |
| **P1** | State recognizer (CLIP + OCR) | Eliminate redundant VLM calls, enable state machine |
| **P1** | Standardized action types + workflow JSON | Replace per-script hardcoding |
| **P2** | Recording system (ShowUI-Aloha style) | Capture engineer demos for new workflows |
| **P2** | Pre-execution safety constraints | Prevent dangerous actions |
| **P3** | Exploration crawler | Auto-discover all RCS screens and elements |
| **P3** | Knowledge distillation to smaller model | Production deployment in fab |

---

## 6. Key Papers & Tools Reference

### Must-Read (directly applicable to our engineering use case)

| Paper/Tool | Key Contribution | Link |
|---|---|---|
| **InfraMind** (Sept 2025) | Industrial GUI framework: exploration, safety, state ID, knowledge distillation | arxiv:2509.13704 |
| **ActionEngine** (Feb 2025) | State machine memory for GUI agents, 95% success, 11.8x cost reduction | arxiv:2602.20502 |
| **AgentRR** (May 2025) | Record-and-replay with multi-level experience abstraction | arxiv:2505.17716 |
| **ShowUI-Aloha** (Dec 2025) | Complete demo recording → trace generation pipeline | github:showlab/ShowUI-Aloha |
| **VeriSafe Agent** (Mar 2025) | DSL-based pre-execution safety verification | arxiv:2503.18492 |
| **OmniParser V2** (Feb 2025) | Pure-vision screen parsing (YOLOv8 + Florence-2 + PaddleOCR) | github:microsoft/OmniParser |
| **WorldGUI** (Feb 2025) | Desktop GUI benchmark exposing agent failure modes | arxiv:2502.08047 |

### Important Context

| Paper/Tool | Key Contribution | Link |
|---|---|---|
| **UI-TARS 2** (2025) | SOTA end-to-end GUI agent with System-2 reasoning | arxiv:2509.02544 |
| **Agent-S3** (2025) | Two-tier architecture, surpasses human on OSWorld | github:simular-ai/Agent-S |
| **UFO2** (Apr 2025) | Dual-agent + hybrid perception (accessibility + vision fusion) | github:microsoft/UFO |
| **GUI-Actor** (2025) | Coordinate-free grounding, outperforms UI-TARS-72B at 7B | microsoft.github.io/GUI-Actor |
| **RegionFocus** (2025) | Visual zoom for test-time scaling, +28% grounding | arxiv:2505.00684 |
| **PC Agent-E** (2025) | 141% improvement with 312 human trajectories | arxiv:2505.13909 |
| **VAGEN** (2025) | Three-stage verification: static → retrospective → probing | arxiv:2602.00575 |

### Semiconductor-specific

| Resource | Relevance |
|---|---|
| **GUIDE-X** (SPIE 2025) | VLM+LLM for semiconductor X-ray inspection guidance |
| **Canopus AI / Siemens** (Jan 2026) | AI-driven metrology workflow automation |
| **Applied Materials AIx** | Real-time process recipe optimization |
| **Design-Based Metrology** | Existing CAD-to-recipe automation (different level than GUI) |

---

## 7. Summary: Key Takeaways for Our Project

1. **State machine > ReAct for known workflows**: Recipe setup is predictable — don't waste VLM calls on reasoning when the workflow is fixed. Use VLM only for element location and verification.

2. **Set-of-Mark is the easiest accuracy win**: Overlay numbered bounding boxes on screenshots before sending to VLM. OmniParser V2 can detect interactive elements purely from vision.

3. **Record once, replay many**: Use ShowUI-Aloha or custom recording to capture one demo, parameterize it, then replay with different recipe values.

4. **Dual state identification beats single**: CLIP embedding similarity + OCR keyword matching is more robust than either alone (InfraMind approach).

5. **Readback verification is non-negotiable for engineering**: After every typed value, screenshot → OCR → verify. This is what separates industrial automation from consumer-app agents.

6. **Observation masking > summarization for history**: Keep last 10 turns detailed, replace older with text placeholders. Simpler, cheaper, and often more effective than LLM summarization.

7. **Safety constraints must be explicit**: Define a DSL or blacklist for dangerous actions. Never rely on VLM "common sense" for safety in engineering contexts.

8. **Our two-stage OCR pipeline is validated by research**: The field confirms that pipeline OCR outperforms end-to-end VLMs on dense/complex layouts. Keep and strengthen this approach.
