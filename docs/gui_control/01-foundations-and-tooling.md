# GUI Automation Foundations And Tooling

This document merges the old overview, library survey, capability notes, Microsoft automation tooling notes, and CPU PoC summary into one starting point for this repository.

## 1. What This Repo Is Automating

The target problem is not generic desktop automation. It is RCS/CD-SEM recipe work on Windows applications with a mix of:

- legacy Win32-like controls
- text-heavy parameter panels
- dense engineering layouts
- custom-drawn or partially inaccessible UI regions
- remote-session latency and focus instability

That combination means no single automation method is enough.

## 2. Control Modes

### 2.1 Object-Based Control

Representative tools:

- `pywinauto`
- `uiautomation`
- `WinAppDriver`

Good for:

- standard text fields
- buttons with accessible names or automation IDs
- deterministic dialogs and menus

Weak for:

- custom DirectX/OpenGL surfaces
- remote-rendered content
- controls with poor accessibility exposure

### 2.2 Input-Simulation Control

Representative tools:

- `pynput`

Good for:

- final mouse and keyboard execution
- drag, scroll, hotkey, and fallback clicks
- cases where UIA/object APIs fail

Weak for:

- discovering what should be clicked
- background-safe execution
- strongly typed control semantics

### 2.3 Vision-Led Control

Representative tools:

- `UI-Venus`
- `UI-TARS`
- `MAI-UI`
- external baselines such as `Kimi-K2.5`

Good for:

- screenshot understanding
- icon/tab/button grounding
- layout reasoning when selectors do not exist

Weak for:

- exact string authority
- small dense text without crop or OCR support
- safety-critical execution without verification

### 2.4 OCR And Parser Sidecars

Representative tools:

- `PaddleOCR-VL-1.5`
- `GOT-OCR-2.0-hf`
- `OmniParser V2`

Good for:

- exact text
- table/grid extraction
- row/label anchors
- structured UI element hints

Weak for:

- deciding the final clickable surface by themselves
- acting as the only planner in dense engineering UI

## 3. Recommended Stack In This Repo

| Layer | Preferred tools | Role |
|------|-----------------|------|
| Capture | `mss`, OS window capture helpers | stable screen acquisition |
| Primary grounding | `UI-Venus` or `UI-TARS` | full-screen target selection |
| Zoom-in grounding | `MAI-UI` | small target / crowded crop retry |
| Text authority | `PaddleOCR-VL-1.5`, `GOT-OCR-2.0-hf` | exact text, spotting, hard OCR fallback |
| Structured parser | `OmniParser V2` | interactable boxes + SoM overlay |
| Object fallback | `pywinauto`, `uiautomation` | accessible control access |
| Execution | `pynput` | click, type, drag, hotkey |

## 4. Tool Selection Rules

### 4.1 Standard Windows Controls

Try object-based access first if the screen is known to expose stable controls. It is usually faster and less ambiguous than asking a VLM for a point.

### 4.2 Custom Graphics Or Dense Engineering UI

Prefer screenshot reasoning plus OCR or parser sidecars. This is the default operating regime for many RCS-like screens.

### 4.3 Final Click And Type

Use `pynput` or the existing click/type helpers after the decision stage. Treat execution as a separate layer from perception.

## 5. Known Limits

### 5.1 Permissions And OS Boundaries

- elevated/UAC contexts can break automation
- remote desktop focus can drift
- keyboard modifiers can remain latched in remote sessions

### 5.2 Custom UI Surfaces

Accessibility-based tools can fail completely on custom-rendered controls. This is why the repo keeps vision-first paths active.

### 5.3 Network And Runtime Latency

Screenshot capture is fast. The main cost is model inference and remote/runtime queuing. Never assume the screen is unchanged after a 1-3 second model round-trip.

### 5.4 Safety

Industrial GUI automation should not be treated like ordinary office UI scripting. Some clicks can trigger real equipment movement or bad recipe changes.

## 6. Core Automation Rules

- Use `observe -> decide -> act -> verify`.
- Keep `SAFE_MODE=true` by default.
- Log the prompt, screenshot, response, and final action coordinates.
- Do not trust a single full-screen pass for tiny targets.
- Separate semantic choice from exact text recovery.
- Keep human approval in the loop for high-risk operations.

## 7. CPU Baseline And Why It Still Matters

The repo already proved a useful Tier 1 pattern:

`capture -> optimize image -> call VLM -> parse action -> execute`

That CPU/API baseline still matters because it provides:

- a no-GPU fallback
- benchmark baselines before local serving improvements
- a safe way to validate prompts and action schemas before deploying more aggressive local stacks

Key repo patterns that remain valid:

- local debug images in JPEG
- WebP payloads for VLM requests
- explicit `SAFE_MODE`
- compact JSON outputs for downstream action code

## 8. Current Repo Anchors

- `poc/work2/open_rcs.py`
- `poc/work2/login_rcs.py`
- `poc/work2/login_benchmark.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/util/image_utils.py`
- `poc/work2/util/window_utils.py`
