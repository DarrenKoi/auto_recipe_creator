# Dynamic Screen Safety

This document condenses the dynamic-screen research into a working policy for SEM/probe-monitor automation.

## 1. Why Dynamic Screens Are Different

Static dialogs tolerate latency. Probe-monitor and live SEM screens do not.

Main risks:

- the screenshot changes during model inference
- state transitions are captured mid-animation
- a wrong double-click can trigger physical probe movement

The automation policy must treat these screens as a different safety class.

## 2. Tiered Risk Model

### Tier 1: Read-Only Monitoring

- periodic screenshot analysis
- status and anomaly detection
- no physical action

### Tier 2: Low-Risk Recovery

- click static UI buttons outside danger regions
- retry or dismiss recoverable UI states

### Tier 3: High-Risk Probe Actions

- SEM/probe area targeting
- double-click driven repositioning
- actions that can affect real equipment state

Tier 3 requires the strongest guard rails and should stay opt-in.

## 3. Region-Based Stability

Do not ask whether the whole screen is stable. Ask whether the relevant region is stable.

Typical zone classes:

- `static`: menu bars, toolbars, side panels
- `dynamic`: logs, status bars, changing but non-dangerous data
- `danger`: probe monitor or other physical-impact regions

For static UI clicks:

- ignore the SEM image region
- evaluate stability on the menu/panel area

For probe verification:

- focus on the probe-monitor region only

## 4. Stability Rules

Practical defaults:

- poll every `0.3s`
- require at least `2` consecutive stable checks
- treat `diff_ratio < 0.02` as stable
- abort after about `10s` timeout

Verification rule after a model response:

- if the relevant region changed by more than about `10%`, the predicted coordinates should be considered stale
- recapture and re-evaluate instead of clicking

## 5. Safe Execution Pattern

Use:

`capture -> stabilize -> analyze -> verify -> act`

Not:

`capture -> analyze -> click after a long delay`

For dynamic screens, the verification capture is not optional.

## 6. Double-Click Guard Rails

High-risk double-click actions should pass all three checks:

1. `SAFE_MODE=false`
2. target is in an allowed zone or an explicit danger override is present
3. the action builder explicitly allows `click_count=2`

If any check fails, log the blocked action and keep the evidence.

## 7. Prompting Differences For Probe Screens

Probe-monitor prompts should:

- mention that the image is a live SEM/probe frame
- separate current probe location from target location
- make the monitor boundary explicit
- reject coordinates outside the monitor region

Use OCR hints only as support. The core task is still visual target grounding under motion.

## 8. Operational Guidance

- Start with Tier 1 monitoring before enabling Tier 2 or Tier 3.
- Keep human approval for any action that can move a probe or change a high-risk recipe state.
- Save region overlays showing the static, dynamic, and danger zones.
- Tune zone percentages on office Windows machines, not on macOS.
