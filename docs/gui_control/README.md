# GUI Control Guide

`docs/gui_control/` is now the canonical home for GUI automation design notes that were previously split between `docs/gui_control/` and `docs/research/`.

Use this folder for:

- Windows GUI control strategy
- GUI model selection and hybrid automation patterns
- `UI-Venus` + OCR + crop-retry grounding guidance
- dynamic screen safety rules for SEM/probe workflows
- RCS video-to-action planning

Use `docs/setup_vlms/` for:

- vLLM installation
- model bring-up and runtime settings
- UI-TARS special runtime requirements
- OCR/parser service deployment

## Recommended Reading Order

1. [`01-foundations-and-tooling.md`](./01-foundations-and-tooling.md)
2. [`02-grounding-hybrid-patterns.md`](./02-grounding-hybrid-patterns.md)
3. [`03-ui-venus-ocr-crop-retry.md`](./03-ui-venus-ocr-crop-retry.md)
4. [`04-dynamic-screen-safety.md`](./04-dynamic-screen-safety.md)
5. [`05-rcs-video-to-action.md`](./05-rcs-video-to-action.md)

## Repo Anchors

- `poc/work2/`: current GUI automation experiments and coworker-facing client path
- `poc/work2/flask_vlm.py`: service registry used by `work2`
- `poc/work2/vlm_client.py`: service-slug based image client
- `poc/work2/login_rcs.py`: login screen capture / coordinate extraction entry
- `poc/work2/ocr_login_check.py`: OCR prompt and response inspection
- `test/video_frame_parser/`: offline video parsing and episode extraction
- `test/vlm_input_control/`: older automation, retrieval, and prompt experiments

## Operating Principles

- Prefer `observe -> decide -> act -> verify` over fixed sleep-based macros.
- Keep `SAFE_MODE=true` unless real office validation requires action.
- Save local debug screenshots as JPEG and send WebP to VLM endpoints when possible.
- Treat exact text as an OCR responsibility and target selection as a GUI-grounding responsibility.
- Escalate from cheap/structured methods to expensive/flexible ones only when needed.
