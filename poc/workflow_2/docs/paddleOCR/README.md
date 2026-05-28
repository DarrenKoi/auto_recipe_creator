# PaddleOCR Usage Guide For Workflow 2

Date: 2026-05-28

This note is for the recurring problem in this project: PaddleOCR is available, but it often fails when it is used as a general GUI understanding model. The practical fix is not "use PaddleOCR everywhere". The fix is to use the right PaddleOCR mode on the right crop, then validate its output before any automation action.

## Executive Rule

Use PaddleOCR as an OCR evidence engine, not as the final GUI actor.

In this repository:

- Use OpenCV/Chamfer/ORB for align-key shape matching and SEM target decisions.
- Use UI grounding VLMs such as `ui-venus` and `mai-ui` to propose GUI regions or buttons.
- Use PaddleOCR-VL only to read text from a constrained crop, or to return text boxes from a simple text-heavy crop.
- Never let PaddleOCR alone decide a click on a full RCS screenshot.

## What Model Are We Actually Running?

The repo currently exposes `paddleocr-vl-1.5` through the same OpenAI-compatible VLM proxy used by other models:

- Flask route: `flask_api/vlm_serve/paddleocr_vl.py`
- Upstream port: `8004`
- Model config: `deploy_vlms/config/models/paddleocr-vl-1.5.env`
- Client path: `poc.workflow_1.vlm_client.Workflow1VLMClient`
- Prompt helpers: `poc.workflow_1.prompts.prompt_ocr_assist`

Important distinction from the official docs:

- The full PaddleOCR-VL pipeline has layout analysis plus VLM recognition.
- Running only the VLM component through vLLM/chat completions is not the same as the complete PaddleOCR document parsing pipeline.
- Our current `paddleocr-vl-1.5` route is best treated as a task-prompted OCR sidecar, not the full production document parser unless we explicitly add the full PaddleOCR pipeline service.

This explains a major failure mode: the model is excellent on document-like crops, but unstable on full GUI screenshots that contain many panels, icons, repeated labels, and non-document layout.

## Official Capability Summary

PaddleOCR-VL is designed for document parsing and element recognition. Official materials describe these strengths:

- multilingual OCR
- document layout parsing
- table recognition
- formula recognition
- chart recognition
- text spotting with localization
- robustness improvements for skew, warping, screen photography, illumination, and scans in PaddleOCR-VL-1.5

For general OCR, PaddleOCR also provides the classic PP-OCR pipeline. PP-OCR is a text detection plus text recognition pipeline and is often a better fit when the input is just UI text lines and you need stable text boxes, confidence, and speed.

## Decision Matrix

| Situation | Use PaddleOCR? | Recommended mode | Why |
|---|---:|---|---|
| Wait Input dialog existence check | Yes | `OCR:` on dialog crop | Long signature text such as `alignment mark` is stable evidence. |
| Confirm text after typing into a field | Yes | `OCR:` on field crop or panel crop | It verifies visible text without trusting model reasoning. |
| Find text rows in a narrow list crop | Sometimes | `Spotting:` on one list/row crop | Useful only if parser output is validated and rows are not ambiguous. |
| Read a parameter table or recipe-like panel | Yes | `Table Recognition:` or full PaddleOCR pipeline | This is closest to the model's document/table strength. |
| Parse a screenshot report, PDF, image page, or dense document | Yes | full PaddleOCR-VL pipeline | Use layout detection, reading order, and saved JSON/Markdown outputs. |
| Locate the OK button from full GUI screenshot | No, not alone | VLM region first, OCR crop confirm | Short labels like `OK` are weak and appear in many places. |
| Select an RCS tool row by ID | Not as primary | VLM row proposal, optional OCR strip verify | Full-list spotting has already produced garbage or slow fan-out in this repo. |
| Align-key matching in SEM image | No | OpenCV/Chamfer/ORB | Align keys are shape/edge targets, not text targets. |
| Crosshair or stage movement decision | No | deterministic CV plus bounded automation | OCR cannot validate SEM geometry. |
| Whole RCS screen understanding | Avoid | crop first | Whole-screen GUI OCR caused hallucinated text in project evidence. |

## Best Practices For This Project

### 1. Crop first

PaddleOCR should receive the smallest crop that still contains the text evidence.

Good crops:

- dialog body
- one input field plus label
- one row strip
- one parameter table
- one text-heavy panel

Bad crops:

- entire RCS screen
- entire SEM Monitor plus surrounding tool chrome
- multi-panel screenshots with repeated labels
- button-only crop where the only evidence is `OK`

### 2. Prefer long, unique text signatures

Do not confirm a modal by reading only `OK`. Confirm the body text or title:

- `alignment mark`
- `cross cursor`
- `Wait Input`
- tool ID with neighboring row context
- recipe/class name with label context

The local `vlm_wait_input_ok_button.py` script already follows this rule: VLM proposes the dialog and OK button, then PaddleOCR reads only the dialog crop and checks for unique signature text.

### 3. Use task prompts exactly

For the vLLM/OpenAI-compatible route, use short task labels:

| Task | Prompt |
|---|---|
| Plain text OCR | `OCR:` |
| Text plus location candidates | `Spotting:` |
| Table extraction | `Table Recognition:` |
| Formula extraction | `Formula Recognition:` |
| Chart extraction | `Chart Recognition:` |

Keep `temperature=0.0`. Use a bounded `max_tokens`; for small GUI crop checks, 128 to 512 tokens is usually enough.

### 4. Validate output before use

For `OCR:`:

- normalize case and whitespace
- strip punctuation when matching fixed phrases
- require one of several known signature tokens
- store raw OCR text in debug artifacts

For `Spotting:`:

- parse JSON-like output defensively
- support bbox lists, dict bboxes, polygons, and wrapper keys
- reject boxes outside the crop
- reject duplicate or overlapping row candidates unless a deterministic tie-breaker exists
- map crop coordinates back to full screenshot coordinates only after validation

The repo already has `poc.workflow_1.ocr_spotting.parse_spotting_items()` for best-effort normalization of `Spotting:` responses.

### 5. Make OCR a gate, not the action source

Preferred automation pattern:

1. Observe screen.
2. Use UI VLM or deterministic CV to propose a region.
3. Crop that region.
4. Run PaddleOCR on the crop.
5. Check text signature or text-box evidence.
6. If confirmed, execute the guarded click/movement from the validated source.
7. If not confirmed, do not click. Save artifacts and escalate or retry.

For workflow_2, this means:

- PaddleOCR may confirm a popup before clicking OK.
- PaddleOCR may confirm a tool/recipe/name is visible in a crop.
- PaddleOCR must not choose SEM movement, align-key center, or final match score.

## Workflow 2 Use Cases

### A. Wait Input popup confirmation

Use case:

- VLM finds a possible Wait Input dialog and OK button.
- Crop the dialog with padding.
- Upscale the crop if text is small.
- Send only the crop to PaddleOCR with `OCR:`.
- Confirm only if OCR contains one of the known body/title signatures.

Recommended signatures:

- `alignment mark`
- `cross cursor`
- `wait input`

Failure handling:

- If OCR returns unrelated text, mark popup absent.
- If OCR errors, do not click.
- If VLM finds an OK button but OCR does not confirm the dialog, do not click.

### B. Tool list or recipe row verification

Use case:

- UI VLM proposes a row or point in a list.
- Build a narrow horizontal strip around that row.
- Run `Spotting:` only on the strip.
- Match the expected tool ID or recipe text after normalization.

Do not run `Spotting:` on four large list/full-screen crops in a loop. That was slow and failure-prone in this project.

### C. Parameter or table panel extraction

Use case:

- A recipe parameter pane, run log, or report-like screen needs structured text.
- Crop the table or panel.
- Try `Table Recognition:` if rows/columns matter.
- Save raw response, normalized table, and overlay/crop for review.

If this becomes a common path, add a full PaddleOCR-VL pipeline service rather than only using the chat-completions VLM component.

### D. Debug artifact enrichment

Use case:

- A failed workflow_2 run should include readable evidence for humans.
- Run `OCR:` on selected crops and save the raw text beside overlay images.

This is safe because OCR text becomes evidence, not an automation command.

## When Not To Use PaddleOCR

Do not use PaddleOCR for:

- align-key shape matching
- SEM feature matching
- crosshair localization
- stage movement decisions
- final click coordinates from a full screenshot
- interpreting icons or graphical controls without text
- resolving row ambiguity without geometric validation

For these, keep the current workflow_2 direction:

- deterministic CV for match scoring
- bounded search budgets
- ROI constraints
- VLM only for hints or GUI region proposals
- OCR only for text confirmation

## Implementation Checklist

Before adding a PaddleOCR call, answer these questions:

- Is the target primarily text or a table?
- Can the input be cropped to one region?
- Is there a long unique phrase to confirm?
- Is `max_tokens` capped?
- Is `temperature=0.0`?
- Is the raw OCR output saved?
- Are boxes normalized and bounds-checked?
- Is there a no-click/no-move path on OCR failure?
- Is the result only evidence/gating, not direct authority?

If any answer is no, PaddleOCR is probably the wrong primary tool.

## Recommended Repo Pattern

Use the existing client and prompt helpers:

```python
from poc.workflow_1.prompts import build_ocr_assist_prompt, build_spotting_prompt
from poc.workflow_1.vlm_client import Workflow1VLMClient

client = Workflow1VLMClient(
    service_slug="paddleocr-vl-1.5",
    timeout_sec=120.0,
    log_name="workflow_2_ocr",
)

system_message, user_text = build_ocr_assist_prompt(width=0, height=0)
response = client.chat_with_image_b64(
    image_b64=crop_b64,
    system_message=system_message,
    user_text=user_text,  # "OCR:"
    image_mime="image/webp",
    temperature=0.0,
    max_tokens=256,
)

raw_text = response.text or ""
```

For coordinate-bearing text detection:

```python
from poc.workflow_1.ocr_spotting import parse_spotting_items
from poc.workflow_1.prompts import build_spotting_prompt

system_message, user_text = build_spotting_prompt()  # "Spotting:"
response = client.chat_with_image_path(
    image_path=strip_crop_path,
    system_message=system_message,
    user_text=user_text,
    image_mime="image/webp",
    temperature=0.0,
    max_tokens=512,
)
items = parse_spotting_items(response.text)
```

## Recommended Experiments

Run these before relying on PaddleOCR in a new workflow_2 path:

1. Whole-screen negative test
   - Give the full screenshot to `OCR:`.
   - Confirm that output is not trusted for automation.

2. Crop sensitivity test
   - Test the same target at full screen, panel crop, row/dialog crop.
   - Keep the smallest crop that preserves the signature phrase.

3. Prompt comparison
   - Compare `OCR:` vs `Spotting:` on the same crop.
   - Use `Spotting:` only when bbox candidates are required.

4. Token cap test
   - Compare 128, 256, and 512 max tokens.
   - Pick the smallest cap that preserves required text.

5. Validation replay
   - Save raw response, crop, overlay, parsed JSON, and verdict.
   - Re-run parser logic without another model call.

## Operational Commands

Start the local model service in the deployed VLM environment:

```bash
uv run python deploy_vlms/scripts/start_paddleocr_vl.py
uv run python deploy_vlms/scripts/check_vlm.py http://127.0.0.1:8004 paddleocr-vl-1.5
```

Through Flask, the route is:

```text
/api/vlm_serve/paddleocr-vl-1.5/v1/chat/completions
```

Current model config:

```text
PORT=8004
SERVED_MODEL_NAME=paddleocr-vl-1.5
LIMIT_MM_PER_PROMPT={"image": 1}
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.10
```

## If We Want To Use PaddleOCR Better

The current repo service is enough for crop-level OCR gates. To maximize PaddleOCR, consider adding a separate full PaddleOCR pipeline service for document/table cases:

- `paddleocr doc_parser` or Python `PaddleOCRVL`
- layout detection enabled for document-like inputs
- JSON and Markdown output saved as artifacts
- service endpoint separate from the OpenAI-compatible VLM proxy
- only used for document/report/panel parsing, not SEM geometry

For pure GUI text boxes, also evaluate classic PP-OCRv5:

- faster and more direct for text detection plus recognition
- better fit for line boxes and confidence-driven verification
- less likely to behave like a general chat VLM

## Source Notes

- PaddleOCR-VL official tutorial: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- PaddleOCR OCR pipeline tutorial: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html
- vLLM PaddleOCR-VL usage guide: https://docs.vllm.ai/projects/recipes/en/stable/PaddlePaddle/PaddleOCR-VL.html
- Hugging Face model card: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- PaddleOCR-VL-1.5 paper page: https://arxiv.org/abs/2601.21957

