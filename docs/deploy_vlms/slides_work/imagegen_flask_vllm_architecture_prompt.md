# Imagegen Prompt Spec

This prompt was produced with the `imagegen` skill CLI in `--dry-run` mode on March 12, 2026.
Live image generation was not executed in this environment because `OPENAI_API_KEY` is not set.

## Dry-run payload

```json
{
  "endpoint": "/v1/images/generations",
  "model": "gpt-image-1.5",
  "n": 1,
  "prompt": "Use case: infographic-diagram\nPrimary request: Flask server and vLLM deployment architecture for this repository\nScene/background: architecture view of the repository deployment path\nSubject: Flask API gateway, vLLM services, GPU-hosted VLM model instances, and poc/work clients\nStyle/medium: clean technical infographic\nComposition/framing: 16:9 horizontal layout with three layers: clients and operators on the left, Flask API gateway in the center, upstream vLLM services and GPU-backed model instances on the right\nLighting/mood: flat presentation-ready lighting\nColor palette: charcoal, warm white, amber, teal, muted red accents\nConstraints: use exact service slugs ui-venus, mai-ui, ui-tars, paddleocr-vl-1.5; show /api/vlm_serve/<service>/v1/* proxy pattern; show upstream ports 8001, 8002, 8003, 8004; show config/common.env and config/models/*.env feeding serve_vlm.py and start scripts; show poc/work clients using VLM_API_URL and VLM_MODEL_NAME; no fictional services; no watermark\nAvoid: 3D effects, gradients that reduce readability, generic cloud icons, fake terminal text, unrelated components",
  "quality": "auto",
  "size": "1536x1024"
}
```
