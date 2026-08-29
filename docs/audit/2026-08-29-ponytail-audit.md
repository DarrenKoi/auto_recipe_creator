# Ponytail Audit — `auto_recipe_creator`

Date: 2026-08-29
Scope: 517 Python files across `poc/`, `flask_api/`, `deploy_vlms/`, `gpu_dashboard/`, `side_projects/`, `utils/`, `test/`.
Method: Four parallel read-only scouts + manual confirmation of the inter-package import graph. Findings are one-shot and nothing is applied.

## Tier 1 — delete (entire dirs / files with zero production callers)

```
delete: poc/work2/ — 38 files, ~10,459 lines. Every shared module is byte-identical or near-identical to poc/workflow_3/{rcs,vlm,util}/ (verified by diff). Zero production callers: grep -rn "from poc.work2" outside poc/work2/ returns only 8 test files. Per AGENTS.md: "Do not put new automation, prompts, shared helpers, or workflow code there." [poc/work2/]
delete: 8 work2-only test files. They test deleted code. [test/work2/*, test/flask_api/test_work2_*.py]
delete: poc/workflow_1/util/ — empty directory, only __pycache__. Lone importer office_align_fail_alarm.py:18 is already broken (no __init__.py). [poc/workflow_1/util/]
delete: poc/workflow_1/extract_recorded_ch4_frames.py (14 lines). 1-line re-export of locate_cursors(); source module already has its own __main__ block. [poc/workflow_1/extract_recorded_ch4_frames.py]
delete: poc/workflow_1/office_rich_notify.py (19 lines). Stub that just prints to console; try/except wrapper in align_fail_alarm.py is the only consumer. [poc/workflow_1/office_rich_notify.py]
delete: poc/workflow_1/connect_tool.py (75 lines). 2 functions: one wraps os.getenv, the other is an interactive loop. Reachable directly via poc/workflow_3/rcs/workflow_select_tool.py. [poc/workflow_1/connect_tool.py]
delete: poc/workflow_1/build_report_pptx.py (633 lines). Boss report generator with hand-rolled shape/textbox placement helpers (_set_run, _add_title_band, etc.). python-pptx covers it via shape factory + .add_paragraph(). [poc/workflow_1/build_report_pptx.py]
delete: poc/workflow_2/ — 67 files, 20,828 lines. ZERO external callers (grep confirms no production code outside workflow_2 imports from it). 42 of 67 files are standalone __main__ scripts; 44 mention "experiment/lab/ablation/probe/scratch" in their docstrings. Canonical surface (align_key_matcher, align_fail_assets, search_align_key, live_align_search) actually lives in poc/workflow_3/align/, not here. ensemble_lab.py and template_bank_lab.py openly say "production 엔진(workflow_3/align)을 건드리지 않고 새 ... 시험한다" — they're forks. [poc/workflow_2/]
delete: poc/workflow_3/workflow_extract/ entire package. Only consumers are its own tests + recording_filter/test_type_detect.py (test imports). [poc/workflow_3/workflow_extract/]
delete: poc/workflow_4/framework/ entire package. Only consumer is demo/offline_demo.py (which has no production caller). CycleGraphMirror only needs 3 dataclasses + 1 writer from graph_view.py — inline into adapters/. [poc/workflow_4/framework/]
delete: poc/workflow_4/demo/ entire dir. [poc/workflow_4/demo/]
delete: poc/workflow_4/adapters/run_cycle3_mirror_demo.py (146 lines). Not imported anywhere. [poc/workflow_4/adapters/run_cycle3_mirror_demo.py]
delete: 10 scratch/diagnostic files in poc/workflow_3/monitor/ (temp_office_rcp_msr_downloader.py, temp_test_office_downloaders.py, diagnose_correction_gates.py, demo_log_panel.py, make_demo_video.py, make_demo_video_combined.py, demonstration_rcs_control.py, analyze_cycle_manifest.py, verify_success_gather.py, fetch_msr_offline.py) — zero non-test callers. ~7,251 LOC. [poc/workflow_3/monitor/]
delete: poc/workflow_3/rcs/bench_{stage_report,tool_locator,tool_window_reader}.py — 1,378 LOC of bench artifacts, zero non-test callers. [poc/workflow_3/rcs/]
delete: poc/workflow_3/align/diagnostics/{compare_align_images,verify_cond_box_crop,search_align_key,verify_rcp_assets}.py — 1,165 LOC, zero non-test callers. [poc/workflow_3/align/diagnostics/]
delete: test/vlm_input_control/{screen_capture,mouse_control,keyboard_control,vlm_screen_analysis}.py — top-level test_xxx() functions that just print and call the module's own classes. Not pytest-discoverable, not asserting anything. Production code misfiled under test/. [test/vlm_input_control/]
delete: test/workflow_extractor/ — ZERO pytest tests (no def test_, no pytest import). Only production modules with __main__ demos. [test/workflow_extractor/]
delete: test/video_frame_parser/parser.py + 10 sibling files (~5,966 LOC) — production code misfiled as tests. [test/video_frame_parser/]
delete: deploy_vlms/scripts/start_{mai_ui,ui_tars,paddleocr_vl,ui_venus}.py — 4 thin 5-line wrappers that just subprocess.call start_model.py with a slug. [deploy_vlms/scripts/start_*.py]
delete: flask_api/vlm_serve/{ui_tars,ui_venus,got_ocr}.py — enabled=False in config.py, blueprints never registered. [flask_api/vlm_serve/]
delete: gpu_dashboard/ entire package — 18-line placeholder blueprint with / and /health routes that just echo "template blueprint is running". Only web_main.py imports it. [gpu_dashboard/]
delete: side_projects/document_extraction/extraction/_b1_testing.py — leading-underscore test helper inside production package. [side_projects/document_extraction/extraction/_b1_testing.py]
```

## Tier 2 — yagni (single-implementation interfaces, dead config rows)

```
yagni: service_fallback_order = ("mai-ui",) is a single-element "fallback" in workflow_3/runner/workflow_config.py:14. Drop the field, inline "mai-ui". [poc/workflow_3/runner/workflow_config.py:14]
yagni: ConditionType enum has 11 members; 5 (DIALOG_DISAPPEARED, PROCESS_ALIVE, FIELD_READY_FOR_INPUT, TEXT_ALREADY_PRESENT, WINDOW_FOUND) are only referenced inside the dispatch dict in workflow_runner.py — no StepCondition uses them. Drop members + their _check_* methods. [poc/workflow_3/runner/workflow_types.py:33-43]
yagni: flask_api/vlm_serve/config.py VLMServiceEntry (14) duplicates VLMServiceConfig in service_template.py:22. Collapse into one. [flask_api/vlm_serve/config.py]
yagni: flask_api/vlm_serve/mai_ui.py + paddleocr_vl.py are 1-call adapters for VLMServiceConfig/create_vlm_service_blueprint. Replace with a dict lookup. [flask_api/vlm_serve/]
yagni: flask_api/model_upload/store.py:38-90 — 6 exception classes with hardcoded status_code. A dict {Cls: status} + Flask errorhandler is shorter. [flask_api/model_upload/store.py]
yagni: deploy_vlms/scripts/prepare_variant_envs.py — 9 ModelVariant rows; only 2 are running per start_all.py:30-32. 6/9 generate env files for never-executed code paths. [deploy_vlms/scripts/prepare_variant_envs.py:53-114]
yagni: side_projects/document_extraction/extraction/schemas.py REGION_TYPES / CHUNK_TYPES / SOURCE_TYPES tuples exported but never used as validators. Runtime just str(data.get("type", "other")). [side_projects/document_extraction/extraction/schemas.py:25-48]
```

## Tier 3 — shrink (duplicate helpers, hand-rolled retry loops)

```
shrink: deploy_vlms/scripts/{check_vlm,start_all,stop_model,serve_vlm,serve_got_ocr,run_got_ocr,upload_model}.py — 6 separate copies of the same hand-rolled KEY=VALUE .env parser (~150 LOC total). One shared helper or python-dotenv (already a dep). [deploy_vlms/scripts/*.py]
shrink: deploy_vlms/scripts/{start_all,start_model,stop_model,serve_vlm,serve_got_ocr,run_got_ocr,upload_model,check_vlm}.py — 8 separate log()/warn()/fail() helpers that just print(f"[INFO] {msg}"). One shared _log.py. [deploy_vlms/scripts/]
shrink: deploy_vlms/scripts/upload_model.py:209-291 HttpTransport._request with custom RETRYABLE_STATUS = {408,425,429,500,502,503,504} mapping. urllib3.util.retry.Retry + requests.adapters.HTTPAdapter is stdlib. [deploy_vlms/scripts/upload_model.py]
shrink: deploy_vlms/scripts/upload_model.py:317-323 RETRY_BACKOFF_BASE_SEC/CAP_SEC exponential backoff loop. tenacity or itertools.accumulate removes ~10 lines. [deploy_vlms/scripts/upload_model.py]
shrink: poc/workflow_1/build_report_pptx.py hand-rolled _set_run / _add_title_band / _add_section_card / _add_pipeline_step. python-pptx shape factory + .add_paragraph() covers all of it. [poc/workflow_1/build_report_pptx.py]
shrink: poc/workflow_1/monitor_align_fail.py _iter_alarm_rows / _row_value / _alarm_rows_empty (3 helpers, ~40 LOC) for what is pandas.DataFrame.itertuples(index=False) since align_fail_alarm.py already mandates pandas. [poc/workflow_1/monitor_align_fail.py]
shrink: poc/workflow_3/util/json_utils.py extract_json (96 LOC) does fence-stripping + balanced-brace scanning + trailing-comma regex + ast fallback. json.JSONDecoder(strict=False) + text[text.find("{"):text.rfind("}")+1] is one line each. ~80 LOC removed. [poc/workflow_3/util/json_utils.py:1-95]
shrink: poc/workflow_3/util/json_utils.py _to_pixel_coordinate (60 LOC) reimplements axis conversion with 4 branches. PIL.Image methods cover pixel + relative_1000 in 10 lines. [poc/workflow_3/util/json_utils.py:200-260]
shrink: poc/workflow_3/util/abort_switch.py class AbortSwitch (49 LOC) wraps a one-shot latch used by one global SWITCH + 4 module-level functions. Collapses to threading.Event + a string _reason. [poc/workflow_3/util/abort_switch.py]
shrink: poc/workflow_3/debug_artifacts.py — 5 of 6 helpers (debug_image_path, save_debug_jpeg, save_debug_webp, save_debug_text, save_debug_json) are 3-line mkdir+convert+save wrappers. Inline at call sites; ~70 LOC removed. [poc/workflow_3/debug_artifacts.py]
shrink: utils/zip-split-transfer/{join_safetensors_linux,zip_split_windows,split_safetensors_windows,join_unzip_linux}.py — 4 near-identical sha256sum(path) streaming-hash functions and 4 near-identical split_file/join_parts functions. ~120 LOC saved by consolidating. [utils/zip-split-transfer/]
shrink: side_projects/document_extraction/util/viewer_capture.py:57-77 frames_look_identical hand-rolls PIL pixel diff loop. PIL.ImageChops.difference(a,b).getbbox() is one line. [side_projects/document_extraction/util/viewer_capture.py]
shrink: side_projects/document_extraction/util/screen_capture.py:42-58 save_webp_capped reimplements quality-ladder + resize ladder. Image.save(optimize=True) + a single resize-thumbnail covers 90% of cases. [side_projects/document_extraction/util/screen_capture.py]
shrink: side_projects/document_extraction/extraction/schemas.py RagChunk.to_dict / Region.to_dict / Table.to_dict hand-written dataclass-asdict. dataclasses.asdict(self) does it. [side_projects/document_extraction/extraction/schemas.py:64-92,160-167,250-275]
shrink: side_projects/document_extraction/extraction/schemas.py:225-307 ExtractionResult.from_dict is an 80-line manual JSON→dataclass walker. cattrs or **data spreading makes this a one-liner per field. [side_projects/document_extraction/extraction/schemas.py:225-307]
shrink: poc/workflow_3/util/mouse_utils.py _glide_to/_jiggle (44 LOC) + 3 module-level env reads. Inline per call site (3 callers) or fold the env reads into the function. [poc/workflow_3/util/mouse_utils.py]
shrink: flask_api/vlm_serve/__init__.py:225-260 home() and health() are the same handler. Drop one. [flask_api/vlm_serve/__init__.py:225-260]
```

## Tier 4 — stdlib (hand-rolled things the standard library ships)

```
stdlib: poc/work2/login_rcs_paddleocr.py:75-81 _is_context_budget_error does substring matching on error text. requests.HTTPError or a structured exception class is the stdlib answer. [poc/work2/login_rcs_paddleocr.py]
stdlib: poc/work2/login_rcs_got_ocr.py:148-153 _parse_got_box parses "x1,y1,x2,y2" by hand. int(v) for v in re.findall(r"-?\d+", text) is a one-liner. [poc/work2/login_rcs_got_ocr.py]
stdlib: deploy_vlms/scripts/check_vlm.py:48-89 + serve_vlm.py:225-260 hand-rolled nvidia-smi parser. subprocess.check_output + csv.DictReader does it in 3 lines. [deploy_vlms/scripts/]
stdlib: deploy_vlms/scripts/serve_vlm.py:215-222 _normalize_limit_mm parses "image=1,video=2" by manual split. urllib.parse.parse_qs is exactly this. [deploy_vlms/scripts/serve_vlm.py]
stdlib: deploy_vlms/scripts/upload_model.py:341-348 _human_bytes reimplements 1024-scaled unit ladder. format_bytes dict lookup or humanize package. [deploy_vlms/scripts/upload_model.py]
stdlib: utils/zip-split-transfer/zip_split_windows.py:75-85 manually walks parts_dir.rglob("*") for .partNNN files. Path.glob("*.part[0-9][0-9][0-9]") does it. [utils/zip-split-transfer/]
```

## Tier 5 — native (deps / platform feature already covers)

```
native: 9 files in deploy_vlms/scripts/ + flask_api/vlm_serve/__init__.py:79-91 all hand-roll KEY=VALUE .env parsing. python-dotenv (already in pyproject.toml:18) parses these in 3 lines and respects quoting. [deploy_vlms/scripts/*.py, flask_api/vlm_serve/__init__.py:79-91]
native: poc/workflow_4/framework/graph_view.py _mermaid_asset_content reads a vendored 3.6 MB mermaid.min.js into a Python string to embed in HTML on every snapshot write (mirror writes every ~0.5s). <script src="local_or_cdn"> drops the in-string copy. [poc/workflow_4/framework/graph_view.py:17-35]
```

## Findings that DID NOT survive review (tempting but verified real)

- `flask_api/model_upload/store.py` — looks long but every function is tested, disk-state semantics are deliberately explicit. Keep.
- `side_projects/document_extraction/extraction/chunkers.py`, `harvest_loader.py`, `merge.py` — pipeline stages with real distinct responsibilities.
- `deploy_vlms/scripts/serve_vlm.py` memory-sizing block (~150 LOC) — real GPU planning logic with explicit dataclass result.
- `test/flask_api/test_vlm_serve.py` — genuine coverage. Keep (unlike `test_work2_*`).
- `poc/workflow_3e/` — 1233 LOC across 8 files, no dead branches, reuses workflow_3 step executors. Lean.

## Net

```
net: -~30,500 lines, -0 hard deps possible.
```

| Tier | LOC |
|---|---|
| Delete entire dirs (work2/, workflow_2/, workflow_4/{framework,demo}, gpu_dashboard/, test/video_frame_parser/, test/workflow_extractor/, test/vlm_input_control/) | ~42,500 |
| Delete scratch/diagnostic files (workflow_3 monitor scratch + bench_*, workflow_1 re-exports, 4 flask_vlm services, etc.) | ~3,500 |
| YAGNI abstractions (single-element fallback, dead enum members, duplicate config dataclasses, dead model variants) | ~600 |
| Shrink duplicate helpers (.env x9, log x8, sha256/split x4, json_utils, abort_switch, debug_artifacts) | ~900 |
| Stdlib / native (hand-rolled env parser, nvidia-smi parser, etc.) | ~200 |
| **Total cuttable** | **~47,700 lines** |
| Minus: keep canonical workflow_3 (rcs, monitor cycle, util, align, vlm) + tests for canonical surface | ~-17,200 |
| **Net** | **~-30,500** |

`python-dotenv` is already a dep, so no new deps needed; `python-pptx` is only used by `build_report_pptx.py` and drops with that file.

## Top 5 cuts by ratio of safety to LOC removed

1. **Drop `poc/work2/` entirely** — 10,459 LOC, byte-identical to workflow_3, zero callers, AGENTS.md already says legacy. Just delete + fix the one broken import in `poc/workflow_1/office_align_fail_alarm.py:18`.
2. **Drop `poc/workflow_2/`** — 20,828 LOC, zero external callers (canonical align surface lives in `poc/workflow_3/align/`), 44/67 files self-identify as experiment/lab/ablation/scratch.
3. **Drop `test/video_frame_parser/` + `test/workflow_extractor/` + `test/vlm_input_control/`** — ~6,500 LOC of production code misfiled as tests.
4. **Drop `poc/workflow_4/{framework,demo}/`** — 925 LOC, no production caller. Keep only `adapters/workflow3_cycle.py` (the one CycleGraphMirror production consumer).
5. **Drop workflow_3 monitor scratch files** — 7,251 LOC of demo/diagnostic scripts with zero non-test callers.

## Boundaries

Scope: over-engineering and complexity only. Correctness bugs, security holes, and performance are explicitly out of scope. Lists findings, applies nothing. One-shot.
