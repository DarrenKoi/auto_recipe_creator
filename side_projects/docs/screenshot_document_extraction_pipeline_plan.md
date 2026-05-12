# Screenshot Document Extraction — 단계별 파이프라인 계획

대상 프로젝트: `side_projects/screenshot_document_extraction/`
브랜치: `claude/document-processing-pipeline-FCRwI`

## 배경

`screenshot_document_extraction` side project 는 이미 다음 plan 파일들을 가지고 있다.

- `README.md` — 전반적인 목적 (DRM 보호 문서에서 화면에 보이는 정보를 RAG database 로 추출)
- `pipeline_plan.md` — VLM 추출 stage 1-8 설계 (paddleocr-vl-1.5, ui-venus, mai-ui, kimi-k2.5)
- `rag_db_plan.md` — RAG chunk schema
- `research_notes.md` — 모델 capability 정리
- `benchmark_plan.md` — 평가 계획

이번 계획은 그 위에 다음 **세 가지**를 추가한다.

1. **Step 1 — capture**: Windows 에서 Office 파일을 실제로 열고 page/slide/sheet 마다 JPEG 로 저장하는 단계 (지금까지는 화면 캡처 image 가 이미 있다고 가정한 단계만 있었다).
2. **Resumability ledger**: 어느 파일이 어디까지 진행됐는지 한 곳에 기록해, VLM 이 중간에 멈춰도 다시 실행하면 이어서 처리한다.
3. **명확한 폴더 분리**: `ls` 만으로 "지금 어느 단계까지 와 있는지" 파악할 수 있도록 입력/캡처/추출/정리 출력 폴더를 분리한다.

이 첫 버전은 **Markdown + per-page JSON sidecar** 까지를 범위로 한다. RAG chunk JSONL, mai-ui crop refinement, Kimi-K2.5 synthesis, conflict resolution 은 `pipeline_plan.md` 그대로 미래 작업으로 둔다.

## 폴더 구조

```
side_projects/screenshot_document_extraction/
├── pipeline/                          # 새 Python 패키지
│   ├── __init__.py
│   ├── settings.py                    # 경로 상수, tunable, 확장자 매핑
│   ├── state_ledger.py                # 재개 가능한 JSON 원장 (atomic write)
│   ├── ids.py                         # source_path → doc_id (sha1[:12])
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── capture_runner.py          # inputs/ 순회 → 확장자별 dispatch
│   │   ├── common.py                  # PageArtifact dataclass, JPEG 저장 헬퍼
│   │   ├── powerpoint_handler.py      # COM: Slide.Export("JPG")
│   │   ├── word_handler.py            # COM: ExportAsFixedFormat → PyMuPDF
│   │   ├── excel_handler.py           # COM: 시트별 ExportAsFixedFormat → PyMuPDF
│   │   └── pdf_handler.py             # PyMuPDF page.get_pixmap()
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── extract_runner.py          # captures/ 순회 → 페이지별 VLM 호출
│   │   ├── extract_one.py             # paddleocr-vl + ui-venus per JPEG
│   │   └── prompts.py                 # ui-venus 용 document region prompt
│   └── organize/
│       ├── __init__.py
│       └── organize_runner.py         # raw.json 머지 → Markdown + JSON
├── data/                              # git 에 trackable 한 .gitkeep 만 포함
│   ├── inputs/                        # 사용자가 원본 파일을 떨어뜨리는 곳
│   ├── captures/<doc_id>/             # page_001.jpg + page_001.meta.json
│   ├── extracted/<doc_id>/            # page_NNN.{paddleocr,uivenus,raw}.json
│   └── organized/<doc_id>/            # document.md, document.json, pages/
├── logs/
│   └── pipeline_state.json            # 재개 원장 (단일 진리)
├── run_capture.py                     # entry point
├── run_extract.py                     # entry point
├── run_organize.py                    # entry point
└── run_all.py                         # capture → extract → organize 직렬 호출
```

각 entry point 는 `uv run python side_projects/screenshot_document_extraction/run_<step>.py` 로 실행한다.
CLAUDE.md 규약대로 `argparse` 는 쓰지 않으며 설정은 `pipeline/settings.py` 와 `SHARED_PIPELINE_SETTINGS` 에서 온다.

## 재개 원장 (`logs/pipeline_state.json`)

```json
{
  "version": 1,
  "updated_at": "2026-05-12T14:30:00",
  "documents": {
    "a1b2c3d4e5f6": {
      "source_path": "data/inputs/recipe_setup.pptx",
      "source_type": "powerpoint",
      "discovered_at": "...",
      "capture":  {"status": "done", "page_count": 12, "completed_pages": [1,...,12], "error": null, "finished_at": "..."},
      "extract":  {"status": "in_progress", "completed_pages": [1,2,3], "error": null, "finished_at": null},
      "organize": {"status": "pending", "finished_at": null}
    }
  }
}
```

- `doc_id` = `sha1(절대경로)[:12]` — 파일명을 바꾸면 새 doc 으로 취급된다.
- 상태값: `pending | in_progress | done | failed`.
- capture/extract 는 **페이지 단위** 진행 기록 → 중간 crash 후 재실행 시 끝낸 페이지를 건너뛴다.
- 페이지 하나 완료할 때마다 ledger 한 번 갱신 (tmp + `os.replace` 로 atomic).

## Step 1 — Capture

**원칙: native export 우선, GUI screenshot fallback.**
사용자가 "Office 설치돼 있다" 라고 했지만 목표는 "본문 내용 추출" 이므로 깨끗한 native rasterization 을 기본으로 한다. 추후 `settings.CAPTURE_MODE` 로 screenshot 모드를 켤 수 있도록 여지를 남겨둔다.

- **PowerPoint**: `win32com PowerPoint.Application` → `Presentations.Open(path, WithWindow=False)` → 슬라이드마다 `slide.Export(out, "JPG", w, h)`.
- **Word**: COM 으로 열고 `ExportAsFixedFormat(tmp_pdf, ExportFormat=17)` → PyMuPDF 로 페이지별 JPEG.
- **Excel**: COM 으로 열고 visible 시트마다 `ws.ExportAsFixedFormat(0, tmp_pdf)` → PyMuPDF 로 JPEG. 시트 하나가 여러 페이지로 분할되면 그대로 보존.
- **PDF**: 순수 PyMuPDF (`fitz.open` → `page.get_pixmap(dpi=200)`). Acrobat 없이 동작.

페이지마다 다음을 저장한다.

1. `data/captures/<doc_id>/page_NNN.jpg`
2. `page_NNN.meta.json` (`{page_index, width, height, source_path, source_type, capture_method, captured_at}`)
3. `state_ledger.mark_page_done(doc_id, "capture", N)` 즉시 호출

새 의존성:

- `pymupdf>=1.24` (cross-platform, GUI 없음)
- `pywin32>=306` (Windows-only — import 시 platform guard)

## Step 2 — Extract

`extract_runner.run()`:

1. ledger 에서 `capture.status="done"` 이고 `extract.status != "done"` 인 doc 만 처리.
2. 각 doc 의 captured page 를 `extract.completed_pages` 기준으로 skip 하며 순회.
3. 페이지마다 `extract_one.process(jpg_path)`:
   - JPEG 로드 → `encode_image_webp(image, quality=90)`
   - PaddleOCR call: `Work2VLMClient("paddleocr-vl-1.5").chat_with_image_bytes(image_bytes=..., *build_ocr_assist_prompt(w, h))` → `page_NNN.paddleocr.json`
   - UI-Venus call: 새 `build_doc_region_prompt(w, h)` (region list with `type`, `bbox`, `text`) → `page_NNN.uivenus.json`
   - `extract_json()` 로 파싱 후 `page_NNN.raw.json` 으로 머지
4. 페이지 단위 ledger 갱신. VLM 에러 시 doc 의 `extract.status="failed"` + error 기록, 다음 doc 으로 계속.

## Step 3 — Organize

`organize_runner.run()`:

1. ledger 에서 `extract.status="done"` 인 doc 만 처리.
2. `data/extracted/<doc_id>/page_*.raw.json` 페이지 순서로 머지.
3. 출력:
   - `data/organized/<doc_id>/document.md` — 페이지별 heading + OCR 텍스트 + UI-Venus region 요약
   - `data/organized/<doc_id>/document.json` — 문서 전체 메타 + 페이지 배열
   - `pages/page_NNN.json` — 페이지 단위 sidecar
4. `mark_stage(doc_id, "organize", "done")`.

## 재사용하는 utility

| 용도 | 함수 / 클래스 | 위치 |
| --- | --- | --- |
| 서비스 → URL | `resolve_service_proxy_url()` | `poc/work2/flask_vlm.py` |
| 공용 설정 dict | `SHARED_PIPELINE_SETTINGS` | `poc/work2/flask_vlm.py` |
| VLM 호출 | `Work2VLMClient(service_slug=...).chat_with_image_bytes(...)` | `poc/work2/vlm_client.py` |
| OCR prompt | `build_ocr_assist_prompt(w, h)` → `("", "OCR:")` | `poc/work2/prompts/prompt_ocr_assist.py` |
| WebP 인코딩 | `encode_image_webp(image, quality=90)` | `poc/work2/util/image_utils.py` |
| VLM JSON 파싱 | `extract_json(text)` | `poc/work2/util/json_utils.py` |
| 구조화 로그 | `log_work2_event(component=..., message=..., log_name=...)` | `poc/work2/logger.py` |
| State file 패턴 | `write_open_rcs_state(pid, status)` 참고 | `poc/work2/open_rcs.py` |

## 검증 방법

**macOS (Claude Code) — 부분 검증:**

- `uv run python side_projects/screenshot_document_extraction/run_organize.py` 를 손으로 만든 `page_001.raw.json` 으로 실행 → Markdown/JSON 생성 확인.
- PDF handler 는 PyMuPDF 만 쓰므로 mac 에서도 동작. 입력 PDF 로 `run_capture.py` 테스트 가능.
- Office COM 핸들러는 macOS 에서 import 시점에 platform guard 로 `[WARNING]` 만 찍고 doc 을 `failed` 처리.

**Windows (사용자) — 전체 검증:**

1. `pywin32`, `pymupdf` 설치 후 `uv run python poc/work2/connection_check.py` 로 proxy 도달 확인.
2. `.pptx`, `.xlsx`, `.docx`, `.pdf` 각 1개씩 `data/inputs/` 에 둔다.
3. `run_capture.py` → 페이지별 JPEG 확인.
4. 멀티페이지 PDF 처리 도중 `Ctrl-C` → 재실행 → 끝낸 페이지는 건너뛰고 이어서 처리되는지 확인.
5. `run_extract.py` → `Ctrl-C` mid-doc → 재실행 → 끝낸 페이지 skip 검증.
6. `run_organize.py` → `document.md` 가 페이지 순서대로 정렬돼 있는지 확인.

**실패 주입 테스트:**

- 손상 PDF → handler 가 raise → runner 가 doc `capture.status="failed"` 로 기록 → 다음 doc 계속.
- VLM 500 → 해당 페이지가 `completed_pages` 에 추가되지 않음 → 재실행 시 자동 재시도.

## 이번 버전에서 제외 (out of scope)

- RAG chunk JSONL 생성 (`rag_db_plan.md`)
- `mai-ui` crop refinement (Stage 4)
- `kimi-k2.5` synthesis (Stage 6)
- paddleocr/ui-venus conflict resolution (Stage 5)
- Embedding / FAISS index
- Benchmark harness (`benchmark_plan.md`)

이 첫 cut 의 capture → extract → organize 루프가 안정적으로 돌면 follow-up PR 에서 위 항목들을 단계적으로 추가한다.
