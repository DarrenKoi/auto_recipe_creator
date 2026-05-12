# 스크린샷 문서 추출 Side Project

## 목적

이 side project는 회사 내부에서 제공되는 VLM/OCR 모델로 DRM 보호 문서의 스크린샷에서 유용한 정보를 얼마나 복원할 수 있는지 검토합니다.

실질적인 목표는 RAG에 바로 사용할 수 있는 retrieval-ready database를 구축하는 것입니다. 단순 요약이 아니라, 캡처된 문서 증거에서 나중에 agent가 답변할 수 있도록 content, context, provenance, confidence를 함께 보존해야 합니다.

대상 입력은 다음 문서의 스크린샷입니다.

- PowerPoint slide
- PDF page
- Excel sheet

이 프로젝트는 screenshot-only 방식입니다. 보호된 파일을 직접 파싱하거나, DRM을 제거하거나, 접근 제어를 우회하거나, 문서 보호 기능을 우회하는 자동화를 만들지 않습니다. 목표는 화면에 이미 보이는 정보만 추출하는 것입니다.

## 사용 가능한 모델 역할

현재 repo에는 `poc/work2` 기반의 모델 서비스 구조가 이미 있습니다.

- `poc/work2/flask_vlm.py`: shared service slug, model name, endpoint mapping을 정의합니다.
- `poc/work2/vlm_client.py`: OpenAI-compatible image client를 제공합니다.
- `poc/work2/connection_check.py`: Flask proxy와 service별 `/v1/models` readiness를 확인합니다.

이 side project에서는 모델을 tiered system으로 사용합니다.

| Model | Service slug | Primary role |
| --- | --- | --- |
| PaddleOCR-VL-1.5 | `paddleocr-vl-1.5` | OCR, reading order, table, formula, chart, document parsing |
| UI-Venus-1.5-8B | `ui-venus` | 전체 스크린샷 layout 및 UI/document visual understanding |
| MAI-UI-8B | `mai-ui` | dense region 또는 작은 region의 crop-level refinement |
| Kimi-K2.5 | `kimi-k2.5` | 느리지만 고품질 synthesis, ambiguity resolution, final reasoning |

## 기대 출력

각 스크린샷에 대해 extraction pipeline은 다음을 생성해야 합니다.

- Raw OCR text
- title, body, table, chart, formula, footer, legend, notes 같은 detected region
- visible cell boundary와 text를 복원할 수 있는 경우 structured table
- visible label, axis, legend, trend 기반 chart summary
- 사람이 읽기 위한 final Markdown summary
- confidence, source region, unresolved field를 포함한 final JSON payload
- source path, screenshot/page order, region type, bbox, surrounding context, confidence를 포함한 RAG-ready chunk

여러 스크린샷 묶음에 대해서는 다음을 생성해야 합니다.

- Merged outline
- Key facts
- Table inventory
- Chart inventory
- Low-confidence review checklist
- Document/session-level search와 filtering을 위한 retrieval metadata

## 현실적인 한계

스크린샷에는 원본 Office/PDF object structure가 없습니다. Pipeline은 visible pixel에서만 추론할 수 있습니다. 약한 지점은 다음과 같습니다.

- 너무 작은 text
- 흐리거나 압축된 screenshot
- 숨겨진 Excel row 또는 column
- 잘린 cell
- 겹친 label
- visible numeric label이 없는 chart
- screenshot에 보이지 않는 speaker note 또는 comment

Large VLM은 reasoning error를 줄이는 데 도움이 되지만, 보이지 않는 정보를 복원할 수는 없습니다.

## 권장 첫 실험

1. `uv run python poc/work2/connection_check.py`를 실행해 사용 가능한 service를 확인합니다.
2. 작은 local screenshot set을 준비합니다.
   - PowerPoint screenshot 3개
   - PDF page screenshot 3개
   - Excel screenshot 3개
3. 수동 또는 minimal script로 extraction을 실행합니다.
   - 먼저 `paddleocr-vl-1.5`
   - 그다음 `ui-venus`
   - `mai-ui` 또는 `paddleocr-vl-1.5`로 crop retry
   - 필요한 경우에만 `kimi-k2.5`로 final merge
4. 추출된 region을 `rag_db_plan.md` 기준으로 RAG chunk로 변환합니다.
5. `benchmark_plan.md` 기준으로 결과를 평가합니다.

## 관련 문서

- [research_notes.md](./research_notes.md)
- [pipeline_plan.md](./pipeline_plan.md)
- [rag_db_plan.md](./rag_db_plan.md)
- [benchmark_plan.md](./benchmark_plan.md)
- [../docs/screenshot_document_extraction_pipeline_plan.md](../docs/screenshot_document_extraction_pipeline_plan.md) — 단계별 capture → extract → organize 계획

## 단계별 실행 (capture → extract → organize)

세 단계는 모두 `logs/pipeline_state.json` 원장(ledger)을 공유한다.
중간에 멈춰도 같은 명령을 다시 실행하면 끝낸 페이지를 건너뛰고 이어서 처리한다.

```bash
# 0) 의존성 설치
uv sync --extra dev
# Windows 에서 Office 자동화도 같이 쓰려면:
# uv pip install ".[windows]"

# 1) 입력 파일을 data/inputs/ 에 떨어뜨린 뒤 캡처 실행
uv run python side_projects/screenshot_document_extraction/run_capture.py
# → data/captures/<doc_id>/page_001.jpg ...

# 2) 캡처된 이미지를 VLM + OCR 로 추출
uv run python side_projects/screenshot_document_extraction/run_extract.py
# → data/extracted/<doc_id>/page_NNN.{paddleocr,uivenus,raw}.json

# 3) LLM 이 읽기 좋게 정리
uv run python side_projects/screenshot_document_extraction/run_organize.py
# → data/organized/<doc_id>/document.md + document.json + pages/

# 또는 세 단계 한 번에:
uv run python side_projects/screenshot_document_extraction/run_all.py
```

지원 확장자: `.pptx/.ppt`, `.docx/.doc`, `.xlsx/.xls`, `.pdf`.

폴더 의미:
- `data/inputs/` — 사용자가 원본 파일을 떨어뜨리는 곳.
- `data/captures/<doc_id>/` — 페이지별 JPEG 와 메타 JSON.
- `data/extracted/<doc_id>/` — paddleocr/ui-venus 응답과 머지 raw.json.
- `data/organized/<doc_id>/` — LLM 친화적 Markdown 과 JSON.

원장(`logs/pipeline_state.json`)을 직접 보면 각 doc 의 단계별 상태(`pending|in_progress|done|failed`)와
완료된 페이지 번호를 확인할 수 있다. 실패한 항목을 다시 시도하려면
`SCREENSHOT_EXTRACTION_RETRY_FAILED=true` 환경변수로 실행한다.
