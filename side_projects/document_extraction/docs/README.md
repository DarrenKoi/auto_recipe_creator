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
| Kimi-K2.6 | `kimi-k2.6` | 느리지만 고품질 synthesis, ambiguity resolution, final reasoning |

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
   - 필요한 경우에만 `kimi-k2.6`로 final merge
4. 추출된 region을 `rag_db_plan.md` 기준으로 RAG chunk로 변환합니다.
5. `benchmark_plan.md` 기준으로 결과를 평가합니다.

## 관련 문서

- [pipeline_overview.md](./pipeline_overview.md) — **구현된 파이프라인 end-to-end 개요**
- [runbook.md](./runbook.md) — **사내 실행 절차서**(캡처 → 추출 → 검색/벤치/Marp)
- [status.md](./status.md) — **현황표**(계획 / 완료 / 필요 테스트)
- [research_notes.md](./research_notes.md)
- [pipeline_plan.md](./pipeline_plan.md)
- [rag_db_plan.md](./rag_db_plan.md)
- [benchmark_plan.md](./benchmark_plan.md)
- [marp_roundtrip_design.md](./marp_roundtrip_design.md)
