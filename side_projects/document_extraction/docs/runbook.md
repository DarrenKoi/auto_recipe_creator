# 문서 추출 side project — 오피스 실행 런북

> 대상: `side_projects/document_extraction/` 전체 파이프라인(Stage 0 캡처 -> Stage 1~8
> 추출 -> 검색/벤치/Marp 복원).
> 목적: 사내 PC(Windows)에서만 가능한 단계(스크린샷 캡처 · 실제 VLM 호출 · marp-cli
> 렌더 · ground-truth 작성)를 **순서대로, 막힘 없이** 돌리는 절차서.
> 설계/개요 선행 문서: [`pipeline_overview.md`](./pipeline_overview.md)(구현된 end-to-end),
> [`marp_roundtrip_design.md`](./marp_roundtrip_design.md)(Stage 5/6/7),
> [`benchmark_plan.md`](./benchmark_plan.md), [`rag_db_plan.md`](./rag_db_plan.md).

집(사외 dev PC)에서 검증 끝난 것 / 사내에서만 가능한 것의 경계는 `pipeline_overview.md` §10
에 있다. 이 런북은 **사내에서 실제로 해야 하는 일**만 다룬다.

---

## 0. 한눈에 (TL;DR)

```
0) 사전: uv sync,  VLM 서버 readiness 확인,  (Marp 쓸 거면) marp-cli + Chromium 설치
1) 캡처:  extract.py 상단 INPUT_DIR/OUTPUT_DIR 채우고 실행 -> page_NNN.webp
2) 추출:  extract_screenshot.py 상단 INPUT_IMAGE_DIR/OUTPUT_DIR 채우고 실행
          -> raw_evidence/*.json + rag_chunks.jsonl     (서버 죽으면 DOC_EXTRACT_OFFLINE=1 로 골격만)
3) 소비:  search(검색) / run_benchmark(채점) / build_marp(deck) / verify_and_downgrade(Stage 7)
4) 회신:  각 단계 콘솔의 [INFO] 요약 줄 + (벤치) comparison_matrix.md + (Marp) 강등 카운트
```

모든 스크립트는 **CLI 인자 없음** — 매 실행 전 모듈 상단 상수를 직접 수정한다(프로젝트 규약).

---

## 1. 사전 준비 (1회)

### 1-1. 파이썬 환경
```bash
uv sync --extra dev
```

### 1-2. VLM 서버 readiness (Stage 2~6 실호출용)
추출은 사내 Flask VLM proxy + 4개 서비스를 쓴다(`pipeline_overview.md` §8).
- 헬스: `GET {FLASK_BASE}/api/vlm_serve/health` 가 `paddleocr-vl-1.5 / ui-venus / mai-ui /
  kimi-k2.6` 를 `ready` 로 보고하는지 확인.
- 클라이언트: `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient`(service slug 으로 호출).
- **서버에 못 붙으면** Stage 1~8 골격이 OFFLINE 폴백으로 도므로(§6) 추출 자체는 멈추지
  않지만, evidence 가 stub 이라 정확도 평가는 무의미하다. 실측 전 반드시 ready 확인.

### 1-3. (Marp Stage 6/7 쓸 때만) marp-cli + Chromium
```bash
# 둘 중 하나:
npm i -g @marp-team/marp-cli      # 전역 설치(권장, PATH 의 `marp`)
# 또는 설치 없이 npx 폴백:  resolve_marp_command() 가 npx --yes @marp-team/marp-cli 사용
```
> **중요(오피스 함정):** marp-cli 의 `--images png` / `--pdf` / `--pptx` export 는 내부적으로
> **headless Chromium(puppeteer)** 을 띄운다. 락다운된 사내 PC 에서 Chromium 자동 다운로드가
> 막히면 렌더가 실패한다. 이미 깔린 Chrome 을 쓰려면 환경변수로 지정:
> ```bash
> set CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe   # Windows
> ```
> 렌더가 안 되면 `render_deck` 은 예외 없이 `available`/`ok=False` 로 빠지고(graceful),
> Stage 7 검증은 자동으로 건너뛴다 — 깨지진 않지만 **충실도 검증을 못 한다**.
> HTML(`--html`)만은 Chromium 불필요 — Chromium 이 안 되면 우선 HTML 로 sanity check.

---

## 2. Stage 0 — 캡처 (Windows COM, 사내 전용)

`extract.py` 상단 상수를 채우고 실행한다.

| 상수 | 의미 |
|---|---|
| `INPUT_DIR` | PPT/Excel/Word/PDF 가 든 폴더 |
| `OUTPUT_DIR` | 파일명별 하위 폴더에 page 이미지 저장할 루트 |
| `OVERWRITE` | 기존 출력 폴더 덮어쓰기(기본 스킵) |
| `RECURSIVE` | 하위 폴더 재귀 탐색(기본 비재귀) |

```bash
uv run python side_projects/document_extraction/extract.py
```
- 산출: `<OUTPUT_DIR>/<파일stem>/page_NNN.webp`(페이지별 WebP, 1MB 캡).
- 확인: 콘솔 `[INFO] 완료 - 성공: N, 실패: M, 스킵: K`. 실패 파일은 traceback 으로 표시.
- 제약: PowerPoint/Excel/Word 는 Windows COM(설치된 Office) 필요, PDF 는 PyMuPDF.
  **DRM/접근제어 우회 없음** — 화면에 보이는 페이지만 캡처.

---

## 3. Stage 1~8 — 추출 (evidence + RAG chunk)

`extraction/extract_screenshot.py` 상단 상수를 채운다.

| 상수 | 기본 | 의미 |
|---|---|---|
| `INPUT_IMAGE_DIR` | `""` | Stage 0 가 만든 한 문서의 page 이미지 폴더 |
| `OUTPUT_DIR` | `""` | evidence/chunk 출력 루트(관례: `<문서>/_rag`) |
| `COLLECTION_ID` | `default_collection` | retrieval 묶음 식별자 |
| `ENABLE_CROP_REFINE` | `False` | Stage 4 crop 재인식(오피스 캘리브레이션 후 켠다) |
| `SYNTHESIS_MODE` | `deterministic` | `deterministic`(0콜) / `kimi`(고품질·느림) / `none` |
| `OFFLINE` | `None` | `None`=env 결정, `True/False` 강제 |

```bash
# 서버 ready 일 때(실측)
uv run python -m side_projects.document_extraction.extraction.extract_screenshot

# 서버 없이 골격만(집/사외, 또는 서버 점검 중)
DOC_EXTRACT_OFFLINE=1 uv run python -m side_projects.document_extraction.extraction.extract_screenshot
```
- 산출:
  - `<OUTPUT_DIR>/raw_evidence/<screenshot_id>.json` — `ExtractionResult` 전체(사람 검토용).
  - `<OUTPUT_DIR>/rag_chunks.jsonl` — retrieval store(append).
- **실데이터를 읽고 있나?** 확인:
  - `summary_model_sources` 가 `["deterministic"]`/`["kimi-k2.6"]` 인지(= 실제로 어떤
    경로로 합성됐는지 정직하게 기록됨). OFFLINE 폴백이면 stub 흔적이 보인다.
  - `region.conflicts` / `unresolved` 에 충돌이 쌓이면 모델 출력이 엇갈린 것 — 검토 대상.
- 권장: 먼저 `SYNTHESIS_MODE=deterministic`(값 창작 위험 0)로 한 바퀴, 품질이 부족한
  문서만 `kimi` 로 재합성(latency 큼).

---

## 4. 후속 소비

### 4-1. 검색 (RAG sanity)
```bash
uv run python -m side_projects.document_extraction.extraction.search
```
- `quality_gate` 가 chunk 를 `trusted` / `lower_trust` 로 분류, `keyword_search` 가 토큰
  AND + metadata 필터로 랭킹. embedding 없이 도는 1차 검색이라 사내/사외 무관.

### 4-2. 벤치 채점 (정확도 — ground-truth 필요)
1. 사내에서 9장(권장: PPT 3 / PDF 3 / Excel 3) 캡처 + **사람이 ground-truth JSON 작성**.
2. 4개 파이프라인 추출 결과를 `extractions/<pipeline>/` 에 배치.
3. 채점(채점 자체는 순수 Python — 어디서나):
```bash
uv run python -m side_projects.document_extraction.benchmark.run_benchmark
```
- 산출: `scores.json`, `comparison_matrix.md`, `acceptance.json`.
- 메트릭: Text Recall / Table Accuracy / Chart Understanding / Layout Accuracy /
  Hallucination Rate / RAG Readiness / Latency (`benchmark_plan.md`).
- **회신**: `comparison_matrix.md` 본문 + `acceptance.json` 의 통과/미통과.

### 4-3. Marp 복원 — Stage 5 (deck 생성)
`marp/build_marp.py` 상단 `RAW_EVIDENCE_DIR` / `OUTPUT_MD`(+ 선택 `CROP_LOOKUPS`) 채우고:
```bash
uv run python -m side_projects.document_extraction.marp.build_marp
```
- 산출: `deck.md`(텍스트류 네이티브 + 차트는 crop 재삽입 또는 데이터표 대체).

### 4-4. Marp 렌더 + 검증/자동 강등 — Stage 6/7 (신규)
`build_marp` 가 만든 `deck.md` 와 **원본 캡처(page_NNN.webp)** 를 슬라이드 순서대로 묶어
`verify_and_downgrade` 를 한 번 호출한다(이 함수가 내부에서 렌더까지 한다 — deck 을 따로
미리 렌더하지 말 것). 작은 드라이버 예:

```python
# run_marp_verify.py (상단 상수만 채워 uv run python run_marp_verify.py)
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from side_projects.document_extraction.extraction.schemas import ExtractionResult
from side_projects.document_extraction.marp.verify import verify_and_downgrade

RAW_EVIDENCE_DIR = Path(r"")   # <문서>/_rag/raw_evidence
DECK_MD          = Path(r"")   # build_marp 가 만든 deck.md
CAPTURE_DIR      = Path(r"")   # Stage 0 page_NNN.webp 폴더
OUT_DIR          = Path(r"")   # 렌더 png + deck_corrected.md 출력 폴더

results = sorted(
    (ExtractionResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
     for p in RAW_EVIDENCE_DIR.glob("*.json")),
    key=lambda r: r.screenshot_index)
captures = [str(p) for p in sorted(CAPTURE_DIR.glob("*.webp"))]   # 슬라이드 순서와 일치해야 함

report = verify_and_downgrade(results, DECK_MD, captures, out_dir=OUT_DIR)
print("[RESULT] rendered=%s flagged=%s" % (report["rendered"], report["flagged"]))
print("[RESULT] scores=", [round(s, 4) for s in report["scores"]])
print("[RESULT] corrected_deck=", report["corrected_deck"])
```

동작:
- deck 을 `--images png` 로 렌더 -> 슬라이드별 PNG.
- 각 PNG vs 원본 캡처 SSIM(`slide_fidelity`, 해상도 자동 보정).
- floor(`DEFAULT_SSIM_FLOOR=0.90`) 미만 슬라이드를 자동 강등:
  - 차트 crop 가용 -> 그 영역만 래스터로(편집성 보존),
  - 살릴 게 없으면 슬라이드 전체를 원본 캡처 래스터(`![bg]`)로 = **최후 안전망**.
- 강등 반영한 `deck_corrected.md` 를 `OUT_DIR` 에 기록.

**확인 / 회신:**
- `report["rendered"]` 가 `False` 면 marp/Chromium 문제(§1-3) — 렌더 자체가 안 된 것.
- `report["scores"]` 분포(슬라이드별 SSIM) + `report["flagged"]`(강등된 인덱스) 개수.
- 첫 실데이터 회신 후 **floor(0.90) 보정**: SSIM 이 폰트 렌더 차이로 전반적으로 0.90 근처면
  너무 많이 강등될 수 있다. `verify_and_downgrade(..., threshold=0.85)` 처럼 낮춰 재측정.

---

## 5. 캘리브레이션 대상 (첫 실데이터 후)

| 항목 | 어디서 | 무엇을 본다 |
|---|---|---|
| SSIM floor(0.90) | Stage 7 `threshold` | 슬라이드별 SSIM 분포 — 강등 빈도가 과하면 0.85~0.88 로 |
| crop 분기(Stage 3) | `generate.py` 분기 규칙 | 어느 region type 을 선제적으로 래스터로 강등할지 |
| `ENABLE_CROP_REFINE` | extract_screenshot 상수 | Stage 4 crop 재인식 켠 뒤 dense 표/차트 recall 비교 |
| synthesis 모드 | `SYNTHESIS_MODE` | deterministic vs kimi 의 요약 품질 ↔ latency |
| Kimi 엔드포인트 | VLM 게이트웨이 | base64 이미지 입력 1콜 스모크(coding 엔드포인트엔 이미지 없음) |

---

## 6. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| 추출은 되는데 evidence 가 빈약/이상 | OFFLINE 폴백으로 돌았을 가능성 — VLM health 확인, `DOC_EXTRACT_OFFLINE` 해제 |
| `report["rendered"]=False` | marp-cli 미설치(`marp`/`npx` 둘 다 없음) 또는 Chromium 부재 — §1-3 |
| 렌더 PNG 수 < 슬라이드 수 | 빈 슬라이드가 deck 에서 제외됐거나 렌더 부분 실패 — 부족분은 SSIM 0.0 처리되어 강등됨 |
| 모든 슬라이드가 강등됨 | floor 가 너무 높음(폰트 렌더 차이) — `threshold` 낮춰 재측정(§4-4) |
| 표가 깨져 보임 | 셀 `|`/줄바꿈 escape 는 되어 있음 — 원본이 병합셀이면 데이터표 대체가 한계, 차트처럼 래스터 강등 고려 |
| Excel 숨은 행/열 누락 | 스크린샷에 안 보이면 복원 불가(설계 한계) — ground-truth 도 보이는 것만 |

---

## 7. 스모크 테스트 (서버/마프 불필요 — 사외에서도 회귀 확인)

```bash
uv run python -m side_projects.document_extraction.extraction.test_extraction_smoke
uv run python -m side_projects.document_extraction.extraction.test_retrieval_smoke
uv run python -m side_projects.document_extraction.benchmark.test_benchmark_smoke
uv run python -m side_projects.document_extraction.marp.test_marp_smoke      # Stage 5
uv run python -m side_projects.document_extraction.marp.test_render_smoke    # Stage 6 (순수 인자/그레이스풀)
uv run python -m side_projects.document_extraction.marp.test_verify_smoke    # Stage 7 (SSIM/강등 결정)
```
전부 통과해야 코드 회귀가 없다(현재 Stage 5/6/7 = 8/6/11 통과). 실제 모델·marp 렌더는
이 스모크 범위 밖 — 정확도/충실도는 위 §3~4 의 사내 실행으로만 확정된다.
```
