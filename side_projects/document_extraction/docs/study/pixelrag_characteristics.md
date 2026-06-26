# PixelRAG 특성 정리

> 출처: PixelRAG GitHub(StarTrail-org, Apache-2.0, Berkeley Sky Computing Lab/BAIR/
> Berkeley NLP), 동반 논문 *"PixelRAG: Web Screenshots Beat Text for
> Retrieval-Augmented Generation"*, 및 해설 글(아래 링크). 2026-06 기준.

## 1. 한 문장 정의

PixelRAG 는 문서를 **텍스트로 파싱하지 않고 스크린샷 타일(image)로 렌더 → vision
embedding 으로 인덱싱 → 질의 시 관련 타일을 검색 → VLM 이 그 이미지에서 직접 답을
읽는** **pixel-native retrieval** 시스템이다. 즉 "검색 단위가 텍스트 chunk 가 아니라
**이미지 타일**"이다.

핵심 가설(논문):

> **텍스트 추출은 layout, table, figure, styling 을 버린다.** 그 신호가 바로 페이지를
> 읽을 수 있고 답할 수 있게 만드는 신호인데, text RAG 는 그걸 인덱싱 전에 날린다.

## 2. 파이프라인 (5단계)

GitHub README 기준 단계:

```
Render ─▶ Chunk(Tile) ─▶ Embed ─▶ Build-Index ─▶ Serve
```

| 단계 | 하는 일 | 도구/모델 |
|---|---|---|
| **Render** | 웹페이지/PDF/이미지를 **고정 폭(fixed-width) 스크린샷**으로 렌더 | Playwright CDP(웹), poppler/PDF |
| **Chunk (Tile)** | 긴 이미지를 **고정 높이(fixed-height) 타일**로 슬라이스. 각 타일이 검색 단위(= text chunk 의 시각판) | 내부 타일러 |
| **Embed** | 각 타일을 vision embedding 모델로 벡터화(정규화) | **Qwen3-VL-Embedding-2B** + **LoRA**(screenshot 데이터로 fine-tune) |
| **Build-Index** | 벡터를 FAISS 인덱스로 적재 | FAISS |
| **Serve** | FastAPI 검색 엔드포인트(`/search`, `n_docs` 지정). top-N 타일 반환 | FastAPI, CPU/GPU |

생성(answer)은 파이프라인 밖이다: **검색된 타일을 reader VLM(예: Claude, Kimi)이
직접 읽어 답을 만든다.** PixelRAG 자체는 "retrieve + read" 에 집중하고 별도
generation 컴포넌트를 강제하지 않는다.

## 3. 핵심 설계 선택

- **임베딩 모델**: `Qwen/Qwen3-VL-Embedding-2B` (multimodal embedding) 를 베이스로,
  webpage screenshot 데이터에 **LoRA fine-tuning**. 공개 adapter:
  `Chrisyichuan/wiki-screenshot-embedding-lora` (재학습 없이 사용 가능).
- **검색 방식**: **single-vector** retrieval (타일 1장 → 벡터 1개 → FAISS 코사인 유사도).
  README 기준 **late-interaction(MaxSim) 아님** — 이 점이 ColPali 계열과 결정적으로 다르다(§6).
- **벡터 스토어**: FAISS. 사전 빌드 Wikipedia 인덱스 = 약 8.28M article / 28.1M 타일,
  인덱스 크기 ~217GB. 호스티드 API: `https://api.pixelrag.ai`.
- **플랫폼**: Linux(CUDA) / macOS(MPS) / CPU 폴백. 소규모 인덱스는 GPU 없이도 추론 가능.
- **라이선스**: Apache-2.0 (self-host 자유). Claude Code 플러그인(`pixelbrowse`/`pixelshot`)도 제공.

## 4. 보고된 성능 (논문/해설)

| 지표 | 수치 | 비고 |
|---|---|---|
| 정확도 | text RAG 대비 **+18%** (aggregate) | layout-heavy 문서(표/인포그래픽/figure PDF)에서 gap 더 큼, 순수 prose 에선 작음 |
| 토큰 절감 | **약 3x ~ 10x** | Wikipedia-scale 테스트: text RAG 37.5M prompt token → pixel 3.6M (≈10x). "fewer searches, less history, fewer passes" |
| Parser loss 개선 | 놓친 답의 **1/3 이상이 parser loss** 에 기인, PixelRAG 가 **36.6%** 개선 | text RAG 가 파싱 단계에서 답을 잃는 케이스를 pixel 이 회수 |

## 5. 명시된 한계 / 약점

- **Reader 모델 크기 의존**: vision 모델이 **4B 미만**이면 벤치 정확도 **12.5점 이상**
  하락. frontier 급(예: Claude Opus)이라야 타일을 제대로 읽는다 → **reader 는 작은 모델 금지**.
- **인제스트 비용(대규모)**: 7M Wikipedia 페이지 인덱싱에 **800 CPU × 2일, 5TB** 소요.
  (단, 소규모 사내 corpus 에는 무관 — §application 참고.)
- **순수 prose / linear 문서**: 레이아웃 신호가 적은 "straightforward" 소스에선
  **plain text RAG 가 여전히 우세**. 즉 만능이 아니다.
- **Chrome/Chromium 의존**(웹 렌더 경로). PDF 경로는 poppler.
- **생성 미포함**: 검색 결과는 타일(이미지). 답은 downstream reader 가 해석해야 함 →
  provenance/confidence/structured field 는 PixelRAG 가 주지 않는다(우리 프로젝트엔 중요, §application).

## 6. 주변 기법과의 관계 (visual RAG 지형)

PixelRAG 는 "document-image 를 직접 임베딩한다"는 **visual RAG** 계열의 한 구현이다.
같은 아이디어의 대표 연구와 비교:

| 기법 | 검색 표현 | 매칭 | 특징 / 트레이드오프 |
|---|---|---|---|
| **VisRAG** (ICLR 2025) | 문서 이미지 1장 → **single 벡터** | dense 코사인 | text RAG 대비 e2e **+20~40%**. 페이지=1벡터, 단순/저장 작음 |
| **ColPali / ColQwen** | 페이지를 **patch 다중 벡터** | **late-interaction (MaxSim)** | patch 단위 localization 우수, ViDoRe SOTA. 단 인덱스 크고 무거움 |
| **PixelRAG** | 페이지를 **타일로 잘라 타일당 single 벡터** | dense 코사인 | VisRAG 의 "single-vector" 단순함 + 타일 슬라이싱으로 긴 문서 granularity 보강. late-interaction 안 씀 → 인덱스 가볍고 빠름 |

요점: **PixelRAG = "VisRAG의 단순 single-vector 검색" + "긴 페이지를 타일로 쪼개
granularity 확보"**. ColPali 처럼 patch-level MaxSim 의 정밀 localization 은 포기하는
대신, 인덱스/추론이 가볍다. (우리처럼 corpus 가 작으면 ColQwen late-interaction 도
감당 가능 — §application 의 모델 선택에서 다룬다.)

## 7. 우리 프로젝트 관점의 시사점 (요약)

1. **입력이 이미 맞다**: 우리는 DRM 때문에 어차피 페이지를 스크린샷(`page_NNN.webp`)으로
   뜬다. PixelRAG 의 Render 단계가 우리 Stage 0 와 동일 — 추가 캡처 비용 0.
2. **우리의 약점을 정조준**: `rag_db_plan.md` 가 걱정하는 "표/차트가 텍스트로 flatten
   되며 맥락 손실"(예: `Yield improved 12%`)이 바로 PixelRAG 가 회수하는 parser-loss
   케이스다.
3. **하지만 대체 불가**: 우리는 content+**context+provenance+confidence**+Marp roundtrip 을
   보존해야 한다. PixelRAG 는 그걸 안 준다 → **보완(2nd retrieval arm)** 으로만 타당.
4. **게이트는 임베딩 모델**: 사내 4서비스(paddleocr/ui-venus/mai-ui/kimi)는 전부
   **generative VLM 이지 embedding 엔드포인트가 아니다.** vision embedding 모델 확보가
   적용의 1차 조건이다(§application §3).

## 출처

- PixelRAG GitHub: <https://github.com/StarTrail-org/PixelRAG>
- The AI Automators 해설: <https://www.theaiautomators.com/pixelrag-visual-rag-without-text-parsing/>
- explainX 해설: <https://www.explainx.ai/blog/pixelrag-visual-rag-web-screenshots-berkeley-guide-2026>
- VisRAG (ICLR 2025): <https://arxiv.org/abs/2410.10594>
- RegionRAG / patch-to-region 등 후속(배경): arXiv 2510.27261, 2512.02660
