# PixelRAG 스터디 — 인덱스

이 폴더는 **PixelRAG(픽셀-네이티브 visual RAG)** 를 조사하고, 그 방법을 현재
`document_extraction` side project(사내 DRM 스크린샷 → RAG)에 적용할 수 있는지
판단한 노트 모음이다. 작성: 2026-06-26.

## 한 줄 결론

> PixelRAG 는 **retrieval recall 을 올리는 보완 기법**이지, 현재 text-extraction
> 파이프라인의 대체재가 아니다. 우리는 이미 PixelRAG 가 필요로 하는 **페이지
> 스크린샷(Stage 0)** 을 공짜로 가지고 있으므로, **pixel 검색 arm 을 하나 더 붙여
> 기존 text-chunk arm 과 RRF 로 융합**하는 hybrid 구성이 가장 비용 대비 효과가 크다.
> 단, 사내에 **vision embedding 모델 엔드포인트**가 없으므로 그게 1차 게이트다.

## 문서

| 파일 | 내용 |
|---|---|
| [`pixelrag_characteristics.md`](./pixelrag_characteristics.md) | PixelRAG 가 무엇인가 — 파이프라인, 임베딩 모델, 타일링, 검색 방식, 벤치마크, ColPali/VisRAG 와의 관계, 한계 |
| [`pixelrag_application_methods.md`](./pixelrag_application_methods.md) | 우리 프로젝트·사내 환경에 적용하는 구체적 방법 — hybrid 2-arm 검색기, 재사용 가능한 자산, 신규 모듈, 단계별 로드맵, 게이트/리스크 |

## 출처 (조사 시 확인한 1차 자료)

- PixelRAG GitHub (StarTrail-org): <https://github.com/StarTrail-org/PixelRAG>
- VisRAG (ICLR 2025), arXiv 2410.10594: <https://arxiv.org/abs/2410.10594>
- ColPali / late-interaction visual retrieval (배경): ViDoRe, ColQwen 계열
- 해설 글: The AI Automators, explainX, EveryDev (아래 본문에 링크)
