# LLM 컨텍스트 제공 아키텍처 — 연구 기반 설계 (RAG + LLM 위키)

> 목적: 추출(harvest+chunk)의 **최종 목표는 LLM 에게 최적의 컨텍스트를 주는 것**.
> 사내 스택(OpenSearch + Redis + bge-m3 / Qwen3-Embedding-8B / bge-reranker-v2-m3 /
> Kimi-K2.6)에 맞춰, 2025–2026 최신 기법을 deep-research(28 소스 / 25 claim 검증 /
> 19 confirmed / 6 refuted)로 조사해 권장 아키텍처로 정리한다. 작성 2026-06-26.

---

## 0. 한 문장 결론

**하이브리드(BM25+dense+sparse) 검색 → cross-encoder 재랭크 → "lost-in-the-middle"
대응 컨텍스트 조립**을 코어로, 각 chunk 에 **Contextual Retrieval 헤더**를 붙이고,
그림은 **인덱스 때 캡션하지 말고 답변 때 Kimi 가 원본 페이지를 직접 읽는(Deferred
Visual Ingestion)** 방식으로 처리한다. 위키는 **TOC 계층 위 RAPTOR식 재귀 요약**으로
얹어 chunk-RAG 를 보완한다.

---

## 1. 검증된 핵심 발견 (confirmed)

| # | 발견 | 신뢰 | 출처 |
|---|---|---|---|
| 1 | **Contextual Retrieval** = 최고가치 enrichment. chunk 마다 LLM 이 50–100토큰 "이 chunk 가 문서 어디에 속하는지" 컨텍스트를 생성해 **dense 임베딩과 BM25 색인 양쪽에** prepend. 검색 실패율 5.7%→3.7%(−35%)→2.9%(+BM25, −49%)→1.9%(+rerank, −67%) | high | Anthropic; arXiv 2504.19754 |
| 2 | **하이브리드 lexical+dense + RRF(k=60)** 가 단일보다 우위(R@5 0.695 vs BM25 0.644 vs dense 0.587). **에러/알람 코드·파라미터명 같은 정확토큰은 BM25 가 dense 를 이긴다** → dense 단독 금지 | high | arXiv 2604.01733 (EACL'26, 23k 질의) |
| 3 | **Cross-encoder 재랭크가 단일 최대 레버**. hybrid+rerank R@5 0.816(+12.1pp), MRR +17.2pp. bge-reranker-v2-m3 를 **마지막 단계**에 | high | arXiv 2604.01733; Anthropic |
| 4 | **표는 flatten 금지, 구조 보존**. 표-구조 불일치가 검색 실패의 **73%**. lookup 표는 **행 단위 색인**, 행·열 관계 유지 | high | arXiv 2604.01733; 2507.12425 |
| 5 | **lost-in-the-middle 대응**: 가장 관련 높은 chunk 를 질의에 **가까운 위치(꼬리)** 에 배치 | med | arXiv 2407.01219 |
| 6 | **late chunking 회피, Contextual Retrieval 우선**. late chunking 은 **bge-m3 에서 붕괴**(NDCG@5 0.070 vs early 0.246) | high | arXiv 2409.04701; 2504.19754 |
| 7 | **그림은 Deferred Visual Ingestion(DVI)**: 인덱스 때 VLM 0콜, 구조/BM25 로 후보 페이지만 찾고 **답변 때 원본 페이지 이미지+질문을 Kimi 에 전달**. 사전 캡션보다 기술문서 QA에서 +41pp | high | arXiv 2602.14162 |
| 8 | **LLM query planner + cross-query 재랭크** 로 어휘 불일치 완화(한·영 혼용에 유효) | med | arXiv 2603.07612 |

## 2. 기각된 통념 (refuted — 채택 금지)

- "**512 토큰이 최적 + sliding window**" — 기각. chunk 크기는 **우리 corpus 로 실측 튜닝**.
- 고정 하이브리드 가중치(**alpha=0.3 / 0.6:0.4**) — 기각. 가중치도 실측 튜닝.
- **4단계 트리 임베딩**(doc→section→para→sentence 집계) — 기각.
- "prompt 순서가 +80% 단일 최대 레버" — 기각(과장). 순서는 보조 레버.
- "**HTML 이 markdown 보다 표에 낫다**" — 기각. markdown/구조-JSON 으로 충분.

> 시사점: chunk 크기·fusion 가중치는 **정설을 베끼지 말고** held-out CD-SEM 질의셋으로 튜닝.

## 3. 권장 아키텍처 (우리 스택 매핑)

```
[harvest 번들] → B1 chunkers (이미 구현: procedure/table_row/error_code/figure/region_text)
      │
      ▼  ① Contextual header 부착 (section_path+heading = 무료 근사, 필요시 LLM enrich)
   chunk + embedding_text + bm25_text
      │
      ▼  ② OpenSearch 단일 인덱스 (primary store)
   ┌─ BM25(text)            ← 정확 코드/파라미터 (레버 #2)
   ├─ dense kNN (bge-m3)    ← 의미 검색
   └─ neural sparse(bge-m3 sparse, rank_features) ← 학습된 lexical
      │  RRF(k=60) 융합 (레버 #2)
      ▼  ③ bge-reranker-v2-m3 cross-encoder 재랭크 (레버 #3, 최대 효과)
   top-K
      │  ④ 컨텍스트 조립: 관련 높은 순을 꼬리에 (레버 #5), dedup, 표=구조보존 markdown,
      │     provenance(page/section/bbox) 인용 + figure 는 페이지 이미지 첨부
      ▼  ⑤ Kimi-K2.6 reader (텍스트 + 필요한 figure 페이지 이미지 직접 읽기 = DVI, 레버 #7)
   답 + 인용
```

보조:
- **⓪ Query planner(레버 #8)**: 한·영 질의를 여러 표현으로 확장 → cross-query 재랭크.
- **Redis**: (a) **semantic cache**(질의→답/검색결과 캐시, 반복질의 저지연), (b) embedding 캐시.
  primary 검색은 OpenSearch, Redis 는 **캐시·서빙** 역할(둘 다 필요할 때만 벡터 인덱스).
- **위키 레이어(레버 보완)**: 이미 가진 **TOC 계층** 위에 **RAPTOR식 재귀 요약**(section→chapter
  →doc 요약 트리)을 얹어 별도 chunk 로 색인 + 네비게이션 위키로 서빙. 단순 lookup 은 chunk-RAG,
  개요/절차/다중참조는 위키/요약 노드. (주의: 단일-hop 사실질의는 표준 RAG 가 GraphRAG 보다
  우위라는 신호 — 위키는 *보완*이지 대체 아님.)

## 4. Phase 1 스펙에 미치는 변화

| 항목 | 기존 스펙 | 연구 반영 후 |
|---|---|---|
| 벡터 스토어 | FAISS (text_index/pixel_index) | **OpenSearch 단일 인덱스**(BM25+dense+sparse+RRF). FAISS 불요 |
| 픽셀 arm | Qwen3-VL-Embedding 배포 게이트 | **삭제 → DVI**(vision embedding 불필요; Kimi 가 답변 때 페이지 읽기) |
| 재랭크 | "충돌시 large model" | **항상 bge-reranker-v2-m3 마지막 단계**(최대 레버) |
| chunk enrichment | embedding_text(heading+section) | 유지 = **Contextual Retrieval 무료 근사**; eval 부족시 LLM 헤더로 강화 |
| 표 | table_row/table_summary | **검증됨**(행 단위·구조보존이 정답) — 변경 없음 |
| late chunking | (스펙 §3 후보) | **금지**(bge-m3 붕괴) |
| 캐시/서빙 | — | **Redis semantic cache** 추가 |
| 위키 | (스펙에 약함) | **RAPTOR식 TOC 재귀 요약** 명시 |

B1 산출물(`rag_chunks.jsonl`)은 그대로 OpenSearch 색인 입력이 된다 → **B1 폐기 없음**,
다음 단계는 FAISS 대신 **OpenSearch 색인기(`opensearch_index.py`) + RRF + reranker hook**.

## 5. 미해결 / 실측 필요 (open questions)

연구에서 **검증 claim 으로 닫히지 않은** 부분 — 우리 corpus 로 직접 측정:
1. **위키 생성 정량효과**(RAPTOR/GraphRAG/DeepWiki): 절차·다중참조 질의에서 chunk-RAG 대비
   기여도. 단일-hop 우위는 표준 RAG 라는 신호만 확인됨.
2. **평가 프로토콜**(RAGAS faithfulness/groundedness, recall@k, parser-loss recovery):
   기존 `benchmark/` 하네스에 4 질의유형별 golden + 위 지표 추가.
3. **chunk 크기 / fusion 가중치 / overlap**: 정설(512·alpha 고정) 기각됨 → held-out
   CD-SEM 질의셋으로 sweep.
4. **OpenSearch ↔ Redis 분업**의 정확한 경계(어디까지 OpenSearch, 어디부터 Redis 캐시).

## 6. 우선순위 로드맵 (연구 반영)

| 우선 | 작업 | 레버 | 게이트 |
|---|---|---|---|
| 1 | **OpenSearch 색인기**: rag_chunks.jsonl → BM25+dense(bge-m3)+sparse, RRF | #2 | bge-m3 API |
| 2 | **bge-reranker-v2-m3 재랭크 hook**(최종 단계) | #3 | reranker API |
| 3 | **컨텍스트 조립기**: 꼬리정렬+dedup+표 구조보존+인용 | #4,#5 | — |
| 4 | **DVI figure 경로**: 후보 페이지 render PNG + 질문 → Kimi | #7 | Kimi |
| 5 | **Query planner**(한·영 확장) + Redis semantic cache | #8 | — |
| 6 | **평가 하네스 확장** + chunk/weight 튜닝 sweep | — | golden 질의셋 |
| 7 | **위키 레이어**: TOC 위 RAPTOR 재귀 요약 | 보완 | 1–3 |

## 7. 캐비엇 (연구 신뢰도 경계)

- Anthropic 35/49/67% 와 EACL'26 0.695/0.816/73% 는 각각 vendor self-benchmark·단일
  금융도메인 — **"이 방향으로 큰 이득"** 으로 읽되 절대수치는 CD-SEM 에 그대로 이식 가정 금지.
- 재랭크 수치는 클라우드 cross-encoder(Cohere) 측정 — 기법은 이식되나 +12/+17pp 크기는 다를 수.
- DVI 는 2026-02 단일 preprint(독립 재현 없음) — 방법·동기는 견고, 수치는 미검증.
- late-chunking 붕괴는 NFCorpus 단일 벤치 — 안전한 해석은 "bge-m3 에서 위험, Contextual 우선".

## 8. 출처 (verified)

- Anthropic, *Contextual Retrieval*: <https://www.anthropic.com/news/contextual-retrieval>
- arXiv 2504.19754 (contextual vs late chunking 검증)
- arXiv 2604.01733 (EACL'26, text+table 하이브리드·재랭크·표73%)
- arXiv 2507.12425 (행 단위 표 색인)
- arXiv 2409.04701 (Jina late chunking 정의)
- arXiv 2407.01219 (RAG best practices, repacking)
- arXiv 2602.14162 ("Index Light, Reason Deep" — DVI)
- arXiv 2603.07612 (KohakuRAG — query planner)
- arXiv 2401.18059 (RAPTOR), OpenSearch neural sparse docs, Redis RAG 블로그(스택 매핑)
