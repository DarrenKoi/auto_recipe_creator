# 03. workflow_2 — 오프라인 CV 평가 벤치 (Evaluation Bench)

> 목적: align key matching의 정확도를 높이는 CV 변경(matching / ensemble / threshold / consensus)을
> **golden set으로 객관 검증·A/B·튜닝**한 뒤, 검증된 변경만 production(workflow_3)으로 포팅한다.
> workflow_2는 동결이 아니라 **활성 연구·검증 harness**다.

근거: `docs/workflow_2/`, `poc/workflow_2/docs/`(ADR·runbook), `docs/journals/260618/`,
`poc/workflow_3/README.md`.

## 1. 역할과 원칙

- **production 엔진 무수정 원칙**: workflow_2는 `poc.workflow_3.align`에서 엔진을 import만 한다
  (반대 방향 금지). 실험은 `ensemble_lab.py`에서 bit-parity로 fork.
- **"가정 말고 측정"**: 정책 변경 전 항상 golden set으로 수치를 먼저 확인.
- **벤치→프로덕션 파이프라인**: 여기서 검증 → workflow_3/align으로 검증된 변경만 포팅 → 회귀 위험 통제.

## 2. 3대 Golden Driver

| 드라이버 | 측정 대상 |
|----------|-----------|
| `golden_localization_eval_cond.py` | rcp(등록 align key) template 단독의 localization 정확도 |
| `golden_consensus_eval_cond.py` | **consensus template**(최근 성공 median) vs rcp의 A/B |
| `golden_combined_eval_cond.py` | **production 라우팅**(consensus 우선·rcp 폴백) end-to-end + 3축 리포트 |

`golden_combined_eval_cond.py`의 3축:
- **(A) consensus scaling** — 성공 이미지 수(`cons_pool_n`)별 층화: "S가 많을수록 좋아지는가".
- **(B) rcp-only arm** — consensus 부적격 recipe(=`edge_ncc` 등 lab 레버 testbed).
- **(C) routed overall** — frame-weighted 종합.
- 추가로 **OM/SEM modality 층화**(rank1/topk를 OM vs SEM로 분리)와 한 줄 `[DIGEST]` 출력.

평가 지표: `in_topk`(proposer recall — 정답이 top-N 후보 안에 있는가), `rank1`(top 후보가 정답인가).

## 3. 핵심 CV 기법

### (1) Ensemble proposer + RRF + NCC rerank
- 3개 proposer 채널(C1 Canny+Chamfer, C2 Scharr+Chamfer, C3 orientation histogram+Chamfer)의
  후보 리스트를 **RRF(Reciprocal Rank Fusion)** 로 융합 → score scale 차이에 강건.
- 융합 상위 후보를 **NCC(정규화 상관)** 로 rerank해 최종 선택.
- Youden 보정 임계: `match=0.6053`, `adjust=0.4727` (paused/static 프레임용).

### (2) Consensus re-registration (핵심 발견)
- 등록 align key(rcp)는 시간이 지나면 외관이 달라져(stale) 매칭이 약해진다.
- 같은 recipe의 **최근 성공(S) crop들을 crosshair 기준으로 co-register 후 median blend**해
  "현재 외관을 따라가는" consensus template를 만든다.
- blur 가드(edge_ratio·lap variance)로 흐릿한 consensus는 rcp로 폴백.

### (3) Youden threshold 보정
- 모든 S/E 프레임 쌍의 proposer 점수를 모아 **Youden J(=TPR+TNR−1) 최대화**로 match/adjust 임계 결정.

### (4) ensemble_lab 실험 채널
- `edge_ncc` C4 proposer(edge map에 직접 NCC) 등을 production 무수정으로 A/B —
  회귀 없이 정확도가 오르면 포팅 후보.

## 4. 평가 방법론

- **Golden set**: 등록 OM/SEM key vs 측정 S/E 프레임을 ground truth로 사용.
- **cond.txt 기반 ground truth**: CV 검출이 아니라 조건 메타데이터(crosshair_xy, box_ltrb)로 정답 좌표 산출.
- **Leave-One-Out(LOO)**: 각 S에 대해 나머지 S로 consensus를 만들어 held-out S에 매칭(누설 방지).
- **history-first + LOO 폴백**: 별도 history 풀이 `min_s` 이상이면 disjoint 풀로 consensus(LOO 불필요),
  없으면 byte-identical LOO 경로.
- **consensus history 풀**: `<class>/<recipe>` 키(**장비 무관** — 같은 recipe면 장비 달라도 공유)로
  최근 S 8~10장 rolling 적재. production consensus 캐시와 동일 포맷.

## 5. 측정 결과 (golden set 벤치)

| 지표 | rcp 단독 | consensus 라우팅 | 비고 |
|------|----------|------------------|------|
| `in_topk` (proposer recall) | **0.434** | **0.876** | +0.442 (약 +102%) |
| `rank1` (top 후보 정답) | **0.318** | **0.764** | +0.446 |

(LOO A/B, min_s=3 기준. production 기본 min_s=4는 의도적 보수 정책.
근거: `poc/workflow_3/README.md`, workflow_2 bench.)

> **주의**: 위 수치는 **golden set 벤치 기준**이다. 오피스 실데이터의 라우팅 종합 정확도와
> OM/SEM 층화 판정은 office `GOLDEN_ROOT`/`HISTORY_ROOT` 데이터에서 `golden_combined_eval_cond.py`
> 실행 후 `[DIGEST]` 한 줄로 확정 예정([05_status_roadmap.md](05_status_roadmap.md)).

## 6. 의의

- consensus re-registration은 "정렬 정확도의 천장(rcp 단독 ~0.43)"을 뚫은 핵심 발견이며,
  workflow_3의 보정 품질을 좌우하는 알고리즘이다.
- bench/config 분리(`golden_eval_config.py`, gitignore) + `[DIGEST]` 한 줄 회신 구조로
  오피스↔개발 간 결과 전달 비용을 최소화.
