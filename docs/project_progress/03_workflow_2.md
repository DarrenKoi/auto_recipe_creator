# 03. workflow_2 — 오프라인 CV 평가 벤치 (Evaluation Bench)

> 목적: align key matching의 정확도를 높이는 CV 변경(matching / ensemble / threshold / consensus)을
> **golden set으로 객관 검증·A/B·튜닝**한 뒤, 검증된 변경만 production(workflow_3)으로 포팅합니다.
> workflow_2는 동결이 아니라 **활성 연구·검증 harness**입니다.

근거: `docs/workflow_2/`, `poc/workflow_2/docs/`(ADR·runbook), `docs/journals/260618/`,
`poc/workflow_3/README.md`.

## 1. 역할과 원칙

- **production 엔진 무수정 원칙**: workflow_2는 `poc.workflow_3.align`에서 엔진을 import만 합니다
  (반대 방향 금지). 실험은 `ensemble_lab.py`에서 bit-parity로 fork합니다.
- **"가정 말고 측정"**: 정책 변경 전에는 항상 golden set으로 수치를 먼저 확인합니다.
- **벤치→프로덕션 파이프라인**: 여기서 검증한 뒤 workflow_3/align으로 검증된 변경만 포팅하여 회귀 위험을 통제합니다.

## 2. Golden Driver (4종)

| 드라이버 | 측정 대상 |
|----------|-----------|
| `golden_localization_eval_cond.py` | rcp(등록 align key) template 단독의 localization 정확도 |
| `golden_consensus_eval_cond.py` | **consensus template**(최근 성공 median) vs rcp의 A/B |
| `golden_combined_eval_cond.py` | **production 라우팅**(consensus 우선·rcp 폴백) end-to-end + 3축 리포트 |
| `golden_reregister_report_cond.py` | **재등록 우선순위 리포트**(2026-06-23 신규) — align key 재등록이 필요한 recipe를 순위화 + 대체 영역 제안 |

`golden_combined_eval_cond.py`의 3축은 다음과 같습니다.
- **(A) consensus scaling** — 성공 이미지 수(`cons_pool_n`)별 층화: "S가 많을수록 좋아지는가".
- **(B) rcp-only arm** — consensus 부적격 recipe(=`edge_ncc` 등 lab 레버 testbed).
- **(C) routed overall** — frame-weighted 종합.
- 추가로 **OM/SEM modality 층화**(rank1/topk를 OM vs SEM로 분리)와 한 줄 `[DIGEST]` 출력을 제공합니다.

평가 지표는 `in_topk`(proposer recall — 정답이 top-N 후보 안에 있는가)와 `rank1`(top 후보가 정답인가)입니다.

## 3. 핵심 CV 기법

### (1) Ensemble proposer + RRF + NCC rerank
- 3개 proposer 채널(C1 Canny+Chamfer, C2 Scharr+Chamfer, C3 orientation histogram+Chamfer)의
  후보 리스트를 **RRF(Reciprocal Rank Fusion)** 로 융합하여 score scale 차이에 강건하게 만듭니다.
- 융합 상위 후보를 **NCC(정규화 상관)** 로 rerank하여 최종 선택합니다.
- Youden 보정 임계: `match=0.6053`, `adjust=0.4727` (paused/static 프레임용).

### (2) Consensus re-registration (핵심 발견)
- 등록 align key(rcp)는 시간이 지나면 외관이 달라져(stale) 매칭이 약해집니다.
- 같은 recipe의 **최근 성공(S) crop들을 crosshair 기준으로 co-register한 뒤 median blend**하여
  "현재 외관을 따라가는" consensus template를 만듭니다.
- blur 가드(edge_ratio·lap variance)로 흐릿한 consensus는 rcp로 폴백합니다.

### (3) Youden threshold 보정
- 모든 S/E 프레임 쌍의 proposer 점수를 모아 **Youden J(=TPR+TNR−1) 최대화**로 match/adjust 임계를 결정합니다.

### (4) ensemble_lab 실험 채널
- `edge_ncc` C4 proposer(edge map에 직접 NCC) 등을 production 무수정으로 A/B합니다 —
  회귀 없이 정확도가 오르면 포팅 후보가 됩니다.

## 4. 평가 방법론

- **Golden set**: 등록 OM/SEM key vs 측정 S/E 프레임을 ground truth로 사용합니다.
- **cond.txt 기반 ground truth**: CV 검출이 아니라 조건 메타데이터(crosshair_xy, box_ltrb)로 정답 좌표를 산출합니다.
- **Leave-One-Out(LOO)**: 각 S에 대해 나머지 S로 consensus를 만들어 held-out S에 매칭합니다(누설 방지).
- **history-first + LOO 폴백**: 별도 history 풀이 `min_s` 이상이면 disjoint 풀로 consensus(LOO 불필요),
  없으면 byte-identical LOO 경로를 사용합니다.
- **consensus history 풀**: `<class>/<recipe>` 키(**장비 무관** — 같은 recipe면 장비가 달라도 공유)로
  최근 S 8~10장을 rolling 적재합니다. production consensus 캐시와 동일 포맷입니다.

## 5. 측정 결과 (golden set 벤치)

| 지표 | rcp 단독 | consensus 라우팅 | 비고 |
|------|----------|------------------|------|
| `in_topk` (proposer recall) | **0.434** | **0.876** | +0.442 (약 +102%) |
| `rank1` (top 후보 정답) | **0.318** | **0.764** | +0.446 |

(LOO A/B, min_s=3 기준. production 기본 min_s=4는 의도적 보수 정책입니다.
근거: `poc/workflow_3/README.md`, workflow_2 bench.)

> **주의**: 위 수치는 **오프라인 golden set 벤치 기준**입니다. 오피스 실데이터의 라우팅 종합 정확도와
> OM/SEM 층화 판정은 office `GOLDEN_ROOT`/`HISTORY_ROOT` 데이터에서 `golden_combined_eval_cond.py`
> 실행 후 `[DIGEST]` 한 줄로 확정할 예정입니다([05_status_roadmap.md](05_status_roadmap.md)).

## 6. 최근 진행 (2026-06-19 ~ 06-23)

### (1) OM/SEM modality-split 평가 + 조건부 레버
- combined eval을 OM/SEM **modality별로 분리**하여 rank1/topk를 따로 봅니다(같은 recipe라도 OM은
  프레임의 10~20%, SEM은 80~100%를 채우는 등 실패 양상이 다릅니다).
- modality별 **Youden separability**(delta-vs-production) + **failure-mode 히스토그램** +
  **split verdict**(어느 modality에 어떤 레버를 댈지)를 digest에 출력합니다.

### (2) Job 2 box-crop — **기각**(ADR 0005)
- 가설: align key가 SEM 프레임을 가득 채우므로 등록 white box로 crop하면 distractor를 격리하여 매칭이
  오를 것이다.
- consensus arm에 box-crop을 붙여(center vs whitebox, 고정 분모) A/B한 결과 **오피스 실데이터에서 열세**
  였습니다(OM −0.042, SEM −0.110). 또한 hit-tolerance 혼동(`tol_short`) 버그를 잡아 공정 비교로 재측정해도
  이득이 없어 → **production 미포팅, ADR 0005로 기각**했습니다.
- 교훈: "그럴듯한" ROI 축소가 SEM에서는 무효였습니다(distractor가 key 내부 periodic 구조라 frame 밖이 아님).
  남은 절반의 어려운 케이스는 ROI/box-crop이 아니라 **재등록·template bank** 축으로 풀어야 합니다.

### (3) 재등록 우선순위 리포트 (Phase 1, S-only)
- 새 driver `golden_reregister_report_cond.py`: align key 자체가 chronic-ambiguous(반복 패턴 위에
  등록되어 근본적으로 매칭이 어려운) recipe를 **순위화**하고, 더 distinctive한 **대체 영역(box)을 제안**합니다.
- 증거 3-tier(distinctiveness/fidelity 축) + risk score 랭킹. 대체 영역은 rcp self-match로 후보를 찾아
  overlay로 시각화합니다(C1 screening + C2 box-suggestion).
- 현재 **오피스 캘리브레이션 중**입니다: 초기 실행에서 STRONG tier가 과발화(SEM 95% / OM 52%)하여
  STRONG 정의를 "GT-absent + fail-fraction floor"로 조이고 `SPLIT_MIN_S`를 4→2로 조정했습니다.
  정확도는 오피스 실데이터로 확정할 예정이며, Phase 2 = E-frame 확인 단계입니다.

## 7. 의의

- consensus re-registration은 "정렬 정확도의 천장(rcp 단독 ~0.43)"을 뚫은 핵심 발견이며,
  workflow_3의 보정 품질을 좌우하는 알고리즘입니다.
- bench/config 분리(`golden_eval_config.py`, gitignore) + `[DIGEST]` 한 줄 회신 구조로
  오피스↔개발 간 결과 전달 비용을 최소화했습니다.
