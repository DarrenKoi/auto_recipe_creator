# Align-point 정확도 향상 연구 — ML/DL/VLM 적용 가능성

작성일: 2026-06-16 · 작성자: research note (Claude) · 상태: 사고 정리(thought report), 실험 미수행

> **요청**: align fail 시 align point 를 찾는 정확도를 더 올릴 수 있는가. 지금은 rcp/msr
> 이미지를 쓰는 conventional CV + rule-base 다. ML/DL/VLM 으로 올릴 여지가 있는가, 깊게
> 조사해서 생각을 보고서로.
>
> 이 문서는 코드(`poc/workflow_3/align`)와 offline bench(`poc/workflow_2`)의 실측을
> 근거로 한 **전략 보고서**다. 오피스 데이터 반출 불가 제약상 여기서 실험은 못 했고,
> "무엇을 어디서 검증할지"까지를 설계한다.

---

## 0. 결론 먼저 (TL;DR)

1. **여지는 있다. 단 "matcher 를 ML 로 바꾸자"가 아니다.** bench 가 이미 증명한 핵심은
   *rerank 으로는 안 풀린다*(MI −0.013, contour −0.167)는 것이고, 병목은 **proposer 천장
   (recall@8 ≈ 0.70)** 과 **평평한 점수면(top-k 안에서 정답이 decoy 와 동점, wrong_local_peak
   67%)** 이다. ML 이 들어갈 자리는 *rescore* 가 아니라 **(a) 더 좋은 후보를 만드는 proposer**
   와 **(b) 후보들 사이를 가르는 판별기(discriminator)**, 그리고 **(c) 탐색 공간을 줄이는 ROI
   prior** 다.

2. **가장 ROI 큰 한 수: cond.txt crosshair = 공짜 좌표 라벨.** S 이미지의 crosshair 는
   "그때 장비가 실제로 정렬한 좌표"라 신뢰 가능한 supervised 신호다(반면 tool 의 S/E 라벨은
   metadata 일 뿐 CV 입력 금지 — [[feedback_doubt_s_labels]]). 이 라벨로 **patch-level metric
   learning(Siamese/contrastive) 판별기**를 학습하면, 지금 못 가르는 "정답 patch vs 옆의 decoy
   patch"를 직접 공략한다. 이게 67% wrong_local_peak 의 정면 타격이고, 출력이 *점수*라 doc §8
   경계(좌표는 CV)를 안 깬다.

3. **두 번째 한 수: detector-free dense matcher(LoFTR/RoMa 계열)를 proposer 보강으로.**
   ORB(handcrafted, ensemble 에서 기여 0)와 Chamfer 가 못 잡는 appearance-change 대응이
   강점이다. 단 SEM align key 의 **주기성/반복 패턴**이 correspondence 를 다중해(multi-modal)로
   만들어 오히려 독이 될 수 있다 — periodicity gate 와 함께 가야 한다.

4. **VLM 은 좌표를 절대 안 준다. 영역(ROI)만.** bench 가 이미 채택한 방향(procedure §4.3,
   `vlm_align_key_box.py`). VLM grounding 으로 FOV 안 후보 영역을 1~2개로 좁히면 proposer 가
   그 안에서만 peak 를 찾아 평평한 점수면 문제가 완화된다. 비용은 latency(오피스 GPU Flask
   proxy 경유) 와 환각 — 반드시 CV 점수가 거부권을 갖는 구조.

5. **하지 말 것**: (a) VLM/DL 이 최종 좌표를 단독 결정, (b) rerank-only 재시도(이미 기각),
   (c) pixel-identity 가정 모델(NCC/SSIM 1차 지표 — [[project_align_key_matching_constraint]]),
   (d) 오피스 데이터를 Mac 으로 들여와 학습([[feedback_no_office_data_to_mac]]).

6. **우선순위**: ①Phase-A metric-learning 판별기(오피스 내 학습, 라벨 공짜) → ②VLM ROI
   prior(인프라 재사용, 빠름) → ③detector-free matcher proposer(GPU 필요, periodicity 동반)
   → ④heatmap regression(데이터·일반화 리스크 큼, 후순위).

---

## 1. 지금 align point 를 어떻게 찾는가 (정밀 지도)

`poc/workflow_3/align` 의 좌표 결정은 전부 **classical CV + rule** 이다. ML/DL 은 한 곳도 없다.
VLM 은 좌표가 아니라 *영역/버튼*에만 쓰인다.

### 1.1 매칭 엔진 (`matching/engine.py`, `matching/ensemble.py`)

- **전처리**: CLAHE(clip 2.0) → Gaussian(σ=1.0) → Canny(60/160) → distance transform(L2).
  즉 매칭의 1차 표현은 **edge 구조**다. (픽셀 밝기 매칭을 일부러 피한다 — fail 은 "다르게
  보여서" 나기 때문, procedure §8.)
- **Proposer (후보 생성)**: 3-channel ensemble — C1 Canny, C2 Scharr(density-matched),
  C3 orientation-binned(8 bins) → **RRF 융합**(k0=10). 멀티스케일 Chamfer NMS 로 채널별 top-k
  를 뽑아 RRF 로 합친다.
- **Reranker (후보 재정렬)**: top-k 에 대해 `sel = 0.5·chamfer + 0.5·max(0, NCC)`.
  ORB 는 ensemble 경로에서 **기여 0**(`orb_inlier_ratio=0.0`)으로 사실상 퇴출.
- **결정 임계**: Youden-calibrated `match=0.6053`, `adjust=0.4727`. decision ∈ {match, adjust, low}.
- **판별성 게이트(distinctiveness)**: best−second gap ≥ 0.04 이고 second/best ≤ 0.94 여야
  `distinctive=True`. 이게 거짓이면 "평평한 점수면"으로 보고 reject/escalate.
- **스케일**: nm/pixel 메타데이터가 있으면 단일 스케일, 없으면 `DEFAULT_SCALES`(0.7~1.4)
  멀티스케일. zoom-out 상황은 `BROAD_SCALES`(0.15~0.5) 미니어처 탐색.

### 1.2 템플릿 생성 (`templates.py`, `cond_template.py`)

- rcp 등록 이미지 → `AlignKeyTemplate`. cond.txt 의 박스 좌표로 **box-crop** 하고,
  align point 는 이미지 중심이라는 가정 위에 **`align_offset_xy`(박스중심↔이미지중심)** 를
  decouple 해 저장([[project_rcp_white_box_unique_area]]).
- crosshair/box stroke 는 inpaint 로 제거(distractor — [[project_msr_crosshair_cv_distractor]]).

### 1.3 Consensus 재등록 (`consensus_*.py`)

- 최근 성공(S) 이미지 여러 장을 모아(`gather`) crosshair 중심으로 crop →
  **phase-correlation co-registration**(2 iters) → **pixel-wise median** 으로 합성 →
  edge/Laplacian 비율 quality gate(0.70/0.50) 통과 시 rcp 대신 라우팅.
- 효과(실측): recall +0.282, rank1 +0.269. **rcp staleness 는 실재하지만 2차 병목**임을 증명.
- min_s=3(LOO 안정성 floor), recipe 키 충돌 회피 위해 eqp/class/recipe 로 고유화
  ([[project_consensus_sparse_golden_and_recipe_id_collision]]).

### 1.4 보정 흐름 (`correction.py`)

`correct_align_fail_auto` → paused 프레임 ensemble 매칭 → **key visibility gate**
("act" | "fallback_search" | "engineer_review") → reposition(best_xy + offset, 더블클릭
recenter) → VLM 으로 OK 버튼 *영역* 찾아 screen 클릭. 게이트 low 면 live two-phase search 로
폴백. 출력 `CorrectionOutcome.status` 가 cube notify 결정을 구동.

`★ Insight ─────────────────────────────────────`
파이프라인 전체가 "edge 구조 + 순위 융합 + median consensus" 라는 **handcrafted 통계**다.
학습된 파라미터가 0개라는 건, 데이터가 늘어도 시스템이 *자동으로 좋아지지 않는다*는 뜻이다.
ML 의 진짜 가치는 정확도 한 방이 아니라 **"라벨이 쌓일수록 좋아지는 축"을 하나 만드는 것**.
지금은 그 축이 없다.
`─────────────────────────────────────────────────`

---

## 2. 측정된 정확도와 한계 (bench 실증)

`poc/workflow_2` golden eval 의 숫자(파일 근거는 bench docs):

| 지표 | 값 | 의미 |
|---|---|---|
| proposer recall@8 (3-channel RRF, 현재) | **≈0.698** (0.557→0.698) | top-8 안에 정답이 들어온 비율. 천장. |
| rank1 (consensus 후) | ≈0.538 | 1등으로 맞춘 비율 |
| wrong_local_peak | **67%** (153/229) | 정답 위치를 알려줘도 옆 peak 를 고름 |
| free_best bACC (S/E 분리) | 0.549~0.579 | 성공/실패 점수로 구분 거의 불가(때로 E>S 역전) |
| MI rerank lift | **−0.013** | rerank 무효 |
| contour(Hu) rerank lift | **−0.167** | rerank 유해 |
| consensus 재등록 lift | recall +0.282 | rcp staleness 실재(2차) |

해석:
- **천장이 proposer 에 있다.** top-8 에 정답이 30% 확률로 아예 없으면 어떤 rerank 도 못 살린다.
- **top-8 안에서도 평평하다.** 들어와도 decoy 와 점수가 같아 67% 가 엉뚱한 peak 선택.
- **점수가 S/E 를 못 가른다.** Chamfer 점수면 자체가 변별력 부족
  ([[project_matcher_flat_chamfer_distinctiveness]]). MI 만 0.627 로 약하게 신호.
- miss 의 상당수는 **구조적**(far/veryfar) — ensemble-on-consensus 가 기각된 이유도
  "잔여 miss 절반이 구조적이라 천장 +0.02~0.06뿐"([[project_ensemble_on_consensus_rejected]]).

---

## 3. 왜 막혔나 — 근본 진단 (세 개의 벽)

1. **벽 A — Proposer recall 천장(≈0.70).** Chamfer/edge 후보 생성이 30% 케이스에서 정답을
   top-k 에 못 넣는다. 원인: appearance change(공정 변화로 구조가 변형) + 저텍스처/반복 패턴.
   → *후보 생성 알고리즘 자체*를 바꿔야 함. rerank·임계 튜닝으론 못 넘음.

2. **벽 B — 평평한 점수면(within-frame ambiguity).** SEM align key 주변은 비슷한 edge 가
   반복돼, 정답 patch 와 decoy patch 가 거의 동점. handcrafted 점수(Chamfer/NCC/MI/Hu)는
   이 미세 차이를 못 가른다. → *판별 표현(discriminative representation)* 이 필요. 이게 ML 의
   고전적 강점(metric learning).

3. **벽 C — 라벨 신뢰도 비대칭.** tool 의 S/E 라벨은 못 믿지만([[feedback_doubt_s_labels]]),
   **cond.txt 의 crosshair 좌표는 믿을 수 있다**(실제 정렬 위치). 다만 E 이미지엔 crosshair 가
   없다([[project_e_images_no_crosshair]]). → 학습은 **S 이미지의 crosshair 를 positive 좌표
   라벨로** 써야 하고, negative 는 같은 FOV 의 decoy peak 에서 mining.

`★ Insight ─────────────────────────────────────`
벽 A 와 벽 B 는 다른 약이다. A 는 "후보 목록에 정답을 넣는" recall 문제 → dense/detector-free
matcher 나 ROI prior. B 는 "목록 안에서 정답을 1등 시키는" precision 문제 → 학습된 판별기.
지금까지 시도(rerank)는 B 를 *handcrafted 점수로* 풀려다 실패했다. 핵심 통찰: **B 는 표현
학습 없이는 안 풀린다.** 그래서 metric learning 이 1순위다.
`─────────────────────────────────────────────────`

---

## 4. 불변 제약: VLM=영역, CV=좌표

`workflow_2_procedure.md:19`, CLAUDE.md:194 (confirmed 2026-05-25):

> OpenCV produces quantitative scores and final coordinates; VLM only identifies regions,
> explains ambiguous FOVs, and assesses feasibility. **Never let a VLM answer override a low
> CV score or decide a repeatable stage transition.**

이건 협상 대상이 아니다. 따라서 모든 ML/DL/VLM 제안은 다음 둘 중 하나여야 한다:
- **(점수형)** 학습 모델이 *후보의 점수*를 낸다 → 최종 좌표는 여전히 CV 가 top 후보의
  좌표로 결정. (metric-learning 판별기, learned matcher 의 confidence)
- **(영역형)** 학습 모델이 *탐색 영역(ROI/박스)* 을 낸다 → CV 가 그 안에서 peak 를 찾음.
  (VLM grounding)

학습 모델이 직접 (x,y) 를 뱉어 그대로 클릭하는 건 — heatmap regression 포함 — 이 경계를
위반할 소지가 있다(§5.3 에서 어떻게 경계 안으로 넣을지 다룸).

---

## 5. ML/DL/VLM 후보 평가

각 후보를 **(적합도 / 어느 벽을 치나 / 기대 lift / 데이터·런타임 feasibility / 경계 준수)** 로 평가.

### 5.1 Metric-learning patch scorer (Siamese / contrastive) — **1순위**

- **무엇**: 작은 CNN(또는 lightweight backbone) 임베딩 + contrastive/triplet loss. 입력은
  template crop 과 frame patch, 출력은 "같은 align key 인가" 유사도 점수. 후보 top-k 의
  patch 들에 이 점수를 매겨 CV 가 1등 선택.
- **치는 벽**: **B(평평한 점수면).** 정답 patch 와 decoy patch 의 미세 구조 차이를 *학습된*
  거리로 가른다. rerank 이 handcrafted 라 실패했던 바로 그 자리를 표현 학습으로 대체.
- **라벨**: cond.txt crosshair(positive 좌표) + 같은 FOV 의 proposer decoy peak(hard negative).
  공짜이고 신뢰 가능. cross-recipe 로 모으면 일반적 "align-key-ness" 학습 가능.
- **경계**: 출력이 *점수* → doc §8 준수. CV 가 거부권 유지(낮으면 fallback).
- **런타임**: 임베딩 추론은 CPU 로도 top-k(≤24) 에 대해 ms 급 가능. 오피스 GPU 불요.
- **데이터 반출**: 학습을 **오피스 안에서** 수행해야 함([[feedback_no_office_data_to_mac]]).
  → Mac 에서 학습 스크립트/아키텍처를 blind 작성, 오피스 pull 후 실행, 텍스트 digest 로 피드백.
  workflow_2 의 golden eval 하니스가 그대로 학습 라벨 생성기로 재사용 가능.
- **기대 lift**: 정성적으로 가장 큼(67% wrong_local_peak 의 직접 타격). 단 정량은 오피스
  실험 전까지 미지수 — A/B 는 `golden_localization_eval_cond.py` 위에서.
- **리스크**: 작은 라벨셋 overfitting, recipe 편향. → augmentation(밝기/대비/약한 affine,
  단 pixel-identity 가정은 금지) + cross-recipe split eval.

### 5.2 Detector-free dense matcher (LoFTR / RoMa / SuperPoint+SuperGlue) — **3순위**

- **무엇**: 학습된 dense correspondence. template↔frame 간 대응점을 직접 추정, homography/
  중심으로 align point 후보 산출. ORB(sparse, 기여 0) 대체/보강.
- **치는 벽**: **A(proposer recall).** 저텍스처·appearance-change 에 ORB/Chamfer 보다 강함이
  논문/벤치에서 일관. top-k 에 정답을 넣는 능력 향상 기대.
- **경계**: matcher confidence 를 *점수*로, 중심을 *후보 좌표*로 → CV 검증 후 채택하면 준수.
- **런타임**: **GPU 필요**(LoFTR/RoMa). 다행히 오피스에 VLM Flask proxy + GPU 서버가 이미 있음
  → 같은 패턴으로 matcher 를 별도 service 로 서빙 가능(`flask_api/vlm_serve` 구조 재사용).
  단 paused 매칭은 한 프레임이라 latency 허용 범위(수백 ms) 일 듯.
- **결정적 리스크**: **SEM align key 의 주기성/반복 패턴.** correspondence matcher 는 반복
  구조에서 multi-modal(여러 똑같은 대응) → 잘못된 homography. 반드시 §5 이하 **periodicity
  gate**(`ensemble_lab.template_periodicity`, AUC≈0.61) 와 동반해, 주기성 높은 key 는 dense
  matcher 신뢰도를 깎아야 함.
- **기대 lift**: A 벽에 대해 중간~큼이지만 주기성 리스크로 분산 큼. **먼저 office 프레임에서
  "주기성 낮은 key 비율"을 측정**해 적용 대상을 정해야 비용 정당화.

### 5.3 Heatmap / keypoint regression (U-Net 류 detection) — **4순위**

- **무엇**: FOV 전체를 입력받아 align point 확률 heatmap 회귀. argmax 가 좌표.
- **치는 벽**: A+B 동시(이론상). 가장 직접적인 "정답 좌표 찾기".
- **경계 문제**: 모델이 좌표를 직접 뱉음 → doc §8 위반 소지. **해법**: heatmap 을 *후보
  제안(proposer)* 으로만 쓰고, 최종은 그 위치에서 CV(Chamfer/metric-scorer)가 재확인하면
  경계 안. 즉 "DL proposer + CV verifier" 로 감싸야 함.
- **데이터**: 회귀는 metric-learning 보다 라벨/다양성 요구가 큼. recipe 간 일반화가 핵심
  난제(per-recipe 모델은 운영 부담). 초기 설계에서 **이미 한 번 보류**된 방향
  ([[project_align_key_matching_constraint]] 의 정신과 충돌하기 쉬움).
- **판단**: 데이터가 충분히 쌓이고 ①②가 천장에 닿은 *후에* 검토. 지금은 후순위.

### 5.4 VLM ROI grounding — **2순위 (이미 채택된 방향)**

- **무엇**: ui-venus/mai-ui 로 "이 FOV 에서 align key 가 있을 법한 영역 박스" 를 grounding.
  CV proposer 는 그 박스 안에서만 peak 탐색 → 탐색 공간 축소 → 평평한 점수면 완화.
- **치는 벽**: A(공간 축소로 false peak 제거) + B 일부.
- **경계**: VLM 은 *영역*만(좌표 아님). `vlm_align_key_box.py` 코드 이미 존재(office-only 호출).
  procedure §4.3/§6 가 명시적으로 채택한 방향.
- **런타임/인프라**: Flask VLM proxy 재사용 → 추가 인프라 0. latency 는 paused 1프레임이라 수용.
- **리스크**: 환각([[project_paddleocr_vl_screenshot_hallucination]] 의 교훈), grounding 거부
  ([-1,-1]) 정상 처리. **CV 점수가 항상 거부권** — VLM 박스가 비어도 full-FOV 폴백.
- **기대 lift**: 중간. 특히 far/veryfar 구조적 miss(proposer 가 엉뚱한 데서 peak 잡는 case)에
  유효. 비용 대비 빠르게 검증 가능 → 우선순위 높음.
- **First-letter/region anchoring** 등 기존 prompt 원칙 재사용.

### 5.5 Template bank / few-shot — 보조

- **무엇**: 한 장 rcp/단일 consensus 대신, S 이미지들로 **여러 템플릿(appearance 변이별)** 을
  bank 로 유지, 매칭 시 best-of-bank. far/veryfar 구조적 miss 의 일부는 "현재 모습이 등록
  모습과 너무 다름"이라 변이 bank 가 recall 을 올릴 수 있음.
- **위치**: [[project_ensemble_on_consensus_rejected]] 가 "어려운 절반은 template bank/ROI/VLM
  축"이라 명시. consensus median 의 자연 확장.
- **ML 성격**: clustering(appearance 군집) 정도의 가벼운 학습. 저비용·저리스크 → §5.1 과 병행 가능.

### 5.6 그 외 (간략)

- **Self-supervised pretrain(SimCLR/DINO) on unlabeled SEM frames**: §5.1 backbone 의
  초기화로 유용(라벨 적을 때). 오피스 unlabeled 프레임 풍부 → pretrain → metric head fine-tune.
- **Diffusion/generative align**: 과함. 기각.
- **Classical 개선(여전히 유효)**: log-polar/Fourier-Mellin 으로 회전·스케일 불변 후보,
  multi-scale phase correlation. ML 전에 싸게 시도 가능하나 벽 B 는 못 깸.

---

## 6. 도메인 제약이 후보를 거른다

이 프로젝트 특유의 제약이 일반적 ML 추천을 강하게 필터링한다:

1. **오피스 데이터 반출 불가** ([[feedback_no_office_data_to_mac]]). → Mac 학습 불가.
   모든 DL 학습은 **오피스 PC 안**에서. Mac 은 blind 코딩 + digest 피드백.
   *이것이 "큰 모델 fine-tune" 보다 "오피스에서 돌릴 만한 작은 모델 + 공짜 라벨" 을 선호하게
   만드는 결정적 이유.*
2. **런타임 Windows, real-time loop**. paused 매칭은 1프레임(latency 여유) 이지만 live search
   는 프레임마다 → live 경로엔 무거운 DL 금지(lightweight scorer 만).
3. **Repeatability/감사성**. 같은 입력 → 같은 좌표. 확률적 추론·비결정 모델은 좌표 단계에서
   금지(점수 단계는 허용).
4. **라벨 신뢰도**: crosshair O(신뢰), tool S/E X(metadata). 학습 타깃 설계의 제1원칙.
5. **GPU 가용성**: VLM Flask proxy + GPU 서버 존재 → dense matcher/VLM 은 *서비스로* 붙일 수
   있음(인프라 재사용). 단 매 알람마다 GPU 호출 비용/가용성 확인 필요.
6. **데이터 양**: 298 recipe, S 희박([[project_consensus_sparse_golden_and_recipe_id_collision]]).
   → recipe-specific 모델 비현실, **cross-recipe 일반 모델 + recipe prior(consensus)** 조합.

---

## 7. 권장 로드맵

원칙: **싸고 경계 안전한 것부터, 데이터가 쌓이면 좋아지는 축을 심는다.** 검증은 전부
workflow_2 golden eval(`golden_localization_eval_cond.py` / `golden_consensus_eval_cond.py`)
위 A/B, 통과한 것만 workflow_3 포팅([[feedback_ensemble_dev_in_workflow2_then_port]]).

**Phase A (지금~단기) — VLM ROI prior.** 인프라 0, 빠른 검증.
- `vlm_align_key_box.py` grounding 박스를 proposer 탐색 영역으로 주입(옵션 플래그, 기본 off).
- A/B: ROI-제한 proposer vs full-FOV. 측정: recall@8, rank1, far/veryfar miss 감소율.
- 게이트: VLM 거부/빈 박스 → full-FOV 폴백. CV 점수 거부권.

**Phase B (단기~중기, 최대 ROI) — Metric-learning patch scorer.**
- 오피스에서 cond.txt crosshair positive + decoy hard-negative 로 라벨셋 생성(기존 하니스 재사용).
- 작은 임베딩 net(self-supervised pretrain 옵션) + contrastive loss. CPU 추론.
- top-k 후보 rescore 를 `sel = w·chamfer + ...` 의 학습된 항으로 교체/추가.
- A/B: rank1, wrong_local_peak 비율. 목표: 67% → 유의 감소.
- 경계: 점수만, CV top-1 선택은 유지.

**Phase C (중기) — Template bank + (조건부) detector-free matcher.**
- S 이미지 appearance clustering → 변이 bank(저비용). recall@8 천장 돌파 시도.
- 오피스 프레임 주기성 분포 측정 → 주기성 낮은 key 부분집합에만 LoFTR/RoMa 서비스 적용.
  periodicity gate 동반 필수.

**Phase D (장기, 조건부) — Heatmap regression proposer.**
- A/B/C 가 천장에 닿고 라벨이 충분할 때만. "DL proposer + CV verifier" 로 경계 안에 감쌈.

각 Phase 는 **kill switch 플래그**로 출시(기존 `ALIGN_FAIL_*` 컨벤션). 실패 시 즉시 기존 경로.

---

## 8. 하지 말 것 (anti-recommendations)

- **VLM/DL 단독 좌표 결정** — doc §8 위반. 항상 CV 가 최종·거부권.
- **rerank-only 재시도** — MI(−0.013)/contour(−0.167) 로 이미 기각. 표현 학습 없는 rescore 금지.
- **pixel-identity 모델(NCC/SSIM 1차 지표)** — 공정 변화로 무효([[project_align_key_matching_constraint]]).
- **오피스 데이터 Mac 반입 학습** — 금지. 학습은 오피스 안에서.
- **tool S/E 라벨을 학습 타깃으로** — metadata 일 뿐([[feedback_doubt_s_labels]]). crosshair 만.
- **recipe-specific 대형 모델** — S 희박·298 recipe 로 비현실. cross-recipe 일반화 + consensus prior.
- **live 경로에 무거운 DL** — real-time 위반. lightweight scorer 만.

---

## 부록 — 검증 자산 매핑

| 검증할 것 | 어디서 | 지표 |
|---|---|---|
| ROI prior 효과 | `golden_localization_eval_cond.py` + VLM box | recall@8, far/veryfar miss↓ |
| metric scorer | 위 + crosshair 라벨 생성기 | rank1↑, wrong_local_peak↓ |
| template bank | `golden_consensus_eval_cond.py` 확장 | recall@8 천장 돌파 |
| dense matcher | proposer A/B(`proposer_recall_ab`) | recall@8, 주기성 교차분석 |

> 관련 메모리: [[project_matcher_flat_chamfer_distinctiveness]],
> [[project_ensemble_proposer_and_consensus_race]], [[project_ensemble_on_consensus_rejected]],
> [[project_consensus_clean_vs_raw_ab]], [[feedback_doubt_s_labels]],
> [[feedback_no_office_data_to_mac]], [[project_align_key_matching_constraint]].
