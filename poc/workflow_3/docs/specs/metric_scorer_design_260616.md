# Metric-learning Patch Scorer — 학습/라벨 생성 설계 spec

작성일: 2026-06-16 · 상태: 설계(design), 미구현 · 상위 연구: `../study/align_point_accuracy_ml_vlm_research_260616.md`

> **한 줄 목표**: top-k 후보 안에서 "진짜 align key patch" 와 "옆의 decoy patch" 를 가르는
> 학습된 점수를 추가해, **wrong_local_peak 67%** 를 줄인다(= 벽 B, 평평한 점수면 직격).
> 출력은 *점수*이고 최종 좌표는 여전히 CV 가 고른다 → doc §8 경계 준수.

---

## 1. 왜 이게 맞는 약인가

bench 실측: top-8 안에 정답이 들어와도 67%(153/229) 가 엉뚱한 peak 를 1등으로 고른다. handcrafted
rescore(MI −0.013, contour −0.167)는 이 미세 차이를 못 가른다([[project_matcher_flat_chamfer_distinctiveness]]).
원인은 점수 함수가 아니라 **표현(representation)**: Chamfer/NCC 는 "구조가 비슷하다"까지만 보고
"이 구조가 *그 align key* 냐"를 못 본다. metric learning 은 정확히 그 판별 표현을 학습한다.

핵심 자산: **cond.txt 의 crosshair 는 신뢰 가능한 좌표 라벨**(그때 장비가 실제 정렬한 위치).
tool 의 S/E 라벨은 metadata 라 학습 입력 금지([[feedback_doubt_s_labels]]).

---

## 2. 설계 원칙 (불변)

1. **점수형만**: 모델은 후보당 scalar similarity 만 출력. 좌표 선택·stage 전이는 CV. (CLAUDE.md:194)
2. **CV 거부권**: learned score 가 높아도 chamfer/decision 게이트가 low 면 fallback. 학습 점수는
   *blend 항*이지 단독 결정자가 아님.
3. **공짜·신뢰 라벨만**: positive = S crosshair, negative = 같은 FOV 의 proposer decoy. tool S/E 금지.
4. **오피스 내 학습**: 데이터 반출 불가([[feedback_no_office_data_to_mac]]). Mac 은 blind 코딩, 오피스
   pull 후 학습, 텍스트 digest 로 피드백.
5. **pixel-identity 가정 금지**: augmentation 도 밝기/대비/약한 affine 까지. NCC/SSIM 을 라벨이나
   loss 의 1차 신호로 쓰지 않음([[project_align_key_matching_constraint]]).
6. **런타임 경량**: paused 매칭은 top-k(≤24) patch 만 채점 → CPU ms 급. Windows 런타임에 torch 미설치
   가정 → **onnxruntime** 추론.

---

## 3. 라벨 생성 파이프라인

전부 기존 함수 재사용. 새 스크립트는 `poc/workflow_3/align/diagnostics/` 또는
`poc/workflow_2/`(bench) 에 둔다(학습은 bench 성격 → workflow_2 우선, 검증 후 추론만 workflow_3 포팅).

### 3.1 소스 순회

```python
from poc.workflow_3.align.assets import resolve_assets_auto, iter_recipe_dirs, iter_msr_images, load_gray
from poc.workflow_3.align.cond_file import load_cond, msr_modality
from poc.workflow_3.align.clean_align_image import cursor_to_image, clean_image
```

- `iter_recipe_dirs(root)` → 모든 `<eqp>/<class>/<recipe>` (mtime desc).
- recipe 별 `resolve_assets_auto(eqp_id=, class_name=, recipe_name=)` → `AlignFailAssets`
  (`recipe_om`/`recipe_sem`/`from_msr`).
- `iter_msr_images(assets)` → S*/E* 프레임. **S 프레임만** positive 라벨 소스(E 는 crosshair 없음
  — [[project_e_images_no_crosshair]]).

### 3.2 Positive patch (정답)

S 프레임마다:
1. `cond = load_cond(msr_path)` → `CondInfo`. `cond.crosshair_xy` 없으면 skip(detect fallback 은
   라벨 잡음원이라 학습에선 제외).
2. `gx, gy = cursor_to_image(cond.crosshair_xy)` → image 픽셀 좌표.
3. `mod = msr_modality(cond)` → `'om'|'sem'`. 없으면 recipe 모달리티 폴백, 그래도 없으면 skip.
4. `gray = load_gray(msr_path)`; `clean = clean_image(gray, cond)` (crosshair/box inpaint —
   distractor 제거, [[project_msr_crosshair_cv_distractor]]).
5. template 크기 `(tw, th)`(해당 모달리티 `route_template` 으로 결정) 로 crosshair 중심 crop:
   `_matched_crop(clean, (gx,gy), tw, th, 1.0)` (consensus_cv 재사용) → **positive patch**.

### 3.3 Hard negative patch (decoy) — 핵심

> negative 는 **시스템이 실제로 만드는 decoy** 여야 한다. 그래야 scorer 가 "production 에서
> 헷갈리는 바로 그 후보들"을 가르도록 학습된다.

같은 S 프레임 + 해당 모달리티 template 으로 **현 proposer 를 돌린다**:
```python
from poc.workflow_3.align.matching.engine import compute_align_key_score_ensemble, _resize_template, _frame_patch
res = compute_align_key_score_ensemble(template, gray, scales=DEFAULT_SCALES, policy=STRUCTURE_POLICY)
for c in res.candidates:                     # AlignKeyCandidate: xy, scale, chamfer_score
    if dist(c.xy, (gx,gy)) <= tol_px:        # 정답 근처 → 라벨 충돌, 제외
        continue
    tpl_s = _resize_template(template.raw_image, c.scale)
    patch = _frame_patch(clean, c.xy[0], c.xy[1], tpl_s.shape[1], tpl_s.shape[0])
    if patch is not None: negatives.append((patch, c.chamfer_score))   # hard negative
```
- `tol_px`: GT 허용 반경(예: template 짧은 변의 25%). golden eval 의 hit tolerance 와 일치시킴.
- chamfer 높은 decoy 우선(= hard). frame 당 top-3~5 negative.

### 3.4 라벨 레코드 스키마

per-sample manifest(jsonl) + patch 이미지(또는 .npz 묶음):
```
{ "recipe_key": "<eqp>/<class>/<recipe>", "modality": "sem",
  "msr": "E0007.jpg", "role": "pos|neg",
  "patch_path": "...png", "tw": 96, "th": 96,
  "chamfer": 0.71, "gt_dist_px": 0 }   # pos 는 gt_dist_px=0
```
- **recipe_key 로 그룹화** → train/val **cross-recipe split**(같은 recipe 가 양쪽에 들어가면 누수).
  recipe collision 주의([[project_consensus_sparse_golden_and_recipe_id_collision]]) — eqp/class/recipe
  3요소로 키.
- 데이터 현실: 298 recipe, S 희박. → recipe-specific 모델 불가, **cross-recipe 일반 "align-key-ness"**
  학습이 유일하게 현실적.

### 3.5 라벨 생성기 = golden 하니스 재사용

`poc/workflow_2/golden_localization_eval_cond.py` 의 `_process_msr_cond()` 가 이미 GT crosshair 추출
+ 모달리티 라우팅 + proposer 호출을 한다. 이걸 fork 해 "patch dump 모드" 추가(eval 숫자 대신 patch 저장).
production 코드는 무수정([[feedback_ensemble_dev_in_workflow2_then_port]]).

---

## 4. 모델 아키텍처

### 4.1 형태: pair similarity (권장) vs single-patch classifier

- **권장: pair**(template patch, frame patch) → similarity ∈ [0,1]. template 이 recipe 마다 다르므로
  pair 가 일반화에 유리(few-shot 성격). Siamese: 두 patch 를 같은 encoder 로 임베딩 → cosine.
- single-patch("이게 align-key 처럼 생겼나")는 template-agnostic 라 더 단순하지만, "*그* 키냐"를 못 봄.
  → pair 우선, single 은 보조 feature.

### 4.2 입력 표현

- grayscale **+ edge map** 2채널(엔진 전처리 `preprocess_for_matching` 재사용). edge 채널이
  pixel-identity 의존을 낮춤(도메인 제약과 정합).
- patch 크기 통일(예: 96×96 resize). 작은 backbone(MobileNetV3-small / 4-block CNN, embed 64~128d).

### 4.3 Loss

- contrastive/InfoNCE: anchor=template patch, positive=GT frame patch, negatives=decoy patches(같은
  frame). batch 내 다른 recipe 도 negative 로(easy negative). margin/temperature 튜닝.
- **self-supervised pretrain 옵션**: 라벨 적을 때 unlabeled 오피스 프레임으로 DINO/SimCLR pretrain →
  metric head fine-tune. 데이터 양 보고 결정.

### 4.4 런타임 export

- 학습 torch → **ONNX** export. Windows 런타임은 `onnxruntime`(CPU) 만. import-guard 패턴
  (`ONNX_AVAILABLE`) 으로 미설치 시 자동 비활성.

---

## 5. 학습 절차 (오피스 내)

- 스크립트 무인자(no argparse), config 는 env/`Workflow3Settings`, Korean docstring, `[INFO]` print.
- augmentation: 밝기·대비 jitter, ±작은 회전/translation, blur. **금지**: pixel-level 복제 가정,
  강한 호모그래피(매칭이 풀 일을 augment 가 대신하면 안 됨).
- split: cross-recipe(recipe_key 기준 8:2). 같은 recipe 누수 금지.
- 조기 stopping 기준 = **downstream rank1**(아래 §7), val loss 아님(loss↓ ≠ 매칭↑).

---

## 6. 추론 통합 (workflow_3)

### 6.1 어디에 끼우나

`engine.py` ensemble 의 reranker 항. 현재:
```
sel = rerank_chamfer_w * chamfer + rerank_ncc_w * max(0, ncc)      # 0.5/0.5
```
변경(플래그 on 시):
```
sel = w_ch*chamfer + w_ncc*max(0,ncc) + w_ml*metric_score(template_patch, cand_patch)
```
- cand patch 추출은 `_resize_template`+`_frame_patch` 재사용(라벨 생성과 동일 경로 → train/serve skew 최소).
- **CV 거부권 유지**: `decision`(match/adjust/low) 게이트는 그대로. metric 은 *순위*만 바꾼다.
- 임계 재보정: blend 후 Youden 으로 `ensemble_match/adjust_threshold` 재계산(golden 위에서).

### 6.2 플래그 / kill switch

- `ALIGN_FAIL_METRIC_SCORER`(기본 0). on 이고 ONNX 모델 로드 성공 시에만 활성.
- 모델 로드 실패/예외 → 자동 off + `[WARNING]`, 기존 0.5/0.5 reranker 로 폴백.
- 가중치 `w_ml` env 노출(0 이면 사실상 off).

---

## 7. A/B 평가

`golden_localization_eval_cond.py` 위 A/B(baseline reranker vs +metric):

| 지표 | 기대 방향 | 합격선(초안) |
|---|---|---|
| **rank1** | ↑ | 유의 상승(주 KPI) |
| **wrong_local_peak %** | ↓ | 67% → 유의 감소 |
| recall@8 | 불변 | proposer 안 건드리므로 유지 |
| second_ratio 분포 | 분리↑ | 정답/decoy gap 증가 |
| cross-recipe val | 일반화 | held-out recipe 에서도 lift |

합격 시에만 workflow_3 포팅. shadow 모드(점수만 기록, 결정 미반영)로 1차 출시 권장.

---

## 8. 리스크 & 가드

| 리스크 | 가드 |
|---|---|
| 작은 라벨셋 overfit | cross-recipe split, augmentation, self-sup pretrain, downstream-metric early stop |
| recipe 편향 | recipe_key 그룹 split, recipe 별 sample 상한 |
| 라벨 잡음(detect fallback) | crosshair 'cond' 소스만 positive, detect 제외 |
| train/serve skew | 라벨·추론 patch 추출을 동일 함수(`_frame_patch`)로 |
| 런타임 비용 | top-k≤24 만 채점, CPU onnxruntime, paused 경로 한정(live 금지) |
| 경계 위반 우려 | 점수만, decision 게이트 불변, CV 거부권 |

---

## 9. 단계 체크리스트

- [ ] (bench) `_process_msr_cond` fork → patch-dump 모드, manifest jsonl 생성기
- [ ] (bench) negative mining = 현 ensemble proposer 후보, tol_px 정의, cross-recipe split
- [ ] (office) 라벨셋 생성 + 분포 digest(pos/neg 수, recipe 수, 모달리티 비율)
- [ ] (Mac blind) Siamese 학습 스크립트(무인자, ONNX export) + augmentation + downstream-metric stop
- [ ] (office) 학습 → val digest → ONNX 산출
- [ ] (bench) golden A/B: rank1 / wrong_local_peak / recall@8 / cross-recipe lift
- [ ] (port) 합격 시 engine reranker blend + 임계 재보정 + `ALIGN_FAIL_METRIC_SCORER` 플래그
- [ ] (ship) shadow 모드 1차 → 검증 후 결정 반영

---

## 10. 비범위 (Out of scope)

- 모델이 좌표를 직접 결정(클릭) — 금지(doc §8).
- proposer 자체를 ML 로 교체 — 별도 트랙(detector-free matcher, 연구문서 §5.2).
- tool S/E 라벨 학습 — 금지.
- live broad-scan 경로에 적용 — real-time 위반, paused 한정.

> 관련: [[project_matcher_flat_chamfer_distinctiveness]], [[feedback_doubt_s_labels]],
> [[feedback_no_office_data_to_mac]], [[project_e_images_no_crosshair]],
> [[project_msr_crosshair_cv_distractor]], [[feedback_ensemble_dev_in_workflow2_then_port]].
