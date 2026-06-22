# SEM whitebox box-crop in the consensus arm — 설계

작성: 2026-06-22 · 대상 코드: `poc/workflow_2/align_similarity.py`
(`_gt_in_topk`, `_consensus_template_ab`), `poc/workflow_2/golden_consensus_eval_cond.py`
(`_build_cond_by_recipe`), `poc/workflow_2/golden_eval_config.py`(+example, loader)
재사용: `poc/workflow_3/align/cond_template.py`(`cond_template_crop`/`cond_align_offset`),
`AlignKeyTemplate.align_offset_xy`(engine.py:78)
선행: per-modality consensus eval(`20efe49`), consensus proposer = 프로덕션 ensemble 거울
(`c8d5571`); oracle-ROI 폐기(`2026-06-22-oracle-roi-ceiling-design.md`, superseded)

---

## 1. 배경 / 문제

consensus arm 의 SEM `recall_miss ~15%`(rank1 0.71 vs OM 0.91)는 full ensemble+C4 에도
살아남는다 — 정답이 후보 pool 에 아예 없는 **proposer recall** 문제
([[project_edge_ncc_consensus_ab_3arm]]).

**도메인 흐름(사용자 확인, 2026-06-22)** — [[project_om_sem_mag_key_fills_frame]]:
- SEM 은 고배율이라 align-key 가 프레임의 80~100% 를 채운다(OM 은 10~20%).
- 엔지니어가 recipe 등록 시 그 **zoom-in 된 key 안의 특정 위치를 whitebox 로 지정**한다.
  즉 whitebox 는 generic "unique area 힌트"가 아니라 **반복/자기유사 구조 안에서 "이 위치가
  align target"임을 사람이 직접 못 박은 distinctive anchor** 다.

→ SEM recall_miss 의 정체: matcher 가 whitebox(=distinctive anchor) 가 아니라 **자기유사한
key 전체**를 매칭해 wrong-phase peak 에 lock 된다. truth 가 후보에서 밀리는 건 변별 정보를
*안 쓰고 있기* 때문.

**핵심 격차(확인됨):** box-crop(=whitebox crop)은 rcp **localization arm**(`box__inpaint`,
`golden_localization_eval_cond.py`)에서만 검증·포팅됐고, 그것도 **OM/SEM 혼합** 데이터에서
**localization(변위 bin)** 기준이었다(2026-06-11; per-modality 층화는 그 *이후* ADR 0004).
정작 SEM recall_miss 가 사는 **consensus arm 은 whitebox 를 전혀 안 쓴다**:
`_build_cond_by_recipe` 의 consensus crop 은 **crosshair 중심·center-tpl 크기**
(`golden_consensus_eval_cond.py:258,297`)라 주변 주기 구조를 그대로 포함 → median 이 generic
→ 여러 phase 에 매칭. 즉 whitebox 변별 신호가 *정작 필요한 SEM consensus 경로에서 버려진다.*

이전 "box-crop 차이 없음" 결론은 (a) localization arm, (b) OM/SEM 혼합이라 **SEM-specific
효과를 평균이 가렸다**(OM 은 key 가 작고 이미 변별적 recall 0.91 → crop 에 둔감; 그게 혼합
평균을 끌어내린다). consensus arm × per-modality 는 미측정 영역이다.

## 2. 목표 / 비목표

**목표**
- consensus arm 에서 template crop 영역을 A/B: **center-crop(현행) vs whitebox box-crop**,
  `recall_miss`(truth-in-pool) 기준, **OM vs SEM 층화 보고**.
- 가설: box-crop ≫ center on **SEM** recall(whitebox anchor 가 phase tie 를 깸); OM 은 ~flat
  (이미 변별적). 혼합/localization 테스트가 못 본 SEM-specific 레버를 노출.

**비목표 (YAGNI)**
- 단일 정책(box-crop 을 OM·SEM 양쪽 적용)으로 빌드하되 **보고만 per-modality** 로 쪼갠다 —
  데이터가 어디서 돕고 어디서 해치는지 보이게(사용자 선택). modality-specific on/off 분기는
  결과 본 뒤 결정.
- 프로덕션(`workflow_3`) 변경 없음 — SEM 양성일 때만 `consensus_resolve`/templates 로 포팅.
- 새 데이터 계약 없음 — cond.txt 의 `box_ltrb`(이미 존재) + 기존 history S 풀 재사용.

## 3. 설계

### 3.1 Component 1 — box crop 재료: `_build_cond_by_recipe`

각 S frame 에 기존 center `crop` 옆에 `crop_box` 추가: cond `box_ltrb` 로 box-crop,
crosshair 제거(inpaint), **고정 box-size**(rcp box tpl 크기 → modality 내 median 동일 크기
보장), co-register(modality 별). rcp box template + 그 `align_offset_xy`(= image_center −
box_center)도 `rcp_tpls_box` 로 운반. center 경로는 불변.

box-crop/offset 은 신규 구현 금지 — `cond_template.cond_template_crop()` +
`cond_align_offset()` 재사용(프로덕션과 bit-동일, [[project_align_cond_files_and_coords]]).

### 3.2 Component 2 — 핵심 단위: `_gt_in_topk` align_offset 지원

box template 은 match center 가 **box 중심**에 떨어진다(align point=crosshair 아님). 현재
`_gt_in_topk` 은 candidate xy 를 crosshair 와 직접 비교(center tpl 은 offset≈0 라 OK). box arm
이 공정하려면 **predicted align point = candidate_center + align_offset_xy × best_scale** 로
매핑한 뒤 truth(crosshair) 거리 비교. 안 하면 box arm 이 offset 만큼 인위적으로 나빠 보인다.

- `align_offset_xy == (0,0)`(center tpl) → 현행과 **byte-identical**(regression test 고정).
- candidate 에 `scale` 이 있으므로 offset 을 best_scale 로 환산(engine.py:78 계약과 동일).

### 3.3 Component 3 — box arm: `_consensus_template_ab`

center 측정 옆에서 `crop_box` 로 `cons_tpl_box`(align_offset 부여) 빌드 →
`_gt_in_topk` → recall. `(modality, arm∈{center,box})` 별 집계. center arm 불변.

**측정 계약 — 고정 denominator(survivorship 금지, oracle spec Codex 리뷰 재사용).** 각
(modality, arm) 분모 = full-frame center arm 에서 유효(`gc is not None`)했던 *동일* eval set.
그 안에서 hit/miss; 후보 0개·crop<template·예외 = **miss**(skip 아님). 반환:

```
res["box_crop_ab"] = {
  "per_modality": {mod: {arm: {"recall": h/e, "rank1": r1/e,
                               "n_eval": e, "n_hit": h, "n_no_candidate": nc}}}
}
```

### 3.4 Component 4 — config + digest

`golden_eval_config.py` 에 `CONSENSUS_BOX_CROP`(기본 off) 추가(+example, loader.seed_env).
`golden_consensus_eval_cond` 가 읽어 passthrough + per-modality center-vs-box recall digest
출력(SEM 행에 box−center delta + `n_eval/n_hit` 동반). `golden_combined` 자동 노출.

## 4. 실행 / 산출 / 결정

오피스 1회 `CONSENSUS_BOX_CROP=1`. SEM 행:
- **SEM box recall ≫ center**(OM ~flat) → 가설 확정 → **box-crop 을 프로덕션 consensus 경로
  (`consensus_resolve`/templates)로 포팅**. OM flat = 무회귀 확인.
- **SEM 도 flat / box ≤ center** → whitebox anchor 도 phase tie 못 깸 → 다음 후보(재등록
  정량화 / lower-mag context)로 선회. cheap redirect.

digest 가 `recall` 옆 `n_eval/n_hit/n_no_candidate` 동반 → recall 변화가 진짜 회복인지 분모
축소인지 자가 점검.

## 5. TDD 계획 (vertical slices)

1. **align_offset 적용**: 알려진 offset 의 box template → candidate(box center) → predicted
   align point = box center + offset×scale 가 truth 와 일치(in_topk). offset=(0,0) → 현행과
   byte-identical(regression guard).
2. **box crop 재료**: cond `box_ltrb` 로 일관 크기 box crop 생성(co-register, crosshair 제거),
   `cond_template` 경로 재사용 확인.
3. **box arm 집계**: `_consensus_template_ab` 가 per-(modality, arm) **고정 분모** 카운트
   (`n_eval/n_hit/n_no_candidate`) 반환; 기본 off.
4. **메커니즘**: 합성 — 주기 프레임 + whitebox 안에만 distinctive feature. box arm recall >
   center arm(whitebox anchor 가 phase tie 를 깸).

## 6. 위험 / 완화

- **box-crop 이 template 을 작게 만들어 주기장에서 *더* 모호** → whitebox 는 정의상 distinctive
  spot 을 감싸므로 net 이득 가설. 합성 test4 + 오피스 SEM delta 로 실증; 음성이면 선회.
- **align_offset 부호/스케일 오류** → test1 이 offset≠0 을 직접 고정(byte-identical offset=0 +
  알려진 offset 매핑). 프로덕션 `cond_align_offset` 재사용으로 bit-drift 0.
- **혼합 평균 재현** → 반드시 per-modality 분모로 보고(전역 평균 금지). SEM/OM 분리가 본 실험의
  핵심 교훈.
- **survivorship** → oracle spec 의 고정-분모 계약 재사용(no-candidate=miss).
