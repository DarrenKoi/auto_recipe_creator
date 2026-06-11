# Tier 1.1 — cond-box-crop template production 포팅 설계 (2026-06-11)

> 로드맵 [matching-improvement-roadmap](../../../../workflow_2/docs/superpowers/specs/2026-06-11-matching-improvement-roadmap.md)
> §1.1 의 production 포팅 단계. **검증(GREEN-LIGHT) 통과** 후 작성된 설계이며, 실제 구현은
> 별도(나중). 검증 harness/결과는 workflow_2 lab, 본 포팅은 workflow_3 production.

## 0. 배경 — 무엇을 왜 포팅하나

검증(`golden_localization_eval_cond.py`, office 1회 실행, 2026-06-11)에서 cond.box_ltrb 로
crop 한 box template(+decoupled offset)이 center-area crop 대비 **모든 displacement bin 에서
localization 동반 상승**:

| bin | center gt_in_topk / rank1 | box gt_in_topk / rank1 | gt_in_topk lift | rank1 lift |
|-----|---------------------------|------------------------|-----------------|------------|
| near | 0.567 / 0.437 | 0.677 / 0.596 | +0.110 | +0.159 |
| mid | 0.641 / 0.527 | 0.771 / 0.687 | +0.130 | +0.160 |
| far | 0.571 / 0.381 | 0.656 / 0.557 | +0.085 | +0.176 |
| veryfar | (표본 0) | (표본 0) | — | — |

가설(far/veryfar 구조적 rescue)보다 강한 **균일 변별력 향상**. production headline =
**rank1 +0.16~0.18**(=올바른 reposition 비율). 따라서 displacement 라우팅 없이 무조건 포팅.

핵심: box-crop 은 *매칭 알고리즘* 변경이 아니라 **입력 template 변경**이다. ensemble matcher
(`compute_align_key_score_ensemble`)는 그대로, 더 변별력 있는 template 을 먹인다.

## 1. 결정 사항 (확정)

- **fallback (cond box 없음/skip)**: **center-area crop + offset(0)** — eval 의 center arm
  그대로(gt_in_topk 0.567~0.641, 기존 whole-template ~0.434 대비 우수). 모든 recipe 가 검증된
  arm 으로 전환.
- **기본 활성화**: **cond-box-crop 기본 ON + 끄는 flag**. flag OFF → whole-template(구 동작) 롤백.
  reposition+OK 경로는 Tier 0 `second_ratio` 게이트(engineer_review)가 모호 매칭을 backstop.

## 2. Architecture (컴포넌트)

### 2.1 신규 `poc/workflow_3/align/cond_template.py` (primitives 승격)

lab(`golden_localization_eval_cond.py`)의 검증된 cond 기하 함수를 **byte-identical 승격**:
`cond_align_offset`, `check_cond_box`, `cond_template_crop`, `_cond_box_to_xywh`,
`_cond_box_center`, `cond_offset_norm` + 상수(`CROP_INSET_PX`, `OFFSET_WARN`/`OFFSET_SKIP`,
`MIN_INNER_PX`/`WARN_INNER_PX`)와 `centered_area_crop(gray, ratio)` 헬퍼.

- 좌표 primitive(`cursor_to_image`, `OVERSAMPLE`)·`clean_image` 는 이미 `clean_align_image.py`
  에 있으므로 그대로 재사용(중복 생성 금지).
- 이후 **lab 이 이 production 모듈에서 import**(ensemble 과 동일 bit-parity 패턴). lab 의
  기존 cond 테스트(`test_golden_localization_eval_cond.py`)는 re-export 로 계속 통과.
- 의존 방향: workflow_2(lab) → workflow_3.align(prod), 역방향 금지(CLAUDE.md 규약).

### 2.2 `AlignKeyTemplate.align_offset_xy`

`align.matching.engine.AlignKeyTemplate` 에 `align_offset_xy: tuple[int, int] = (0, 0)` 필드 추가,
`build_template(...)` 에 optional 인자. template 이 자기 offset 을 들고 다닌다(병렬 dict 보다
단순; eval 의 `(template, offset)` 튜플을 prod 에선 template 안으로 흡수).

## 3. Template 빌드 — `build_templates_from_assets` cond-aware 화

각 rcp 이미지(OM=IMAP0001 / SEM=IMAP0002)에 대해:

```
cond = load_cond(rcp_path)
if cond_box_crop and cond and cond.box_ltrb and check_cond_box(cond.box_ltrb, gray.shape)[0] != "skip":
    crop, _bbox = cond_template_crop(gray, cond)                 # box arm (검증 win)
    offset      = cond_align_offset(cond.box_ltrb, gray.shape)
elif cond_box_crop:
    crop   = centered_area_crop(gray, CENTER_AREA_RATIO)         # 검증된 fallback
    offset = (0, 0)
else:
    crop   = gray                                                # flag OFF → 구 whole-template 롤백
    offset = (0, 0)
tpl = build_template(crop, recipe_id=..., key_type=..., align_offset_xy=offset)
```

이로써 현재의 **`crop_template_to_box`(CV 흰박스 검출 + offset 누락) 경로를 완전히 대체**한다.
`extract_annotation_box`(검출) 사용 제거. cond 부재/경계밖/너무작음/offset 과도(check_cond_box
skip)는 자동으로 center-crop fallback.

## 4. Offset 적용 — `correct_align_fail` reposition 한 줄 변경

```
result = compute_align_key_score_ensemble(template, frame, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY)
ox, oy = template.align_offset_xy
align_xy = (result.best_xy[0] + round(ox * result.best_scale),
            result.best_xy[1] + round(oy * result.best_scale))
cx, cy = clamp_to_fov(align_xy[0], align_xy[1], fw, fh, config.click_margin_ratio)   # was: best_xy
# 이하 reposition(move_to_point) + OK 동일
```

- **best_scale 로 scale**: offset 은 rcp px 의 (image_center − box_center) 벡터다. 매칭이
  best_scale 로 잡혔으면 frame px 벡터 = offset × best_scale. eval 은 scale≈1 가정으로 unscaled
  적용했고, paused near-native band 에선 동일하지만, off-native match 에도 옳도록 일반화한다.
- 가시성 게이트(`key_visibility_gate`)는 `result` 를 그대로 읽어 **무변경**(offset 은 reposition
  좌표에만 영향). Tier 0 second_ratio 라우팅도 그대로.

## 5. Config

`CorrectionConfig.crop_template_to_box: bool = False` 를 **`cond_box_crop: bool = True`** 로 교체
(env override는 `Workflow3Settings` 경유 추가 가능). 의미:
- `True`(기본) → cond-aware(box-crop 또는 center-crop fallback).
- `False` → whole-template(구 동작 롤백).

## 6. Tests (Mac, TDD)

- **cond_template primitives**: 승격된 함수의 합성 테스트(대부분 lab 에서 relocate;
  `test_cond_template.py` 신설 또는 lab 테스트가 prod 를 import). centered_area_crop 경계.
- **build_template offset 보존**: `align_offset_xy` 가 template 에 실리는지.
- **★ offset 적용(핵심 신규 동작)**: 알려진 `align_offset_xy` 를 가진 합성 template +
  match 가 알려진 `best_xy`·`best_scale` 로 잡히는 frame → reposition 타깃 ==
  `best_xy + round(offset × best_scale)` 검증(`_FakeController` move_calls 로). scale=1 과
  scale≠1 두 케이스.
- **fallback 분기**: cond 없음 → center-area crop + offset(0); flag OFF → whole + offset(0).
- office e2e(paused frame → box template → align point 착지)는 실장비 단계의 후속 검증.

## 7. 범위 밖 / 후속 (non-blocking)

- **paired lift 정밀화**: 검증의 lift 는 box-arm(box-having recipe) vs center-arm(전체 recipe)
  비교라 약한 confound. box-having frame 만으로 center vs box 짝지은 재집계가 추정을 조인다.
  단, box arm 절대 수치(gt_in_topk 0.66~0.77 / rank1 0.56~0.69)가 이미 강하고 box-crop 은
  box-있는 recipe 에만 적용되므로 포팅 차단 사유 아님.
- **gate threshold box 재보정**: ensemble decision 임계(0.6053/0.4727)는 비-box template 분포로
  보정됨. box template 분포가 다르면 게이트가 약간 보수/공격적일 수 있음(Tier 0.2 영역).
  localization 은 이미 box 로 검증됐으니 차단 아님.
- **veryfar 미검증**: 이 golden 셋엔 표본 0 → 극단 displacement 효과는 미관측(실패 아님).

## 참고

- 검증 설계/플랜: `workflow_2/docs/superpowers/specs/2026-06-11-box-crop-localization-validation-design.md`,
  `workflow_2/docs/superpowers/plans/2026-06-11-box-crop-localization-validation.md`
- lab primitive 원본: `golden_localization_eval_cond.py`(cond_template_crop / cond_align_offset /
  check_cond_box / _build_offset_templates_cond)
- prod 대상: `vision/align_fail_correct.py`(build_templates_from_assets / correct_align_fail /
  key_visibility_gate), `vision/align_key_matcher.py`(AlignKeyTemplate / build_template),
  `vision/clean_align_image.py`(cursor_to_image / OVERSAMPLE / clean_image), `config.py`
- 도메인 메모: [[project_rcp_white_box_unique_area]], [[project_align_cond_files_and_coords]],
  [[project_align_fail_modality_om_vs_sem]]
