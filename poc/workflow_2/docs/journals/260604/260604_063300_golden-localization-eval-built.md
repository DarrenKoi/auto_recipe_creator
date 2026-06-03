# golden S-only 위치추정 검증 스크립트 구축 — `golden_localization_eval.py`

> 작성: 2026-06-04 · 대상: `poc/workflow_2/golden_localization_eval.py`
> 관련: `docs/align_success_dataset_plan.md`, `docs/study/reranker_ab_failure_analysis.md`,
>        `align_index_ablation.py`(2×2 구조 차용), `align_similarity.py`(헬퍼 출처)

---

## 0. 왜 만들었나 (한 줄)

"흰 box·crosshair 제거가 끝난 뒤, **GT 가 믿을 만한 성공(S) 데이터에서 매처가 align point 를
실제로 맞히는가**" 를 분류(bACC)가 아니라 **위치추정 정오(거리)** 로 직접 재기 위해.

## 1. 기존 지표의 한계 (이 스크립트가 푸는 문제)

`align_similarity.py` / `align_index_ablation.py` 의 주 판정은 **S/E 분리도(bACC)** 였다. 그런데:

1. **과제 불일치** — bACC 는 "key 있음/없음" *분류* 인데, 생산 과제는 "좌표 찾기" *위치추정*이다.
2. **cross-recipe confound** — recipe 마다 template/배율이 달라 절대 align score 의 baseline 이
   다른데, bACC 는 전 recipe 의 S/E score 를 한 전역 임계로 자른다. 특히 **S 없는 recipe의 E** 가
   같은 recipe S 앵커 없이 전역 E pool 에 섞여 들어가 임계를 편향시킨다.
3. **E 의 GT 불신** — E 의 crosshair 는 정의상 *틀린 위치/없음*이라 위치 GT 로 못 쓴다.

→ "내 방식이 정확한가" 의 올바른 검증은 **GT 가 믿을 만한 S 에서의 위치추정**이다. 그 데이터가
의도 수집한 golden set(`align_images_golden/`, S-only). 위치추정 hit/miss 는 *recipe 안에서
자기 crosshair* 로 판정하는 **장당 이진값**이라, bACC 의 cross-recipe confound 에 면역이다.

## 2. 평가 규약 (사용자 확인 2026-06-04)

- **흰 box 제거 = rcp template 쪽.** 등록 이미지의 흰 unique-area box *안쪽만* photometric 으로
  깨끗이 crop 한 `box` template 을 쓴다. 구 `center`(면적 crop)와 A/B.
- **crosshair 제거 = msr frame 쪽.** crosshair 를 **먼저 검출해 그 위치를 GT 로 고정**한 뒤,
  그 자리를 inpaint 로 지운 frame 에서 매칭한다. 즉 *정답은 보존, 매칭은 crosshair 없는 wafer 에서*.
  - 이유: live 생산엔 crosshair 가 없다(우리가 찍을 위치를 *찾는* 중). inpaint frame 이 생산에
    더 충실하고, 매처가 crosshair 의 강한 edge 에 "치팅" lock-on 하지 못하게 한다.
- **점수화:** 정답지가 이미 있으므로 분류가 아니라 **거리 정오**. inpaint 후에도 best_xy 가
  *원래 crosshair 위치* 의 허용오차(`GT_TOL_NORM`=0.20, template 짧은 변 대비) 내면 hit.
- **검출/inpaint 순서가 핵심:** raw 에서 crosshair 검출 → GT 기록 → 그 위치 inpaint → 매칭.
  검출 실패(GT 없음) 행은 위치추정에서 **제외**(합성 fallback 없음 — 생산과 동일).

## 3. 설계 — 2×2 (`align_index_ablation` 구조 차용, 지표만 교체)

```
              frame=raw          frame=inpaint
tpl=center    OLD baseline       crosshair-only
tpl=box       box-only           NEW (box + inpaint)
```

verdict = **NEW(box+inpaint) − OLD(center+raw)** 의 지표 상승폭. box-only/crosshair-only 셀이
어느 변수가 효과의 주범인지 분해한다. (ablation 과 같은 2×2 이나, 셀 지표가 confound 있는 bACC →
confound 없는 위치추정으로 바뀐 것이 핵심 차이.)

## 4. 지표 정의 (셀별, S+crosshair 행만)

| 지표 | 정의 | 읽는 법 |
|---|---|---|
| `rank1_hit_rate` | best **align point** 가 crosshair 허용오차 내인 비율 | **주 판정** — 생산 "1발 명중률" |
| `gt_in_topk_rate` | 정답이 chamfer top-N 후보에 드는 비율 | proposer recall (보조) |
| `topk_not_rank1_rate` | 후보엔 있으나 1등이 아닌 비율 | 리랭킹으로 메울 수 있던 갭 (참고; 리랭커는 폐기됨) |
| `median/p90 dist_norm` | best align point 정규화 거리 분포 | 명중의 *정밀도* |

**중요 — align_offset:** 매칭이 잡는 위치는 *template 중심*(center crop 중심 / box 안쪽 crop
중심)이지만 recipe 의 align point 는 *이미지 중심*이다. 따라서 match 중심·top-N 후보 좌표에
`align_offset = image_center − template_center` 를 더해 **align point 로 환산한 뒤** crosshair 와
비교한다(생산 `_build_rcp_template` 와 동일 규약). center 는 offset (0,0); box 가 off-center 면
offset≠0. 이 보정을 빼면 box 셀이 *box 중심 정확도*를 재게 되어 verdict 가 오도된다(Codex 리뷰
2026-06-04 에서 발견·수정).

**GT 위생(전역, 지표 신뢰도 게이트):** S 장수, **crosshair 검출률·평균 conf**, label `?`/`E` 수.
golden 은 S-only 가정이라 `E`/`?` 가 있으면 라벨 오염 경고. crosshair 미검출 S 는 GT 없음 → 제외.
(검출률은 fail S 에서 79% 였음 — golden 에서 다시 측정해 GT 신뢰도를 먼저 확인.)

## 5. 구현 메모

- **offset-aware 로컬 빌더(`_build_offset_templates`)**: `align_similarity._build_templates` 는
  offset 없는 bare template 을 반환하므로 재사용하지 않는다. 대신 production 헬퍼
  (`_centered_area_crop_bbox`/`_detect_white_box`/`_inner_crop_for_box`/`build_template`)로
  modality 별 `(template, align_offset)` 를 만든다. center=(0,0), box=image_center−inner_center.
- **`_localize`(신규 핵심)**: modality 당 `compute_align_key_score` 1회 → match 중심·후보에 offset
  가산 → align point 로 crosshair 거리 채점. modality race 는 생산 free-best 와 같은 *최고 score*.
  rank1_hit/dist 와 topk_rank 를 *같은 후보 집합*에서 일관되게 뽑는다(과거 `_race`+`_gt_in_topk`
  이중 패스 + box offset 누락 버그를 한 번에 제거).
- **재사용 상수/헬퍼:** `STRUCTURE_POLICY`/`COMPARE_SCALES`/`GT_TOL_NORM`, `detect_crosshair`,
  `_inpaint_crosshair`, `_tool_label`.
- **overlay(`_save_overlay`)**: S 한 장마다 inpaint frame 위에 GT(초록 십자+원) vs 예측 align point
  (box=주황, center=시안)를 선으로 이어 그려 `overlays/<recipe>/<msr>_overlay.jpg` 로 저장.
  점수뿐 아니라 *어디를 찍었나* 를 눈으로 본다. `SAVE_OVERLAYS` 로 토글. inpaint 잔흔도 여기서 확인.
- **golden 루트:** 기본 `align_images_golden/`(fail 트리의 형제), env `ALIGN_GOLDEN_ROOT` override.
  `iter_recipe_dirs(root)` / `resolve_assets_auto(root=...)` 가 root 인자를 받으므로 reader 재사용.
- **단일 recipe:** env `ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME`(golden 루트 기준).
- **CLAUDE.md 규약 준수:** Korean docstring(`run()` 포함), print 로깅, argparse 없음, 절대 import,
  JPEG 디버그. (표 본문 행은 sibling `align_index_ablation._print_summary` 와 동일하게 `[INFO]`
  헤더 아래 무접두 — 표 가독성 유지.)

## 6. self-test (Mac/데이터 없이 검증됨, 2026-06-04)

golden 데이터가 없으면 **임시 트리를 합성**해 reader→template→crosshair→위치추정 전 경로를 돈다:

- rcp: 패턴(<200 으로 눌러 유일한 255 island)+ 흰 box outline. box 면적 ≈ 19%(`<40%` 상한),
  가장자리 비접촉, 얇은 hollow → `_detect_white_box`(photometric) 게이트 통과하도록 맞춤.
  **RCP_A = 중앙 box(offset 0), RCP_B = off-center box(shift (24,16) → offset≠0)** 로 둬서
  offset 보정 경로를 회귀 가드한다(보정이 빠지면 RCP_B box 셀 rank1_hit 가 떨어짐).
- msr S: wafer 배경 + drift 패턴 + *align point*(= 중심 + align_shift, align_shift=−box_shift)에
  full-span 밝은 십자(검출되도록).
- 결과(2 recipe × 4 S, crosshair 8/8 검출): **box__raw/box__inpaint rank1_hit=1.0**(off-center
  RCP_B 도 offset 보정으로 정확 명중, med_dist 0.016). **center__raw=0.75**(offset 모르는 center 는
  RCP_B 의 shifted align point 를 못 맞힘) → **verdict NEW better, rank1_hit delta +0.25** (비퇴화).
  overlay 8장 저장 — GT(초록) vs box(주황 d=0.016 HIT)/center(시안)로 명중을 눈으로 확인.

```bash
uv run python poc/workflow_2/golden_localization_eval.py   # 데이터 없으면 self-test
```
출력: stdout 표 + `DEBUG_IMAGE_DIR/golden_localization_eval/<ts>/{rows.jsonl, summary.json,
overlays/<recipe>/<msr>_overlay.jpg}`.

## 7. 오피스 실행 절차 & acceptance

1. **수집(사용자):** `align_images_golden/<eqp>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr}`
   (msr = S 만). recipe 다양하게, recipe 당 ≥ 8~10 S, **wafer/lot/시간 분산**(한 wafer 몰빵 ❌).
2. **GT 위생 먼저:** 출력의 crosshair 검출률·`?`/`E` 수 확인. 검출률이 낮으면(GT 오염) 그 위에서
   나온 위치추정 수치를 믿지 말 것 — 검출기부터 점검.
3. **방식 정확성 판정:** `box__inpaint` 의 `rank1_hit_rate` 가 매처의 진짜 1발 명중률.
   `center__raw`(OLD) 대비 delta 가 cleanup 의 *위치추정* 효과(confound 없는 판정).
4. **다음 단계 연결:** 방식이 통과하면(높은 rank1_hit) `success_vs_fail_compare.py` 로 임계
   calibration(거짓양성 0 검증) → fail 데이터에 적용 → drift recipe 가이드라인(별도 워크스트림).

## 8. 한계 / 주의

- inpaint 가 crosshair 자리에 잔흔(faint artifact)을 남기면 매처가 거기에 약하게 lock 할 수 있다 →
  오피스 실데이터에서 inpaint 품질을 overlay 로 한 번 눈으로 확인할 것(`stronger inpaint verify` 계열).
- **현 `align_images/`(fail 트리)의 S 로 돌리면 안 된다.** 그건 *결국 실패한 run 의 성공 step* 이라
  표본 적고 편향(`align_success_dataset_plan.md` §7). 이 스크립트의 대상은 *의도 수집한 golden*.
- self-test 가 1.0 인 건 합성이 쉬워서다 — 실데이터의 절대 수치는 다를 것. self-test 는 *파이프라인
  정상 동작*만 보증한다.
