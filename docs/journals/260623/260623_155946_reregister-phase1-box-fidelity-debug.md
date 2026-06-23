# 재등록 리포트 Phase 1 — box-fidelity 버그 추적 및 해결

날짜: 2026-06-23 15:59
대상: `poc/workflow_2/golden_reregister_report_cond.py` (offline CV 벤치, 재등록 우선순위 리포트)

## 1. 진행 사항

재등록 리포트(`golden_reregister_report_cond.py`)의 box-suggestion 경로에서 **fidelity 가 전 recipe 에서 0** 으로 떨어져 제안이 0건(`w_sugg=0`)이던 문제를 끝까지 추적해 해결했다. 오피스 데이터를 Mac 으로 반입할 수 없어, blind 작성 → 오피스 실행 → digest/경고 라인 relay 의 반복으로 진행했다.

- **버그 진단**: `_compute_fidelity_from_patch` 가 rcp sub-crop(엔지니어 whitebox)을 msr S 프레임에 매칭한 뒤, 후보를 **crosshair(gt_xy)** 근처에서만 받아들였다. 그러나 매칭 엔진은 후보 `xy` 를 **patch 중심** 위치로 돌려주고 `align_offset` 을 적용하지 않는다(`engine.py` 확인). off-center 박스(특히 OM unique-area)는 후보가 crosshair 근처에 절대 안 떨어져 fidelity 가 전부 0.
- **codex:rescue 2회 활용**: (1) offset 픽스의 기하 검증(부호·scale 곱 모두 correct 확정), (2) all-zero 잔존 원인 진단(코드 추론으로 좌표계/풀/예외 가설 표 제시).
- **/code-review (8각도)**: offset 픽스 리뷰 — 핵심 기하 정확 확인, sign-collision 가드/테스트 결합/경고 문구 등 안전 개선 반영, tolerance/scale 일관성은 pre-existing 한계로 문서화.
- **단계적 자가-진단 계측**: all-zero 경고를 exc/empty/offtarget 로 분해 → call-site 태그(baseline/cand) + top1 후보 덤프 추가. 오피스 한 줄 relay 만으로 원인을 좁혔다.
- **오피스 A/B 실측 해석**: old band(scale 0.60, distractor 100~150px off) → tight band(scale 0.85, nearest 33~38px = 박스 위치). 참 localization 은 short 의 0.20~0.24, distractor 는 ≥0.42 로 깨끗이 분리됨을 확인.
- **결론**: tight scale band + tolerance widen 으로 `w_sugg 0 → 1`(양 modality). box-fidelity 경로 정상화. SEM strong ~68% 는 **별개의 SEM center-crop proposer-recall 축**(이 thread 아님)으로 분리.

## 2. 수정 내용

모두 `poc/workflow_2/` 하위. 커밋 `f7501fb`..`b7cb567` (main 직접 push). 테스트 38 passed.

- **`f7501fb`** — `golden_reregister_report_cond.py`: off-center box fidelity 보정. `_box_offset_xy(box, w, h)`(= box_center - frame_center) 신설, near-test 를 `expected = gt_xy + offset*c.scale` 로 변경, `_suggest_for_row` 4개 call-site 에 offset 전달. 기본 `(0,0)` 으로 중심 박스 동작 보존. 부호는 엔진 `align_offset_xy`(image_center - box_center)의 **반대** — 주석으로 명시. TDD 3 테스트(`test_reregister_report.py`).
- **`47cda7e`** — all-zero 경고를 `exc/empty/offtarget` 로 분해(어느 경로 실패인지 한 줄 보고).
- **`6aa23af`** — 경고에 call-site 태그(`baseline-sel/val`, `cand`, `chosen-val`) + 엔진 top1(최고 score) 후보 xy/scale/score/dist 덤프 추가.
- **`ca31403`** — all-zero 경고를 `baseline*` 태그에만 한정. `cand`(비변별 후보 박스)가 distractor 에 off-target 매칭되어 0 나는 건 **정상 기각**이라 noise 제거.
- **`463426b`** — `_FIDELITY_SCALES` 신설(기본 tight band `(0.85,1.0,1.15)`). 작은 box crop 이 최소 scale 0.60 으로 줄어 주기 SEM distractor 에 매칭되던 것을 차단. `_resolve_fidelity_scales` + env `REREGISTER_FIDELITY_SCALES`(A/B). `run()` 이 활성 band 를 자가-라벨로 출력.
- **`7284572`** — `_cap_recipes` + env `REREGISTER_MAX_RECIPES`(fast-mode, 앞 N개만). 전체 sweep 가 >10분이라 빠른 A/B 용.
- **`8dbe880`** — `REREGISTER_MAX_RECIPES` / `REREGISTER_FIDELITY_SCALES` 를 `golden_eval_config`(loader + example)로 브리지.
- **`b7cb567`** — `_FIDELITY_GT_TOL_NORM` 기본 `0.20 → 0.30`(env `REREGISTER_GT_TOL_NORM`, config 브리지). 참 localization(0.20~0.24)을 1~6px 차로 놓치던 것을 잡고 distractor(≥0.42)는 계속 기각.

변경 파일: `golden_reregister_report_cond.py`, `test_reregister_report.py`, `golden_eval_config.example.py`, `golden_eval_config_loader.py`.

## 3. 다음 단계

- **(선택) 전체 confirm run**: cap 없이(전체 67 recipe), tight band + tol 0.30 기본으로 1회 실행해 `w_sugg` 전수 확인. `[DIGEST]` relay 받아 제안율 sanity-check.
- **Phase 2 (E-frame confirmation) 플랜 작성** — 이번 세션 후속으로 바로 진행. Phase 1 은 S-only latent-risk screening(생존 편향 정직). Phase 2 는 같은 free-search 를 E(fail) 프레임에 돌려 latent → **confirmed** 로 승급(`evidence_tier` 모델에 `E_CONFIRMED` 슬롯 이미 예약). 설계 결정 사항(E 프레임 수집 경로, 승급 임계, 출력/digest 변경)은 brainstorming 으로 확정 후 plan.
- **(별개 축) SEM proposer-recall**: SEM strong ~68%(center-crop `_gt_in_topk`)은 box fidelity 와 무관한 장기 병목. template-bank / VLM-ROI 레버 필요. 이 thread 아님 — 별도 작업으로 분리.

## 4. 메모리 업데이트

- `memory/project_reregister_report_phase1.md` 업데이트 완료 — box-fidelity 디버그 전 과정(offset 픽스 → tight band → tol widen, `w_sugg 0→1`, SEM strong 별개 축) 기록. 신규 env/config knob(`REREGISTER_MAX_RECIPES` / `REREGISTER_FIDELITY_SCALES` / `REREGISTER_GT_TOL_NORM`, 모두 `golden_eval_config` 브리지) 명시.
- 루트 `MEMORY.md` 인덱스는 이미 해당 메모리를 가리키고 있어 추가 변경 없음.
