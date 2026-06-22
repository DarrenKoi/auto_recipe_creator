---
status: draft
date: 2026-06-23
---

# Re-registration 우선순위 랭킹 리포트 (Phase 1, S-only risk screening)

**Goal:** 골든 align_images 셋을 오프라인 스캔해, align key 재등록(re-registration)이 가장
필요한 recipe 를 **modality 별로 worst-first 랭킹**하고, flagged recipe 에 대해 더 변별력 있는
**교체 whitebox 후보를 제안**하는 workflow_2 벤치 드라이버. Phase 1 은 **성공(S) 프레임만** 쓰는
*latent-risk screening* 이다 (survivorship 한계를 정직하게 명시; E-confirmation 은 Phase 2).

**Architecture:** 신규 드라이버 `golden_reregister_report_cond.py` 가 production 매칭 엔진
(`compute_align_key_score_ensemble`)과 기존 localization 드라이버 머신(`_gt_in_topk` free-search)을
*재사용*해 recipe·modality 별 3-tier 증거를 산출 → 연속(threshold-free) 랭킹 → flagged recipe 박스
제안. 출력은 relayable 텍스트 리포트 + 1줄 `[DIGEST]` + office-only overlay.

**Tech Stack:** Python 3.10+, OpenCV, numpy. `poc.workflow_3.align` 엔진/템플릿/clean.
설정은 `golden_eval_config.py` (gitignore) + `seed_env()` 브리지 (다른 골든 드라이버와 동일).

## Global Constraints

- **No CLI args** — 설정은 `golden_eval_config.py` 상수 + env(seed_env) 만. `uv run python <script>.py` 로 실행.
- **Korean docstrings**, `[INFO]/[WARNING]/[ERROR]` print 로깅(logging 모듈 금지). print() 안 em-dash(U+2014) 금지(cp949).
- **Absolute imports** `from poc.workflow_3.align ...` / `from poc.workflow_2 ...` (workflow_2 가 workflow_3 을 import; 역방향 금지).
- **production 코드 무수정** — 드라이버/헬퍼는 `poc/workflow_2/` 안에만. 엔진(`align/matching/engine.py`)은 읽기 전용 재사용.
- **오피스 데이터 Mac 반입 금지** — accuracy 숫자는 오피스 골든에서만. Mac 은 `py_compile`+합성 테스트+`no_data` 실행만.
- **연속 랭킹** — live 보정용 임계(reregister τ=0.98, match decision τ=0.6053)는 *분포가 다른* 본 용도에 hard flag 로 쓰지 않는다. 리포트에 **참조선(annotation)** 으로만 표기.

---

## 1. 동기와 Codex 리뷰 반영

`feasibility_check.py` 는 이미 호출당 `reregister_recommended`(verdict=ambiguous, second_ratio>τ)를
낸다. 본 작업은 그 신호를 recipe 모집단 전체로 **정량화·랭킹**한다. 초기 설계(rcp-only 단일 self-similarity)는
Codex adversarial review 에서 다음을 지적받아 재설계했다:

- **(반영) success-only 편향** — S 프레임만 보면 실제로 실패하는 recipe(E)를 구조적으로 놓친다. → Phase 1 출력을
  "성공 recipe 중 *잠재 위험* screening"으로 정직하게 reframe. E-confirmation 은 Phase 2.
- **(반영) rcp self-match 가 NMS 지배** — crop 을 자기 이미지에 매칭하면 best peak 은 origin 의 trivial
  self-peak(~1.0). second_ratio 는 엔진 NMS 반경에 좌우되어 *distinctiveness 가 아니라 NMS 반경*을 잰다.
  SEM 은 key 가 자기 이미지의 80~100% 를 채워 슬라이드 여지가 거의 없어 **near-degenerate**. → self-match 는
  **명시적 template-size exclusion zone** 을 두고 **ADVISORY tier(OM 보조)** 로 강등, 단독 트리거 금지.
- **(반영) 제안 박스의 순환성** — 모호하다고 판정한 같은 엔진으로 교체 박스를 고르면 같은 blind spot 을 최적화.
  → **held-out 검증**(선택용 S 프레임 ≠ 검증용 S 프레임) + inpaint-dodge 가드 + "엔지니어 검토 *후보*"로만 표기.
- **(반영) 임계 오적용** — live τ 는 free-search 분포 기준. → 연속 랭킹 + 참조선.
- **(반영) 축 상관 이중계수** — second_ratio 계열(self/msr)·동일 inpaint 프레임 의존 축을 독립 flag 로 세지 않는다.
  → flag 개수 합산이 아니라 **evidence tier**(STRONG>MEDIUM>ADVISORY)로 랭킹.
- **(반영) truth-forced fidelity 의 맹점** — crosshair 위치 점수는 "GT 가 점수 높나"지 "free-search 가 GT 를
  *고르나*"가 아니다. wrong-peak 실패를 놓친다. → STRONG tier 를 **free-search localization**(GT rank /
  free-best displacement / GT-vs-top1 gap)으로 정의(`_gt_in_topk` 재사용).
- **(반영) median 이 brittle 프레임 은폐** — modality당 S 2~3장 median 은 드문 모호 프레임을 가린다. → **worst-case(tail)**.
- **(반영) inpaint 가 metric 화** — 후보가 inpaint 영역을 피해 깨끗한 픽셀로 이기는 가짜 이득. → inpaint-mask overlap 추적.
- **(반영) OM/SEM 기하 차이** — window/exclusion/normalization 을 modality 별로.

## 2. Scope

**Phase 1 (본 spec, IN):**
- recipe·modality 별 3-tier 증거 산출 (S 프레임 + rcp 만).
- 연속 evidence-tiered 랭킹, modality 별 worst-first.
- flagged recipe 박스 제안(held-out 검증, candidate-only).
- 텍스트 리포트 + DIGEST + office overlay.

**Phase 2 (OUT, 후속):** E(fail) 프레임 confirmation — 같은 free-search 를 E 프레임에 돌려 latent→confirmed
승급, confirmed 를 최상위 tier 로. 본 랭킹 모델은 `evidence_tier` enum 에 `E_CONFIRMED` 슬롯을 남겨 무리 없이 확장.

**Non-goals:** production 루프 통합(본 작업은 순수 오프라인 벤치), VLM 사용(전부 classical CV),
align_images 트리/`align/assets.py` 수정.

## 3. 데이터 입력

```
<GOLDEN_ROOT>/<eqp>/<class>/<recipe>/
├─ align_img_from_rcp/  IMAP0001.*(OM)  IMAP0002.*(SEM)     # 등록 key + .<img>/cond.txt(box,crosshair)
└─ align_img_from_msr/  S*/E*                               # 측정 프레임(S=성공). Phase 1 은 S 만.
```

- 모든 recipe 는 cond.txt 보유(전량 다운로드 확인) → no-cond-box 엣지케이스 없음.
- modality 배정: cond 규약 재사용. **주의(리뷰 확인): `_route_modality`(localization 드라이버)는 recipe-cond 당
  modality 1개를 주지 프레임당이 아니다.** 한 측정은 같은 마크의 OM 2장/SEM 3장을 내므로, S 프레임은 폴더 규약
  (`align_img_from_msr` 의 modality 하위/파일 태그)로 **프레임별** modality 를 배정하고, 해당 modality 의 rcp
  키(OM↔IMAP0001, SEM↔IMAP0002)로만 매칭한다. 프레임별 배정 규칙은 plan 에서 기존 `_process_msr_cond` 의
  프레임 modality 결정 로직을 그대로 따른다.
- msr S 프레임은 매칭 전 `align.clean_align_image.clean_image(...)` 로 crosshair inpaint(TELEA) — 기존 규약.
  **함수명은 `clean_image`**(모듈이 `clean_align_image`; 리뷰 확인). `clean_image` 는 inpaint mask 를 반환하지
  않으므로, 박스 제안 가드(§7.3)용 removal mask 는 같은 모듈의 **`build_removal_mask(...)` 공개 헬퍼를
  `clean_image` 와 동일 파라미터로 별도 호출**해 얻는다(또는 GT crosshair 좌표에서 직접 crosshair 사각형을
  유도 — 둘 다 가능, plan 에서 택1).
- GT = S 프레임 cond.txt 의 crosshair px@5120 → 프레임 좌표 환산. **변환 체인 명시(리뷰 확인):** px@5120 →
  native px(이미지 실해상도 비율) → 프레임 px. `cursor_to_image` 는 oversampled-cursor 공간(OVERSAMPLE 배율)
  변환이라 본 용도와 다르므로 쓰지 않는다; plan 은 px@5120→프레임 px 환산을 명시적으로 구현(기존 cond
  파서가 주는 native 좌표 + 프레임/native 스케일).

## 4. 아키텍처

신규 `poc/workflow_2/golden_reregister_report_cond.py`. 상단에서 `golden_eval_config_loader.seed_env()` 호출
(다른 드라이버와 동일, import 전 env 브리지). 재사용 모듈:

- `poc.workflow_3.align.matching.engine.compute_align_key_score_ensemble(template, frame, *,
  frame_nm_per_pixel=None, roi_hint=None, scales=None, policy=DEFAULT_POLICY)` → `AlignKeyMatchResult`:
  `.score`, `.second_ratio`(<2 후보면 **`None`**), `.candidates[].xy/.score`, `.best_xy`, `.best_scale`,
  `.distinctive`. **`candidate.xy` 는 ROI-local(template-center) — 호출자가 origin 을 더해 풀프레임 환산**
  (리뷰 확인). `frame_nm_per_pixel`/`roi_hint` 는 기본 None 으로 생략 가능. ADVISORY self-match 전용.
- `align_similarity._gt_in_topk(gray, crosshair_xy, center_tpls, *, topk, scales, tol_short)`
  → `{topk_rank, in_topk, n_cand, best_cand_dist_norm, modality, cand_xys, peak_ratio, cand_scores,
  cand_ncc}` — **STRONG+MEDIUM tier free-search 1-pass**. `peak_ratio` = top2/top1 chamfer(= 2nd/1st
  모호도) 가 MEDIUM 신호다(리뷰 확인: `_gt_in_topk` 는 `second_ratio` 를 노출하지 않음 — `peak_ratio` 사용).
- `align_similarity._build_templates(assets)` → modality별 center/box 템플릿(+ `align_offset_xy`).
- `golden_localization_eval_cond` 의 프레임 로딩/clean/GT 환산 헬퍼(있으면 import, 없으면 thin wrapper).
- `align.assets.resolve_assets_auto()` — recipe 트리 리졸버.

순수 헬퍼(아래 §5~7 로직)는 드라이버 파일 안에 두되 **I/O 와 분리**해 합성 row/이미지로 단위 테스트 가능하게 한다.

## 5. Evidence 모델 (recipe × modality, 연속값)

한 recipe·modality 에 대해 세 신호를 산출한다. STRONG/MEDIUM 은 **rcp 템플릿 → 각 S 프레임 `_gt_in_topk`
1-pass** 에서 동시 추출(STRONG=위치/순위, MEDIUM=`peak_ratio`; 같은 dict). ADVISORY 는 rcp self-match 별도 pass.

**STRONG 전제(정의 지점에 명시 — 리뷰 반영):** 본 free-search(`_gt_in_topk`)는 **stage/cond prior 없는
무가드 매처**다. production 의 실제 정렬은 **가드된** 탐색(stage 위치·cond prior 보유)이라, S(성공) 프레임이
"free-search 실패"를 보이는 것은 모순이 아니다 — *무가드 매처라면 틀린 점을 골랐을* 것을, production 가드가
구제해 성공한 latent 위험이다. 따라서 `strong_fail_frac>0` 은 "**무가드 free-search** 가 ≥1 S 프레임에서 진짜
점을 못 고름"의 binary 사실이며, "production 가 실패했다"가 아니다(§11 라벨과 일관).

### STRONG — free-search localization (가드 의존 위험 직격)
각 S 프레임에 `_gt_in_topk`:
- `gt_rank`(=`topk_rank`; 채택 modality top-N 내 GT 순위; rank1 아니면 위험),
- `free_best_disp_norm`(=`best_cand_dist_norm`; free-best 가 GT 에서 얼마나 떨어졌나).
recipe 집계: `strong_fail_frac` = (`in_topk=False` 또는 `topk_rank>1` 인 프레임 비율) + worst-case
`free_best_disp_norm`. **클수록 위험.** "무가드 매처가 진짜 점을 *고르지 못함*."

### MEDIUM — msr ambiguity tail (`peak_ratio`)
같은 `_gt_in_topk` dict 의 `peak_ratio`(top2/top1 chamfer) → recipe 집계 **worst-case(max)** + `n_s_frames`.
median 금지(brittle 프레임 은폐). 높을수록 측정 외형에서 모호. 참조선 0.98(표기만, 경계 아님). `peak_ratio` 가
None/부재인 프레임(후보<2)은 **모호 아님(=0 risk)** 으로 보고 tail 산정에서 제외하지 않고 0 으로 반영.

### ADVISORY — rcp self-distinctiveness (OM 보조)
rcp 템플릿을 자기 *full* rcp 이미지에 `compute_align_key_score_ensemble` 매칭, **명시적 exclusion zone**
(origin self-peak 중심 template 풋프린트 1개 반경 = `EXCL_RADIUS_FOOTPRINTS×max(tw,th)`, module const 기본 1.0)
밖 잔여 후보 중 2nd peak / best 로 `self_ratio`. **`second_ratio`/잔여후보가 None(<2 후보)이면 look-alike 없음
= 최대 변별 → `self_sev=0`**(리뷰 반영).
- SEM: key 가 자기 이미지를 거의 채워 슬라이드 여지 부족 → near-degenerate. `advisory_confidence="low"`.
- OM: 정상 보조 신호.
단독 트리거 금지(STRONG/MEDIUM 없는데 ADVISORY 만으로 surface 하지 않음). 동률 tiebreak·근거 보강용.

## 6. 랭킹 (연속, evidence-tiered, modality 별)

flag 개수 합산이 아니라 tier 가중 risk score. **severity 는 신호 raw 값 그대로 쓴다 — cohort 정규화(min-max)
안 함**(리뷰 반영: min-max 는 항상 최저를 0/최고를 1 로 만들어 절대 floor 가 무의미해지고 1-recipe/동값에서
0/0). 모든 신호는 이미 [0,1] 유계다: `strong_fail_frac`(비율), `peak_ratio`(top2/top1), `self_sev`(2nd/best),
`free_best_disp_norm`(정규화 거리). floor 는 그 raw 값에 대한 **절대** 컷이다. 이 floor 들은 본 리포트의 자체
상수(live 보정 τ 0.98/0.6053 차용 아님); 오피스 분포 보고 보정 대상.

```
# 자체 절대 floor (module const; live τ 아님). 오피스에서 1회 보정.
MSR_FLOOR  = 0.85   # peak_ratio tail 이 이 미만이면 모호 아님(MEDIUM 미달).
SELF_FLOOR = 0.85   # self_ratio 가 이 미만이면 변별 충분(ADVISORY 미달).

evidence_tier(recipe,mod):
  STRONG   if strong_fail_frac > 0                        # 무가드 free-search 가 틀림 (binary 사실)
  MEDIUM   elif msr_peak_ratio_tail >= MSR_FLOOR          # 측정 외형 모호 (raw 절대 컷)
  ADVISORY elif (mod==OM and self_ratio >= SELF_FLOOR)    # rcp self 모호 — **OM 만**
  NONE     otherwise
  # SEM 의 self_ratio 는 near-degenerate(§1/§5)라 단독으로 tier 를 만들지 않는다
  # (MEDIUM 으로 승격시키지 않음 — 리뷰 반영: 그러면 폐기된 신호를 weight 1.0 으로 부풀림).

risk_score = tier_weight + severity      # severity ∈ [0,1) = tier 내 정렬 키 (raw)
  tier_weight: STRONG=2.0, MEDIUM=1.0, ADVISORY=0.3, NONE=0
    STRONG:   severity = strong_fail_frac      (동률 시 worst free_best_disp_norm)
    MEDIUM:   severity = msr_peak_ratio_tail
    ADVISORY: severity = self_ratio
```

- modality 별 독립 테이블, `risk_score` desc 정렬. cohort 통계 없음 → 1-recipe/동값에서도 안전(div 없음).
- live τ(0.98/0.6053)는 컬럼 옆 **참조선 마커**로만(예: `0.991*` 의 `*`= τ 초과), tier 경계 아님(경계는 위 자체 floor).
- STRONG 인데 self_ratio 낮으면(OM) "측정 외형/가드 의존" 해석 라벨.
- 상관 축 이중계수 회피: tier 는 *가장 강한* 증거 1개로 결정(if/elif), severity 만 해당 tier 신호로.
- SEM 은 STRONG 또는 MEDIUM 으로만 surface(self 단독 불가). OM 은 셋 다 가능.

## 7. 박스 제안 (flagged recipe 만, candidate-only)

baseline = 엔지니어 현재 whitebox(`cond.box_ltrb`). 그 self/msr 점수가 *beat 대상*.

**파생 단위 정의(리뷰 반영):** `fidelity` 는 **per-S-frame** 점수 — 박스(현재 또는 후보)를 그 프레임의 GT
crosshair 위치에 놓고 `compute_align_key_score_ensemble` 의 `.score`. `self_ratio` 는 **per-rcp-image scalar**
(프레임 없음). "paired" = **같은 S 프레임에서** 후보 박스 fidelity − 현재 박스 fidelity(프레임별 delta) →
**mean** 집계. self_ratio 는 프레임이 없으므로 paired 가 아니라 **scalar 비교**(후보 self_ratio < 현재 self_ratio).
두 비교는 분리한다.

1. **후보 생성:** rcp 이미지 위 modality별 window(엔지니어 박스 크기 × `SUGG_SCALES`=`(0.8,1.0,1.25)`, stride
   = `SUGG_STRIDE_RATIO`=0.25 × 박스 short side; module const) 슬라이드. texture 게이트(`_edge_density`/`_lap_var`
   최소치 미만 패치 skip — unique-but-blank 는 무용/불안정).
2. **held-out split (viable 최소 프레임 = `SPLIT_MIN_S`=4):** modality 의 S 프레임이 `SPLIT_MIN_S` 미만이면
   제안 **skip + "insufficient frames"**(리뷰 반영: OM 2/SEM 3 희박 recipe 다수가 여기 해당 — C2 가 다수 recipe 에서
   미동작함을 알려진 한계로 명시. Phase 2 에서 E 프레임 합류 시 완화). 충족 시 홀짝 결정적 분할(각 half ≥2).
   - *select-half* 로 후보별 self_ratio(낮을수록 unique) + per-frame fidelity 계산 → **mean fidelity ≥ 현재 박스
     mean fidelity**(paired delta ≥ 0) 인 후보 중 최저 self_ratio 선택.
   - *validate-half* 로 후보·현재 박스 재측정. 채택 조건(둘 다): **mean paired fidelity delta ≥ `accept_margin`**
     (기본 0.05) **그리고** 후보 self_ratio < 현재 self_ratio − `accept_margin`.
3. **inpaint-dodge 가드:** removal mask(§3, `build_removal_mask` 또는 GT crosshair 사각형)와 후보/현재 박스
   overlap 비율 기록. 후보가 overlap 급감(현재 대비)으로만 이득(validate delta 가 margin 부근)이면 reject
   또는 "inpaint-sensitive" 경고 — 깨끗한 픽셀로 이긴 가짜 이득 차단.
4. 채택 시: 제안 박스 좌표(rcp **native px** + 함의 `align_offset`), self_ratio·mean fidelity(select·validate 둘 다),
   `candidate_for_review=True`. 미채택: `"no distinctive sub-region"`(주기적 SEM 등 정직한 결론).

제안은 절대 "recommended replacement" 가 아니라 **"candidate for engineer review"**. office overlay 가 현재 vs
후보 박스를 그려 엔지니어가 육안 검증.

## 8. 출력

- **리포트 파일** `DEBUG_IMAGE_DIR/golden_reregister_report_cond/reregister_report.txt`
  (리뷰 반영: workflow_2 루트는 git-tracked — 다른 드라이버처럼 gitignored `debug_images/<driver>/` 에 쓴다.
  `from poc.workflow_2 import DEBUG_IMAGE_DIR`. relayable 텍스트 목적 불변 — digest.txt 도 거기서 회신):
  modality별 테이블 — `rank · recipe · tier · strong_fail_frac · free_disp · msr_peak_tail(τ*) ·
  self_ratio(conf) · n_s · suggestion(box|none|insufficient) · sugg_self/fidelity(sel/val)`.
- **`[DIGEST]` 1줄** (재타이핑 없이 결과 회신용; 다른 드라이버처럼 `digest.txt` 에도 기록):
  `[DIGEST] reregister(S-only): om[screened A, strong B, w_sugg C, top r1,r2] | sem[...]`.
- **office-only overlay** (flagged + 제안 있는 recipe): `DEBUG_IMAGE_DIR/golden_reregister_report_cond/`
  아래 현재 박스(자홍)+후보 박스(초록)+strong-fail look-alike 위치 표기. **JPEG 저장**(`.jpg`; 파이프라인에
  VLM 없음 → WebP 비대상). Mac 미반입.
- **print/DIGEST/배너 문자열은 ASCII 만**(리뷰 반영: cp949 콘솔, em-dash U+2014 금지 — 본 spec 산문엔 em-dash·→
  가 있으나 *출력 문자열엔* `-`/`:`/`->` 만). 코드의 string 리터럴은 이 규칙을 따른다.
- Mac dev(`no_data`): `[WARNING] no_data` + exit 0.

## 9. 컴포넌트 분리 & 시퀀싱

두 경계 컴포넌트. plan 은 1 을 먼저 완결·테스트 후 2 가 그 위에 올라가도록 순서화:

- **C1 — screening/랭킹 리포트** (§5 evidence + §6 랭킹 + §8 텍스트/DIGEST). 박스 제안 없이도 완전한 산출물.
- **C2 — 박스 제안 엔진** (§7). C1 의 flagged 목록을 소비. overlay 포함.

## 10. 설정 추가 (`golden_eval_config.example.py` + loader)

- `REREGISTER_BOX_SUGGEST` (기본 1) — C2 on/off. 0 이면 C1 리포트만.
- `REREGISTER_TOPN` (기본 0=전체) — DIGEST/overlay 상위 N 제한(0=무제한).
- (선택) `REREGISTER_ACCEPT_MARGIN` 미설정 시 module const 0.05.
- module const(설정 아님; 오피스 보정 대상): `MSR_FLOOR`/`SELF_FLOOR`(0.85), `REL_FLOOR` 제거됨,
  `EXCL_RADIUS_FOOTPRINTS`(1.0), `SUGG_SCALES`/`SUGG_STRIDE_RATIO`, `SPLIT_MIN_S`(4).
- `seed_env()` 가 `os.environ.setdefault` 로 브리지(OS env 우선). config 상수 `GOLDEN_ROOT` →
  **env 명은 `ALIGN_GOLDEN_ROOT`**(리뷰 확인; §13 실행 스니펫과 일치). rcp+msr 동일 트리라 별도 root 불필요.

## 11. 정직성 / survivorship 명시

리포트 헤더와 DIGEST 에 항상 출력하는 배너(ASCII 만 — em-dash 금지):
**`"S-only latent-risk screening: candidates among historically-successful recipes, NOT a confirmed
fail list. E-frame confirmation = Phase 2."`**. STRONG tier 도 "성공했으나 무가드 free-search 가
look-alike 를 선호(production 가드로 생존)" latent 위험임을 라벨로 구분.

## 12. 테스트

순수 헬퍼(합성, 골든 불요):
- `evidence_tier`/`risk_score`: 합성 신호 dict → STRONG>MEDIUM>ADVISORY 순서, ADVISORY(OM) 단독 비-surface,
  **SEM self 단독은 절대 MEDIUM/ADVISORY 로 안 뜸**(NONE), 절대 floor 경계(MSR/SELF_FLOOR), τ 는 표기만.
- 1-recipe·동값 cohort → div/예외 없이 동작(min-max 제거 회귀 가드).
- `peak_ratio` None 프레임(후보<2) → tail 에 0 으로 반영(모호 아님).
- self_ratio None(self-match 후보<2) → `self_sev=0`.
- 랭킹 정렬: 합성 row 리스트 → modality별 worst-first.
- held-out split: 홀짝 분할 결정성, **S 프레임 < `SPLIT_MIN_S` → "insufficient frames"**(각 half ≥2 보장).
- accept: validate mean paired fidelity delta ≥ margin **그리고** self_ratio 개선 ≥ margin 만 채택.
- inpaint-dodge: 후보가 overlap 급감으로만 이득이면 reject; overlap 비율 계산(합성 mask+box).

합성 이미지(엔진 실매칭):
- unique-patch-over-periodic → 제안 검색이 patch 발견.
- all-periodic → `"no distinctive sub-region"`.
- SEM-fill(템플릿이 이미지 거의 채움) → self-match `advisory_confidence="low"` + tier 비-surface.

Mac dev: `uv run python -c "import py_compile; ..."` 또는 `python -m py_compile`, `uv run pytest
poc/workflow_2/test_reregister_report.py -q`, `uv run python poc/workflow_2/golden_reregister_report_cond.py`
→ `[WARNING] no_data`. accuracy 는 오피스 `ALIGN_GOLDEN_ROOT` 에서만.

## 13. 오피스 실행

```text
# golden_eval_config.py: GOLDEN_ROOT=<align_images_golden>, REREGISTER_BOX_SUGGEST=1
uv run python poc/workflow_2/golden_reregister_report_cond.py
# → reregister_report.txt + [DIGEST] 1줄 회신 + overlay(office-only)
```

판정: STRONG tier recipe 가 재등록 1순위. 제안 박스가 validate-half 에서 baseline 을 paired 로 이기면 엔지니어
검토 후보. Phase 2(E-confirmation)로 latent→confirmed 승급은 별도 작업.
