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
- modality 배정: 기존 드라이버의 `_route_modality`/cond 규약 재사용(OM↔IMAP0001, SEM↔IMAP0002).
- msr S 프레임은 매칭 전 `align.clean_align_image` 로 crosshair inpaint(TELEA) — 기존 규약. inpaint mask 는
  박스 제안 가드용으로 보존.
- GT = S 프레임 cond.txt 의 crosshair px@5120 → 프레임 좌표 환산(기존 변환 재사용).

## 4. 아키텍처

신규 `poc/workflow_2/golden_reregister_report_cond.py`. 상단에서 `golden_eval_config_loader.seed_env()` 호출
(다른 드라이버와 동일, import 전 env 브리지). 재사용 모듈:

- `poc.workflow_3.align.matching.engine.compute_align_key_score_ensemble(tpl, gray, scales, policy)`
  → `.score`, `.second_ratio`, `.candidates[].xy/.score`, `.best_xy`, `.best_scale`, `.distinctive`.
- `align_similarity._gt_in_topk(gray, crosshair_xy, center_tpls, *, topk, scales, tol_short)`
  → `{topk_rank, in_topk, n_cand, best_cand_dist_norm, modality, cand_xys}` — **STRONG tier free-search**.
- `align_similarity._build_templates(assets)` → modality별 center/box 템플릿(+ `align_offset_xy`).
- `golden_localization_eval_cond` 의 프레임 로딩/clean/GT 환산 헬퍼(있으면 import, 없으면 thin wrapper).
- `align.assets.resolve_assets_auto()` — recipe 트리 리졸버.

순수 헬퍼(아래 §5~7 로직)는 드라이버 파일 안에 두되 **I/O 와 분리**해 합성 row/이미지로 단위 테스트 가능하게 한다.

## 5. Evidence 모델 (recipe × modality, 연속값)

한 recipe·modality 에 대해 세 신호를 산출한다. STRONG/MEDIUM 은 **rcp 템플릿 → 각 S 프레임 매칭 1-pass** 에서
동시 추출(효율). ADVISORY 는 rcp self-match 별도 pass.

### STRONG — free-search localization (production 실패모드 직격)
각 S 프레임에 `_gt_in_topk` 로 free-search:
- `gt_rank` (채택 modality top-N 내 GT 순위; rank1 아니면 위험),
- `free_best_disp_norm` (`best_cand_dist_norm`; free-best 가 GT 에서 얼마나 떨어졌나),
- `gt_top1_gap` (top-1 score − GT-cand score; 클수록 look-alike 를 더 선호).
recipe 집계: `strong_fail_frac` = (free-best ≠ GT 인 프레임 비율, in_topk=False 또는 rank>1 포함),
+ worst-case `free_best_disp_norm`. **클수록 위험.** "매처가 진짜 점을 *고르지 못함*."

### MEDIUM — msr ambiguity tail
같은 매칭에서 각 S 프레임 `second_ratio` → recipe 집계는 **worst-case(max)** + `n_s_frames`.
median 금지(brittle 프레임 은폐). 높을수록 측정 외형에서 모호. 참조선 0.98.

### ADVISORY — rcp self-distinctiveness (OM 보조)
rcp 템플릿을 자기 *full* rcp 이미지에 매칭, **명시적 exclusion zone**(origin self-peak 중심
template 풋프린트 1개 이상 반경 제외) 후 잔여 후보 중 2nd peak / best 로 self-second-ratio.
- SEM: key 가 자기 이미지를 거의 채워 슬라이드 여지 부족 → `advisory_confidence="low"` 로 표기, 랭킹 기여 down-weight.
- OM: 정상 보조 신호.
단독 트리거 금지(STRONG/MEDIUM 없는데 ADVISORY 만으로 surface 하지 않음). 동률 tiebreak·근거 보강용.

## 6. 랭킹 (연속, evidence-tiered, modality 별)

flag 개수 합산이 아니라 tier 가중 risk score:

tier 경계는 **modality 내 상대 분포**로 정한다(live τ 차용 금지; τ 는 표기만). STRONG 만 절대 의미를
가진다 — `strong_fail_frac>0` 은 "free-search 가 ≥1 S 프레임에서 실제로 틀린 점을 골랐다"는 시연된 사실이라
분위가 아니라 binary.

```
REL_FLOOR = 0.50   # modality 내 정규화 severity 가 이 미만이면 NONE(잡음 바닥). module const.

evidence_tier(recipe,mod):
  STRONG   if strong_fail_frac > 0
  else: 비-STRONG 신호를 modality 내 min-max 정규화 →
        msr_sev  = norm(msr_second_ratio_tail)
        self_sev = norm(self_second_ratio)  (SEM-low 면 ×0.5)
        lead     = max(msr_sev, self_sev)
        NONE     if lead < REL_FLOOR
        MEDIUM   elif msr_sev >= self_sev      (msr 가 주도)
        ADVISORY else                          (self 가 주도; OM 만, SEM-low 는 MEDIUM 으로만)

risk_score = tier_weight + normalized_severity
  tier_weight: STRONG=2.0, MEDIUM=1.0, ADVISORY=0.3, NONE=0
  normalized_severity ∈ [0,1): tier 내 정렬 키
    STRONG: strong_fail_frac (동률 시 worst free_best_disp_norm)
    MEDIUM: msr_sev
    ADVISORY: self_sev
```

- modality 별 독립 테이블, `risk_score` desc 정렬. 정규화·REL_FLOOR 모두 modality 내에서만(OM/SEM 분포 분리).
- live τ(0.98/0.6053)는 컬럼 옆 **참조선 마커**로만(예: `0.991*` 의 `*`= τ 초과), tier 경계 아님.
- STRONG 인데 self_sev 낮으면(OM) "측정 외형에서만 모호" 해석 라벨.
- 상관 축 이중계수 회피: tier 는 *가장 강한* 증거 1개로 결정, severity 만 해당 tier 신호로.

## 7. 박스 제안 (flagged recipe 만, candidate-only)

baseline = 엔지니어 현재 whitebox(`cond.box_ltrb`). 그 self/msr 점수가 *beat 대상*.

1. **후보 생성:** rcp 이미지 위 modality별 window(엔지니어 박스 크기 ± 소수 scale) 슬라이드. texture 게이트
   (`_edge_density`/`_lap_var` 최소치 미만 패치 skip — unique-but-blank 는 무용/불안정).
2. **held-out split:** recipe 의 S 프레임을 select-half / validate-half 로 나눔(홀짝 등 결정적 분할; 1장뿐이면
   제안 skip + "insufficient frames" 표기).
   - *select-half* 로 후보별 (a) rcp self-second-ratio(낮을수록 unique) + (b) msr fidelity(후보를 select-half S
     프레임 대응 위치에 매칭한 score; 높을수록 안정) 계산 → **fidelity ≥ 현재 박스 fidelity(paired, baseline 만큼은
     안정)** 인 후보 중 최저 self-second-ratio 선택. (절대 τ 가 아니라 baseline 대비 상대 게이트.)
   - *validate-half* 로 그 후보 재측정. select 와 동일 부호로 baseline 을 **paired 하게** 이겨야 채택
     (`accept_margin` 이상; module const, 기본 0.05).
3. **inpaint-dodge 가드:** 후보·현재 박스의 inpaint mask overlap 기록. 후보가 overlap 회피로만 이득이면(현재
   대비 overlap 급감 + validate 이득이 margin 부근) reject 또는 "inpaint-sensitive" 경고.
4. 채택 시: 제안 박스 좌표(rcp px@native + 함의 `align_offset`), self/fidelity (select·validate 둘 다),
   `candidate_for_review=True`. 미채택: `"no distinctive sub-region"`(주기적 SEM 등 정직한 결론).

제안은 절대 "recommended replacement" 가 아니라 **"candidate for engineer review"**. office overlay 가 현재 vs
후보 박스를 그려 엔지니어가 육안 검증.

## 8. 출력

- **리포트 파일** `poc/workflow_2/reregister_report.txt` (relayable, 오피스→Mac 텍스트):
  modality별 테이블 — `rank · recipe · tier · strong_fail_frac · free_disp · msr_sr_tail(τ*) ·
  self_sr(conf) · n_s · suggestion(box|none|insufficient) · sugg_self/fidelity(sel/val)`.
- **`[DIGEST]` 1줄** (재타이핑 없이 결과 회신용, 다른 드라이버 규약과 동일):
  `[DIGEST] reregister(S-only): om[n_screened A, strong B, w/sugg C, top: r1,r2] | sem[...]`.
- **office-only overlay** (flagged + 제안 있는 recipe): `DEBUG_IMAGE_DIR/golden_reregister_report_cond/`
  아래 현재 박스(자홍)+후보 박스(초록)+strong-fail look-alike 위치 표기. Mac 미반입.
- Mac dev(`no_data`): `[WARNING] no_data` + exit 0.

## 9. 컴포넌트 분리 & 시퀀싱

두 경계 컴포넌트. plan 은 1 을 먼저 완결·테스트 후 2 가 그 위에 올라가도록 순서화:

- **C1 — screening/랭킹 리포트** (§5 evidence + §6 랭킹 + §8 텍스트/DIGEST). 박스 제안 없이도 완전한 산출물.
- **C2 — 박스 제안 엔진** (§7). C1 의 flagged 목록을 소비. overlay 포함.

## 10. 설정 추가 (`golden_eval_config.example.py` + loader)

- `REREGISTER_BOX_SUGGEST` (기본 1) — C2 on/off. 0 이면 C1 리포트만.
- `REREGISTER_TOPN` (기본 0=전체) — DIGEST/overlay 상위 N 제한(0=무제한).
- (선택) `REREGISTER_ACCEPT_MARGIN` 미설정 시 module const 0.05.
- `seed_env()` 가 `os.environ.setdefault` 로 브리지. 기존 `GOLDEN_ROOT` 재사용(별도 root 불필요; rcp+msr 동일 트리).

## 11. 정직성 / survivorship 명시

리포트 헤더와 DIGEST 에 **"S-only latent-risk screening — 성공 이력 recipe 중 위험 후보이며, 확정 실패
목록이 아님. E-frame confirmation 은 Phase 2."** 를 항상 출력. STRONG tier 도 "성공했으나 free-search 가
look-alike 를 선호한(가드로 생존)" latent 위험임을 라벨로 구분.

## 12. 테스트

순수 헬퍼(합성, 골든 불요):
- `evidence_tier`/`risk_score`: 합성 신호 dict → STRONG>MEDIUM>ADVISORY 순서, ADVISORY 단독 비-surface,
  SEM-low down-weight, τ 참조선 마킹(hard cut 아님) 검증.
- 랭킹 정렬: 합성 row 리스트 → modality별 worst-first.
- held-out split: 홀짝 분할 결정성, 1-frame → skip.
- accept margin: paired 이득 ≥ margin 만 채택; inpaint-overlap 회피 이득 reject.
- inpaint-overlap 계산: 합성 mask+box → overlap 비율.

합성 이미지(엔진 실매칭):
- unique-patch-over-periodic → 제안 검색이 patch 발견.
- all-periodic → `"no distinctive sub-region"`.
- SEM-fill(템플릿이 이미지 거의 채움) → ADVISORY `confidence="low"`.

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
