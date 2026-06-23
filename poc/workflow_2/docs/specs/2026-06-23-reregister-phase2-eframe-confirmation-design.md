# 재등록 리포트 Phase 2 — E-frame confirmation (score collapse S→E)

상태: 설계 확정(브레인스토밍 승인 2026-06-23). 구현 대기.
선행: Phase 1 spec `2026-06-23-reregister-report-design.md` (구현 완료, box-fidelity 까지 land).
대상: `poc/workflow_2/golden_reregister_report_cond.py` (offline CV 벤치, 동일 드라이버에 post-pass 추가).

## 1. 배경 / 목적

Phase 1 은 **S-only latent-risk screening** 이다 — 성공(S) 프레임 + rcp 만 보고 "이 align key 는
무가드 free-search 가 잘못 짚는다(in_topk=False 등)" 를 근거로 재등록 후보를 랭킹한다. survivorship
한계를 정직하게 명시한다(성공 recipe 만 봄, 실제 fail 여부는 미확인).

Phase 2 는 **실제 fail(E) 프레임으로 그 latent 위험을 confirm** 한다. E 프레임에서 align key 의 free-search
best-match **점수가 S 대비 붕괴(collapse)** 하면, 그 recipe 의 align key 는 fail 조건에서 robust 하게
findable 하지 않다는 직접 증거 → `E_CONFIRMED`(최상위 tier)로 승급.

핵심 제약(도메인 확정 사실): **E(fail) 프레임은 crosshair 가 없다**(오피스 실데이터 with_crosshair=0/182,
[[project-e-images-no-crosshair]]). 따라서 E 에는 GT align point 가 없어, Phase 1 의 `_gt_in_topk`
("정답이 후보 top-k 안에 드는가") 식 GT 기반 채점을 E 에 쓸 수 없다. Phase 2 는 GT 없이 성립하는
**score collapse** 신호만 쓴다.

## 2. Scope

**Phase 2 (본 spec, IN):**
- flagged recipe(Phase 1 tier != NONE)에 한해 **upgrade-only post-pass**.
- E 프레임 free-search best-score 와 S best-score 를 비교, collapse 시 `E_CONFIRMED` 승급.
- tier enum 에 `E_CONFIRMED`(최상위) 추가, 랭킹·DIGEST·리포트 반영.

**OUT (non-goals):**
- **confirm-AND-deny / independent E axis** 기각(브레인스토밍): collapse 안 한 flagged recipe 를 down-rank
  하거나, Phase 1 미flag recipe 를 E 단독으로 surface 하지 않는다. 순수 upgrade-only.
- E 프레임 overlay 이미지 저장(초기 미포함; DIGEST `confirmed` 카운트 + 리포트 s_rep→e_rep 로 충분).
- production 루프 통합, VLM, `align/assets.py`·align_images 트리 수정.
- Phase 1 로직(`_evidence_tier`, tier 산출, 박스 제안) 변경 — post-pass 는 tier 를 **override** 만 한다.

## 3. 데이터 입력 (E 프레임 contract)

Phase 1 과 동일 트리(`<GOLDEN_ROOT>/<eqp>/<class>/<recipe>/align_img_from_msr/`). S 와 E 는 같은
디렉토리에 섞여 있고 `_tool_label(name)` 으로 분류(S=success, E=fail).

E 프레임 로딩 규약(`_load_s_frames` 의 E 버전):
- `label == "E"` 만 채택(S 는 Phase 1 이 이미 소비).
- **crosshair/GT 없음** → cond.txt crosshair 불요, inpaint 불요. **raw gray** 그대로 사용
  (`load_gray(msr_path)`).
- modality 배정은 S 와 동일하게 `msr_modality(cond)` + `_route_modality_for_mod(cond, {modality},
  modality)` 로 한다(E 프레임도 cond.txt 의 box/modality 정보는 보유, crosshair 만 없음). 추론 불가/dual
  모호 프레임은 skip(Phase 1 race-금지 규약 동일).
- cond 부재 등으로 modality 미상이면 그 프레임 skip.

## 4. 핵심 신호: score collapse S→E

**score** = free-search best-candidate match score(proposer 점수, chamfer/ensemble). rcp **center 템플릿**
(`center_tpls[modality]`, `_build_templates` 산출)을 프레임에 매칭해 얻는 후보 중 최고 점수.

- **S best-score**: Phase 1 이 S 프레임마다 이미 `_gt_in_topk` 를 돌려 `cand_scores`(내림차순 proposer
  점수)를 반환·`frame_results` 에 보관한다. S 프레임의 best = `cand_scores[0]`. **신규 매칭 불요** — 기존
  값 재사용.
- **E best-score**: E 프레임마다 동일 proposer 를 GT 없이 돌려 최고 점수만 취한다.

> apples-to-apples 보장: E best-score 는 `_gt_in_topk` 가 S 에 쓰는 **동일 proposer 호출**과 bit-parity
> 여야 한다 — `USE_ENSEMBLE_PROPOSER` 분기 + `_propose_topk(center_tpl, gray, frame_dt,
> scales=COMPARE_SCALES, topk=TOPK_CANDIDATES)`. 둘이 갈라지면 S/E 점수 비교가 무의미해지므로, 향후
> proposer 설정 변경 시 양쪽을 함께 바꿔야 한다(코드 주석으로 `_gt_in_topk` 와 상호 참조).

> **scale band 주의**: E free-search 는 box-fidelity 가 아니라 center-crop localization 이므로
> `_FIDELITY_SCALES`(tight band) 가 아니라 **`COMPARE_SCALES`** 를 쓴다 — `_gt_in_topk` 와 동일.

## 5. 컴포넌트 (신규 함수, 모두 동일 드라이버)

- `_load_e_frames(assets, modality) -> list[np.ndarray]`
  E(label="E") 프레임을 modality 필터해 **raw gray** 리스트로. crosshair/inpaint 없음. §3 규약.
- `_free_search_best_score(center_tpl, gray) -> float | None`
  center 템플릿을 프레임에 free-search, **최고 proposer 점수** 반환. **예외 또는 후보 리스트 자체가 빔**
  (degenerate frame, artifact)이면 `None`(채점 불가, §9 skip 대상). 후보가 있으면 점수가 **낮아도 float**
  반환(낮은 점수 = collapse 증거이므로 절대 None/0 으로 버리지 않는다). `_gt_in_topk` proposer 호출과
  bit-parity(§4 주의).
- `_s_rep_score(frame_results) -> float | None`
  S `frame_results` 리스트에서 프레임별 `cand_scores[0]` 의 **median**. 비었거나 점수 없으면 None.
- `_e_rep_score(center_tpl, e_frames) -> float | None`
  E 프레임별 `_free_search_best_score` 결과 중 **None 을 제외**한 점수들의 **median**. 사용 가능한 E 점수가
  하나도 없으면 None.
- `_e_confirm(s_rep, e_rep) -> bool`
  §6 규칙(순수 함수).
- `run()` 내 **post-pass**: 랭킹 직전, flagged row 마다 E 로딩→e_rep→`_e_confirm`→승급.

`_recipe_row` 는 post-pass 가 s_rep 을 산출할 수 있도록 row 에 `_frame_results`(S `_gt_in_topk` 결과
리스트)를 보관한다(이미 계산된 값 저장만; Phase 1 산출 로직 불변).

## 6. Collapse rule + aggregation + thresholds

aggregation = **median**(소표본 n≈2~3 robust). s_rep, e_rep 모두 프레임 median.

```
def _e_confirm(s_rep, e_rep):
    if s_rep is None or e_rep is None:
        return False                      # E 없음/사용불가 -> confirm 불가(latent 유지)
    if s_rep < S_FLOOR:
        return False                      # high-S premise: 애초에 findable 했어야 'collapse'
    return (s_rep - e_rep >= COLLAPSE_MARGIN) or (e_rep <= E_FLOOR)
```

기본 임계(모두 office-calibratable, env/config 노출):
- `S_FLOOR = 0.60` — S 에서 key 가 적어도 어느 정도 findable 했음(collapse 의 전제).
- `COLLAPSE_MARGIN = 0.15` — E 가 S 대비 이만큼 떨어지면 collapse.
- `E_FLOOR = 0.50` — S-E delta 가 작아도 E 절대값이 이 밑이면 collapse.

> 점수는 matcher 특성상 ~0.6 부근으로 압축되어 있다([[project-matcher-flat-chamfer-distinctiveness]]).
> 위 기본값은 출발점이며 office 에서 S/E 점수 분포를 보고 보정한다(Phase 1 floor 들과 동일한 캘리브 방식).

## 7. Tier model & 랭킹

- tier enum 에 `"E_CONFIRMED"` 추가, **최상위**. `TIER_WEIGHT["E_CONFIRMED"] = 3.0`
  (기존 STRONG 2.0 / MEDIUM 1.0 / ADVISORY 0.3 / NONE 0.0 위).
- `_evidence_tier` 는 **S-only 그대로**(Phase 1 순수 유지). post-pass 가 confirm 시 row 의 tier 를
  `E_CONFIRMED` 로 **override** 하고, `risk_score` 를 top-band 로 재산정(`_risk_score` 가 새 weight 반영).
- 랭킹은 기존 `_rank_rows`(risk_score desc, worst_disp tiebreak) 그대로 — E_CONFIRMED 가 weight 3.0 으로
  자연히 최상단.

## 8. Output (DIGEST / report / row fields)

- row 신규 필드: `e_confirmed`(bool), `s_rep`(float|None), `e_rep`(float|None), `n_e`(int).
- **DIGEST**: modality 별 `confirmed N` 추가. 예:
  `om[screened 64, strong 33, confirmed 4, w_sugg 1] | sem[...]`.
- **리포트(txt)**: E_CONFIRMED 행을 최상단에, `s_rep -> e_rep (n_e=K)` 컬럼 노출.
- **banner**: confirmation 이 돌면 순수 "S-only latent" 문구를 완화 — confirmed 행은 E 증거 보유,
  나머지는 여전히 latent 임을 명시.
- ASCII-only(cp949), `print(...).encode("ascii","replace")` 기존 규약 유지.

## 9. Error handling

- **E 프레임 0개 / 사용 가능한 E 점수 0개**(전부 modality-mismatch·후보없음·예외) → `e_rep=None`
  → `_e_confirm` False → **latent 유지**(confirm 아님). 정상 경로.
- **E 한 프레임에서 엔진 예외 또는 후보 리스트 자체가 빔(artifact)** → `_free_search_best_score` None →
  그 프레임 **skip**(점수 목록에서 제외). **0.0 으로 적지 않는다** — 인프라/degenerate 실패를
  'collapse(낮은 점수)' 로 오독하면 거짓 confirm 이 된다. 사용 가능한 E 점수가 남으면 그들만으로 median.
  단, 후보가 **있으면서 점수가 낮은 것**은 artifact 가 아니라 **collapse 증거 자체**이므로 반드시 포함한다
  (matcher 는 나쁜 프레임에서도 보통 낮은-점수 후보를 반환하므로, 진짜 unfindable key 는 E_FLOOR 로 잡힌다).
- **non-flagged row(tier==NONE)** → post-pass 진입 자체 skip(upgrade-only).
- **REREGISTER_E_CONFIRM=0** → post-pass 전체 skip(Phase 1 동작과 동일, E_CONFIRMED 미발생).

## 10. Config knobs (env + golden_eval_config bridge)

Phase 1 knob 들과 동일 패턴: `golden_eval_config.example.py` 상수 + `golden_eval_config_loader.py`
`setdefault`(실 env 우선), `run()` 시작 `[INFO]` 라인에 활성값 자가-라벨.

- `REREGISTER_E_CONFIRM` (1=on 기본, 0=Phase 1 만)
- `REREGISTER_S_FLOOR` (0.60)
- `REREGISTER_E_FLOOR` (0.50)
- `REREGISTER_COLLAPSE_MARGIN` (0.15)

## 11. Testing & Mac gate

- `_e_confirm` — **순수, 전 분기 TDD**: margin 으로 confirm / E_FLOOR 로 confirm / low-S deny /
  e_rep=None deny / s_rep=None deny.
- `_s_rep_score`·`_e_rep_score` — 순수 aggregation: median 정확, 빈 입력 None, None/0 혼재 처리.
- `_free_search_best_score` — **engine-backed synthetic**(기존 fidelity 합성 패턴 재사용): 고유 마크
  프레임 → 점수 > 저텍스처/blank 프레임. (S vs E 점수 대소를 합성으로 1건 확인.)
- `_load_e_frames` — 순수 로직 일부만 Mac 테스트 가능(label/modality 필터). 실프레임 로딩은 office 픽스처.
- **통합 + 실임계 정확도는 office-gated**(DIGEST `confirmed` 카운트). Mac gate = py_compile + suite +
  위 합성/순수 테스트.

## 12. 캘리브레이션 / 후속

- 첫 office run: `[INFO]` 임계 라벨 + DIGEST `confirmed` 카운트 + 샘플 E_CONFIRMED 행(s_rep→e_rep)
  relay → S/E 점수 분포 보고 `S_FLOOR/E_FLOOR/COLLAPSE_MARGIN` 보정.
- E_CONFIRMED 과다/과소 시 임계만 조정(코드 불변, env A/B). `REREGISTER_MAX_RECIPES` fast-mode 로 빠른 A/B.
- (별개 축, 본 spec 아님) SEM center-crop proposer-recall 한계 — template-bank / VLM-ROI.
