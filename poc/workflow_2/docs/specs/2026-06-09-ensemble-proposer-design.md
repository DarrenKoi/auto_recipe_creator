# Ensemble Proposer 설계 (single rcp box 경로의 proposer recall 끌어올리기)

- **날짜**: 2026-06-09
- **상태**: 설계 승인됨 (구현 계획 대기)
- **관련**: [[project_matcher_flat_chamfer_distinctiveness]], `golden_localization_eval_cond.py`, `align_key_matcher.py`
- **자문**: codex:rescue (설계 옵션 + 파라미터 2회 자문)

## 1. 문제 (측정됨, modality-routed clean)

`golden_localization_eval_cond` 의 `box__inpaint` 셀 (단일 rcp box template, msr modality 라우팅 적용 후):

```
gt_in_topk = 0.557    진실(align point)이 chamfer 후보 top-8 에 드는 비율
rank1      = 0.422    1발 명중
topk!=1    = 0.138    후보엔 있으나 1등 아님 (rerank 회복 가능 갭)
1-gt_in_topk = 0.443  진실이 후보에 아예 없음 → proposer_wall
```

`lever_verdict = proposer_wall`. rerank 천장은 +13.8pp 뿐 → **proposer(후보 생성) 자체를 강화**해야 한다.

**근본 원인**: 단일 edge 채널(Canny-DT) chamfer 점수면이 평평해 진짜 위치가 또렷한 peak를 못 만들고 top-N에 못 든다.

## 2. 목표 / 비목표

**목표**: 다중 구조 채널 ensemble proposer로 `box__inpaint` 경로의 `gt_in_topk`(proposer recall)를 올린다. 이 경로 = **consensus 없을 때의 fallback 천장**이라 생산상 중요(consensus 적용 가능 recipe가 sparse).

**비목표**:
- reranker 개선 (천장 +13.8pp, 별도 이슈).
- consensus 재등록 (직교 레버, 별도 진행).
- VLM 좌표 결정 (도메인 규칙상 금지 — CV가 좌표 authority).
- pixel 동일성 지표(NCC/SSIM) (공정 드리프트로 금지).

## 3. 채널 설계 (3가지 독립 구조 관측)

세 채널 모두 **기존 multi-scale chamfer 엔진(`_chamfer_score_map_at_scale`)을 edge map만 바꿔 재사용** → 새 매칭 엔진 없음(표류/버그 표면 최소).

### C1 — Canny-DT chamfer (기존, baseline 채널)
`preprocess_for_matching`: grayscale→CLAHE(clip2,tile8)→GaussianBlur(σ1)→Canny(60,160)→distanceTransform(L2). 변경 없음.

### C2 — Scharr gradient-magnitude chamfer
- Scharr `|∇|` → **edge-density matched percentile** 이진화. threshold = `(1 - r_c1)` 백분위, 여기서 `r_c1` = C1 Canny foreground ratio, **3~15% clamp**.
- 근거: Otsu는 SEM/OM contrast drift에 흔들림. C1도 binary edge mask 기반이라 **foreground 밀도를 C1에 맞춰야** 채널별 mean_dt 스케일·peak sharpness가 동등 비교됨.
- 잡는 것: Canny hysteresis가 버리는 **약한/저대비 구조**.

### C3 — orientation-binned directional chamfer
- **8 bins**, **0–180° half-angle**(unsigned gradient — polarity/밝기 반전에 강건, full 360°은 같은 물리 edge를 두 bin으로 쪼개 peak 분산).
- bin별 DT 구성 → **same-bin template edge만 매칭** → **edge-count weighted mean distance**로 합산 → `exp(-mean_dt/DT_TAU_PX)`.
  - mean(min 아님): min은 한 방향만 맞아도 과대평가. 단순 sum은 edge 많은 bin이 지배. weighted mean이 기존 chamfer 평균거리 규약과 일관.
- 목적: **방향 일치까지 요구**해 wrong-local-peak(과거 67%)를 깎아 점수면을 뾰족하게 → proposer_wall 직격.

## 4. 융합 = Reciprocal Rank Fusion (RRF)

- 각 채널: multi-scale chamfer → per-channel 후보 (center xy, scale, score). per-scale 내부 **8~12 peaks**, 채널당 **solo top-24**.
- 융합: `fused(c) = Σ_채널 1/(k0 + rank_채널(c))`. 채널 간 후보는 **공간 근접 매칭 반경 = max(8px, 0.05·template_short_side)** 로 동일 후보 병합.
- **k0 = 10** (sweep 10/20/60). 근거: 리스트가 top-8~24 규모면 classic 60은 rank 차이를 과압축해 union voting처럼 변함 → 10이 상위 rank 신호 보존.
- **RRF를 score-sum 대신**: 채널별 점수 스케일이 달라(calibration 위험) **순위 기반이라 스케일 무관** → 보정 실패 리스크 제거.
- post-fusion **global NMS 반경 = 0.25·template_short_side** (기존 per-scale 0.5보다 작게 — fusion 후엔 근접 대안까지 지우면 recall@24 진단이 둔해짐).
- 출력: **top-8 (KPI) + shadow-24 (진단)**.

## 5. 측정 — 정직한 A/B (proposer recall만 격리)

신규 `proposer_recall_ab.py` — `golden_localization_eval_cond` 의 frame 순회 + **modality 라우팅(`_route_modality`)** + box template(`_build_offset_templates_cond`) + cond-GT를 **import 재사용**(drift 방지).

각 routed `box__inpaint` S프레임:
- **GT** = cond crosshair. 후보 xy + `align_offset`이 **GT_TOL_NORM(현행 0.20 short-side, 변경 금지)** 내면 hit. (tolerance 바꾸면 proposer 개선과 평가 완화가 섞임.)
- **recall@{8,16,24}** 산출. @N cutoff 정렬 기준 = **RRF fused score** (생산 후보 순서와 동일 계약).
- 비교: **baseline(C1-only) vs ensemble(C1+C2+C3)**.
- **per-channel attribution**: GT가 해당 채널 **solo top-24** 안에 있으면 credit (fusion 실패 vs channel proposer 실패 분리).
- reranker(ORB) 재정렬 **금지** — 순수 후보 membership만.

**출력 표 + verdict**:
- `recall@8 / @16 / @24` (baseline vs ensemble + 채널별 solo + lift).
- 진단: `recall@24`↑면 proposer 다양성 부족(채널 추가 효과 있음), `@24`도 낮으면 채널 자체가 틀림.
- KPI = **budgeted recall@8** 상승 여부.

## 6. 파라미터 표 (codex 권장값)

| 항목 | 값 | 근거 |
|---|---|---|
| C2 threshold | edge-density matched percentile, r_c1 3~15% clamp | C1 foreground 밀도 매칭 → 동등 비교 |
| C3 bins | 8, 0–180° half-angle | 변별력 vs peak 분산 균형, polarity 강건 |
| C3 합산 | bin별 edge-count weighted mean dist → exp(-mean_dt/10) | 기존 chamfer 평균거리 규약 일관 |
| RRF k0 | 10 (sweep 10/20/60) | 작은 리스트서 상위 rank 신호 보존 |
| 채널 매칭 반경 | max(8px, 0.05·tpl_short) | scale별 size 변동 흡수 |
| post-fusion NMS | 0.25·tpl_short | recall@24 진단 보존 |
| per-channel solo k | 24 (per-scale 8~12 peaks) | @8/@24 + attribution 안정 |
| shadow N | 24 (3× of 8) | 천장 vs budget 손실 분리, rerank 비용 불변 |
| GT_TOL_NORM | 0.20 (현행 유지) | A/B 신호 오염 방지 |
| DT_TAU_PX | 10 (현행) | 채널 간 동일 |

## 7. 실패 모드 (사전 경고 + 완화)

- **FM1 directional bin jitter**: orientation 양자화 경계 근처에서 GT 근처 후보가 여러 bin/scale로 쪼개져 RRF rank가 낮아질 수 있음. → shadow 진단 시 **±1 bin tolerance hit** 여부도 기록.
- **FM2 density-matched clutter**: Scharr가 공정 texture/noise까지 edge로 살려 clutter-rich wrong peak를 채널 합의처럼 보이게 할 수 있음. → **채널별 solo GT coverage와 fused false-consensus 사례를 분리 기록**.

## 8. 컴포넌트 / 파일

- `align_key_matcher.py` (또는 신규 `ensemble_proposer.py`):
  - `_scharr_edges(gray, r_c1)` → C2 edge map.
  - `_orientation_bins(gray, n_bins=8)` → C3 per-bin edge maps + directional chamfer score map.
  - `compute_ensemble_candidates(template, frame, *, channels, top_n=8, shadow_n=24, k0=10)` → fused 후보 + per-channel solo 후보(attribution).
  - 기존 `compute_chamfer_candidates`(C1)는 그대로 — ensemble이 이를 한 채널로 호출.
- `proposer_recall_ab.py` (신규): recall@N A/B 러너 (localization eval 재사용).
- 테스트: `test_ensemble_proposer.py` (합성 단위), `test_proposer_recall_ab.py` (recall@N 산출 순수 헬퍼).

## 9. 빌드 순서 (tracer-bullet)

1. C2 Scharr 채널 + 단위테스트(알려진 위치 peak).
2. C3 directional chamfer + 단위테스트(방향 일치 시 peak 뾰족).
3. RRF 융합 + 단위테스트(스케일 무관 순위 병합, NMS).
4. `proposer_recall_ab.py` recall@N + attribution + 단위테스트.
5. 오피스 A/B 실행 → baseline vs ensemble recall@8/@24 표 → verdict.

## 10. 검증 (Mac/오피스)

- **Mac**: 합성 데이터로 각 채널·융합·recall@N 산출 단위테스트.
- **오피스**: 실 golden set으로 baseline vs ensemble proposer-recall A/B. KPI = budgeted recall@8 상승.

## 11. 미해결 / 리스크

- C3 directional chamfer 비용(8 bin × scale × frame) — 성능 확인 필요(per-bin DT 캐시).
- k0/NMS 반경은 cold-start 추정 → 오피스 sweep으로 보정.
- ensemble이 recall@8을 못 올리고 recall@24만 올리면(다양성은 있으나 fusion이 못 끌어올림) → fusion 재설계 or shadow budget 확대 논의.
