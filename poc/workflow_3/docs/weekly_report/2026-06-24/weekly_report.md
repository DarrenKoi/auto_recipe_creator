# Weekly Report — workflow_3 / 재등록 리포트 Phase 1·2

**기간:** 2026.06.19 ~ 06.24
**대상:** `poc/workflow_2/golden_reregister_report_cond.py` (오프라인 CV 벤치, align-key 재등록 우선순위 리포트)
**커밋:** 약 25개, 전량 `main` 직접 반영
**작업 방식:** Mac blind 작성 → 오피스 `git pull` 실행 → digest/경고 라인 relay 반복 (fab 데이터 Mac 반입 불가)

---

## 요약

만성 모호 align-key를 **데이터로 식별해 재등록 우선순위를 매기는 리포트**를 두 단계로 개발했다.

1. **Phase 1 — S-only 스크리닝 (완료)**: success 프레임만으로 변별력 약한 align-key를 latent-risk로 랭킹 + 대체 박스 제안.
2. **Phase 2 — E-frame confirmation (실험 종료/보류)**: 같은 free-search를 fail 프레임에 돌려 S→E 점수 collapse로 latent → confirmed 승급 시도. **오피스 실측에서 신호가 신뢰 불가로 판명 → 종료.**
3. **다음 단계 — matcher 개선 (착수)**: Phase 2가 드러낸 근본 병목(rcp align-key가 success에서도 변별력 부족)을 정면으로 다루는 template-bank 매칭 벤치 spec 작성.

> 핵심 교훈: **재등록 리포트가 "약하다"고 flag하는 recipe는 애초에 success에서도 약한 key**라, "S에서 높다가 E에서 무너지는" collapse 신호 자체가 성립하지 않았다. Phase 2는 깨끗이 종료하고 진짜 레버(matcher)로 이동.

---

## 1. Phase 1 — S-only 재등록 스크리닝 · **완료**

success(S) 프레임만으로 align-key 변별력을 평가해 재등록 우선순위를 랭킹하는 드라이버를 완성했다.

- **3단 evidence tier + risk score + 랭킹** (`71b9fd0`, `d038fec`, `dd1e57d`) — recipe별 증거를 집계해 latent-risk 순으로 정렬.
- **C1 스크리닝 리포트 드라이버** (`84cb662`) — STRONG = GT-absent + frac floor 기준, S-only 정직한 생존 편향 스크리닝.
- **C2 대체 박스 제안 + 오버레이** (`24f86f8`, `e9b51b6`, `ca81da4`) — 더 변별력 있는 영역을 박스로 제안하고 마킹 이미지로 시각화.

### box-fidelity 버그 추적 (06-23 집중 디버그)

box-suggestion 경로에서 fidelity가 **전 recipe에서 0**으로 떨어져 제안이 0건이던 문제를 끝까지 추적해 해결.

- **근본 원인**: 매칭 엔진은 후보 `xy`를 **patch 중심**으로 반환하고 `align_offset`을 적용하지 않는다. off-center 박스(특히 OM unique-area)는 후보가 crosshair 근처에 절대 안 떨어져 fidelity가 전부 0 (`f7501fb` — `_box_offset_xy` 신설, `expected = gt_xy + offset*scale`).
- **단계적 자가-진단**: all-zero 경고를 `exc/empty/offtarget`로 분해 + call-site 태그 + top1 후보 덤프 → 오피스 한 줄 relay만으로 원인 좁힘 (`47cda7e`, `6aa23af`, `ca31403`).
- **tight scale band + tolerance widen**: scale 0.60이 작은 박스를 주기 SEM distractor에 매칭시키던 것을 `(0.85,1.0,1.15)` band로 차단 (`463426b`); GT tolerance `0.20 → 0.30`으로 참 localization(0.20~0.24)을 1~6px 차로 놓치던 것 복구, distractor(≥0.42)는 계속 기각 (`b7cb567`).
- **결과**: `w_sugg 0 → 1` (양 modality). box-fidelity 경로 정상화.

> 신규 env 노브(`REREGISTER_MAX_RECIPES` / `REREGISTER_FIDELITY_SCALES` / `REREGISTER_GT_TOL_NORM`) 전부 `golden_eval_config` 브리지.

## 2. Phase 2 — E-frame confirmation · **실험 종료 / 코드는 보존**

Phase 1의 latent-risk를 fail(E) 프레임 증거로 **confirmed**로 승급하려는 시도. SDD(Subagent-Driven Development)로 6태스크 전량 구현 + 태스크별 리뷰 + 최종 whole-branch opus 리뷰까지 완수 (`af18d7b`..`61f5222`, 최종 48 테스트 통과, merge-yes/no Critical).

**설계 핵심**: `TIER_WEIGHT`에 `E_CONFIRMED` 최상위 추가 + `_e_confirm(s_rep, e_rep)` 규칙 — high-S 전제(`s_rep >= S_FLOOR`) 후 `(s_rep - e_rep >= COLLAPSE_MARGIN) or (e_rep <= E_FLOOR)`. E 프레임 free-search proposer는 bit-parity로 구현(`_gt_in_topk`와 `cand_scores[0] == max(c.score)` 정합성 양 경로 추적 확인).

**부수 인프라** (`039303b`): 전용 `EFRAME_ROOT` 분리 + 시작 시 dataset-health 사전점검 — `confirmed 0`이 데이터 탓(E/rcp 부재)인지 임계 탓인지 구분. (12 신규 테스트)

## 3. 왜 Phase 2에서 멈추고 다음 단계로 갔는가

오피스 실측 2회로 **신호 자체가 신뢰 불가**임을 확정했다.

### 오피스 실행 #1 (기본 임계) — confirmed 0

```
dataset health: 117 recipes | confirm-capable 28 | E-bearing 28 | incomplete 89
[DIGEST] om[strong 42, confirmed 0] | sem[strong 84, confirmed 0]
e_confirm on: S_FLOOR=0.6 E_FLOOR=0.5 COLLAPSE_MARGIN=0.15
STRONG 샘플: 0.206->-(n_e=0)  0.308->-(n_e=0)  0.244->0.187(n_e=1)
```

- **진단**: 실제 점수대는 **~0.2-0.3**인데 기본 임계는 ~0.5-0.6 가정. `_e_confirm` 첫 줄 `if s_rep < S_FLOOR: return False`라 collapse 로직 도달 전 100% 탈락.
- **데이터 제약**: 117 중 **28만 E-bearing**(89는 E 없음), 그 28도 per-recipe E 장수가 얇음(n_e=1).

### 오피스 실행 #2 (illustrative 임계로 smoke test) — 5/6이 false positive

`S_FLOOR=0.15 / E_FLOOR=0.20 / COLLAPSE_MARGIN=0.05`로 낮춰 파이프라인 발화 확인:

| row | s_rep → e_rep | delta | 판정 |
| --- | --- | --- | --- |
| OM  | 0.244 → 0.187 | 0.057 | 유일한 진짜 collapse 후보 (그마저 noise floor 미측정) |
| SEM | 0.155 → 0.122 | 0.033 | E-floor 분기로만 확정 = **false positive** |
| SEM | 0.182 → 0.162 | 0.020 | **false positive** |
| SEM | 0.158 → 0.153 | 0.005 | **false positive** |

SEM 3건 전부 점수대가 ~0.15-0.25라 S·E 둘 다 floor 밑 → '낮은 점수'를 'collapse'로 오판. `E_FLOOR` 절대임계가 이 데이터에서 무효.

### 결론 (사용자 합의)

**근본 원인 = rcp align-key가 충분히 distinctive하지 않다.** s_rep가 *success* 프레임에서도 낮음(0.15-0.31). key가 S에서도 약하니:

- score-collapse는 'S에서 높다가 E에서 무너지는' key를 전제하는데, Phase 1이 flag하는 recipe는 **애초에 S에서 약한 key**라 **무너질 높이가 없는 구조적 모순**.
- 기존 증거(`project_matcher_flat_chamfer_distinctiveness`, `project_e_images_no_crosshair`: E 매처 변별력 ≈0.62 flat)와 일치하고, production이 rcp보다 consensus 템플릿을 *우선*하는 아키텍처 결정과도 일치.

→ **Phase 2는 이 데이터에서 신뢰할 신호가 아니므로 실험 종료.** 코드/테스트(60 tests)는 보존(더 강한 신호 생기면 재활성), `EFRAME_ROOT`/dataset-health는 유용해 유지.

## 4. 다음 단계 — matcher 개선 · **착수**

Phase 2가 드러낸 진짜 병목은 **매처/rcp의 distinctiveness·discrimination 약함** — 이 워크스트림 전반의 cross-cutting bottleneck. 두 레버:

- **(a) re-registration** — Phase 1 신호 그대로 활용(더 distinctive 영역으로 align-key 재등록).
- **(b) matcher 자체 개선** — 사용자 지정 다음 job. **template-bank 매칭 벤치 experiment** spec 작성 시작 (`697b17a`, `6ae2ce1` — heatmap-primary로 전환, RRF를 extra arm으로).

> 미규명(향후 분리 필요): rcp **이미지** 약함 vs **matcher** 약함이 아직 분리 안 됨(둘 다 prior 증거 있음).

---

## 진행 현황 한눈에

| 항목 | 상태 |
| --- | --- |
| Phase 1 — S-only 재등록 스크리닝 | ✅ 완료 |
| Phase 1 — box-fidelity 버그 | ✅ 해결 (`w_sugg 0→1`) |
| Phase 2 — E-frame confirmation 구현 | ✅ 코드 shipped (60 tests) |
| Phase 2 — 오피스 실측 검증 | ⛔ 종료 (신호 신뢰 불가) |
| EFRAME_ROOT + dataset-health | ✅ 유지 |
| 다음 단계 — template-bank matcher | 🔬 spec 작성 중 |

---

## 부수 미해결

- `8 overlay suggestion 파싱 실패` (Phase 1 box-suggest 오버레이 렌더, E-confirm 무관 — verbatim 라인 확보 시 별도 조사).
- 기본 `REREGISTER_E_CONFIRM=1`은 실데이터서 오해 소지 출력(false-positive confirmed) → 기본 0으로 내리는 1줄 변경 권장(미적용).
