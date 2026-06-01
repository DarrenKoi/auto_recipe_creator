# Align 성공(golden) 데이터셋 수집 플랜

> 작성일: 2026-05-29
> 착수 예정: 2026-06-01 주 (다음 주)
> 관련: `align_similarity.py`(참조 staleness 진단), `align_point_correction_recovery_plan.md`

## 1. 왜 필요한가

"rcp 참조를 교체(재등록)할지" 를 **CV 유사도만으로** 판단하는 것은 단독으로는 부족하다.
현재 staleness 점수는 세 요인에 오염된다.

1. **matcher 자체가 약함** — 정답(S-at-crosshair)에서도 median score 0.62. 낮은 점수가
   "rcp stale" 인지 "matcher 약함" 인지 절대값만으로는 구분 불가.
2. **crosshair 검출 정확도(현재 S 79%)** — ROI 가 어긋나면 점수 하락.
3. **S 라벨 신뢰** — false-positive S 면 비교 기준 자체가 틀림.

→ 절대 임계 대신 **상대 기준**(rcp-vs-S 를 S-vs-S 일관성과 비교)으로 1·2 를 상쇄할 수 있으나,
그 상대 기준의 **임계를 실측으로 calibration** 하려면 "정상" 분포가 필요하다. 그 정상 분포를
제공하는 것이 **항상 성공하는(golden) 데이터셋**이다.

## 2. golden 데이터셋의 역할 (3가지)

1. **임계 calibration anchor** — 건강한 rcp-vs-S 유사도 분포를 알아야 stale 경계를
   cold-start 추측이 아니라 실측 percentile 로 정한다.
2. **recipe 별 consensus 참조** — 한 장의 S 가 아니라 *여러* 성공 측정의 합의로 "현재 align
   영역의 대표 모습" 을 구성 → 교체 대상/교체할 이미지 선정이 robust.
3. **파이프라인 control** — 우리 staleness 방법이 golden recipe 의 rcp 를 잘못 stale 로 찍으면
   방법이 틀린 것. 거짓 양성률 검증용.

## 3. 무엇을, 어떻게 수집하나

### 수집 대상

- **항상(또는 거의 항상) align 성공하는 recipe** 들.
- recipe 당 **여러 장의 성공 측정** (정상 공정 변동을 포괄 — wafer/lot/시간 분산).
- 각 측정에 대해: rcp 참조(IMAP0001 OM / IMAP0002 SEM, 흰 box 포함) + msr 성공 이미지
  (crosshair = ground-truth align point).

### 수집 경로 (주의: fail 기반과 다름)

- 현재 `align_images/` 는 **align-fail 알람으로 트리거된** 수집이다. 항상 성공하는 recipe 는
  알람을 울리지 않아 이 데이터셋에 **없다.**
- 따라서 golden set 은 **의도적 수집**이 필요하다. 후보 경로:
  - MES/이력에서 선정한 "성공 recipe" 의 과거 성공 측정 이미지를 직접 받아온다.
  - 또는 정상 운영 중 성공 케이스를 별도로 캡처/저장한다.
- 저장 레이아웃은 기존 규약을 재사용:
  `align_images_golden/<eqp>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr}` (S 만).

### 최소 규모 (cold-start 가이드)

- recipe 당 성공 측정 **≥ 5장** (consensus 신뢰 가능 최소치). 가능하면 10+.
- recipe **≥ 10개** (분포를 보려면). modality(OM/SEM) 양쪽 포함.

## 4. golden set 으로 무엇을 정하나 (acceptance)

1. **healthy 분포 측정**: golden recipe 들의 rcp-vs-S(consensus) 유사도 분포 → 하위 percentile 을
   stale 임계로 설정 (단기 코드의 `RELATIVE_STALE_RATIO` 를 실측값으로 교체).
2. **거짓 양성 0 검증**: 방법이 golden rcp 를 stale 로 찍지 않아야 한다.
3. **S 내부 일관성 baseline**: S-vs-S 일관성(CV)의 정상 범위 → "S inconsistent (CV 판단 불가)"
   경계(`S_INCONSISTENT_CV`)를 실측으로 확정.

## 5. 단기(이번 주) 작업과의 관계

golden set 이전에도 **기존 S 217장으로 부트스트랩**한다.

- `align_similarity.py` 의 staleness 를 **절대 점수 → 상대 기준**으로 교체:
  recipe 별 rcp-vs-S(consensus) 를 **S-vs-S 일관성**과 비교.
  - S 끼리 잘 뭉치는데 rcp 만 동떨어짐 → rcp outlier = **stale**(강한 신호, matcher 약함 상쇄).
  - S 끼리도 안 뭉침 → consensus 불신 → **"CV 판단 불가, golden/엔지니어 필요"** 로 분리.
- 이때 임계(`RELATIVE_STALE_RATIO`, `MIN_S_FOR_CONSENSUS`, `S_INCONSISTENT_CV`)는 cold-start.
  **golden set(다음 주)으로 실측 calibration** 하여 확정한다.
- **최종 교체 결정은 자동 단독으로 하지 않는다.** CV 는 후보를 거르고, 플래그된 것만
  엔지니어가 확인한다 (재등록 비용이 있으므로).

## 6. 산출/체크리스트 (다음 주)

- [ ] 성공 recipe 후보 목록 선정 (≥10 recipe, recipe 당 ≥5 S)
- [ ] `align_images_golden/` 수집 경로 확정 + 이미지 확보
- [ ] golden set 에 `align_similarity.py` 실행 → healthy 분포 측정
- [ ] `RELATIVE_STALE_RATIO` / `S_INCONSISTENT_CV` 를 실측 percentile 로 확정
- [ ] 거짓 양성 0 검증 (golden rcp 가 stale 로 안 찍히는지)
- [ ] 확정 임계를 fail 데이터셋(217 S)에 적용 → 교체 후보 recipe 리스트 산출
