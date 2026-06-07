# Align 성공(golden) 데이터셋 수집 플랜

> 작성일: 2026-05-29 · **갱신: 2026-06-02 (재조정)**
> 관련: `align_similarity.py`(fail 진단·헬퍼 출처), 신규 `success_vs_fail_compare.py`(golden 처리·비교),
>        분석 저널 `journals/260602/260602_075313_mi-reranker-ruled-out-contour-next.md`
>
> **2026-06-02 재조정 요약:** (1) fail 폴더의 S 로 템플릿을 만드는 **재등록은 보류** — fail 로 모은
> 폴더의 S 는 표본이 적고(결국 실패한 run 의 성공 step) 편향. (2) golden 의 주 목적이 "재등록
> 템플릿 만들기"에서 **success vs fail 의 rcp↔msr 차이 지표화 → 엔지니어 가이드라인**으로 이동.
> (3) golden 처리는 align_similarity 를 직접 돌리지 않고 **전용 모듈 `success_vs_fail_compare.py`**
> 가 한다(golden 은 E 가 없어 align_similarity 의 S/E 블록이 반쪽). (4) align-fail **런타임의 우선
> workstream 은 별도** — contour reranker + VLM(영역)/CV(좌표) 가 msr 에서 align point 선택.

## 1. 왜 필요한가

"rcp 참조를 교체(재등록)할지" 를 **CV 유사도만으로** 판단하기에는 그것만으로는 부족하다.
현재 staleness 점수는 다음 세 요인에 오염되어 있다.

1. **matcher 자체가 약함** — 정답(S-at-crosshair)에서도 median score 0.62. 낮은 점수가
   "rcp stale" 인지 "matcher 약함" 인지 절대값만으로는 구분 불가.
2. **crosshair 검출 정확도(현재 S 79%)** — ROI 가 어긋나면 점수 하락.
3. **S 라벨 신뢰** — false-positive S 면 비교 기준 자체가 틀림.

→ 절대 임계 대신 **상대 기준**(rcp-vs-S 를 S-vs-S 일관성과 비교)을 쓰면 1·2 를 상쇄할 수 있다.
다만 그 상대 기준의 **임계를 실측으로 calibration** 하려면 "정상" 분포가 필요하고, 그 정상 분포를
제공하는 것이 바로 **항상 성공하는(golden) 데이터셋**이다.

## 2. golden 데이터셋의 역할

1. **(주, 2026-06-02 신규) success vs fail 비교 → 엔지니어 가이드라인** — golden(성공)과 fail 의
   **recipe-정규화된 rcp↔msr 차이**(상대 ratio) 분포를 나란히 비교해, align 성공/실패가 rcp
   staleness 와 어떻게 연관되는지 지표화한다. 산출물 = "이 recipe 들의 key 가 drift 했으니 재등록
   검토" 가이드 + 실측 임계.
2. **임계 calibration anchor** — (유지) 건강한 rcp-vs-S 유사도 분포를 알아야 stale 경계를
   cold-start 추측이 아니라 실측 percentile 로 정한다.
3. **파이프라인 control (거짓 양성 0)** — (유지) staleness 방법이 golden recipe 의 rcp 를 잘못
   stale 로 찍으면 방법이 틀린 것. 거짓 양성률 검증용.
4. **(격하) recipe 별 consensus 참조** — 재등록(템플릿 교체) 보류로 후순위. 추후 재등록을 재개하면
   golden 의 풍부한 S 가 fail 폴더 S 보다 나은 consensus 소스가 됨(현재는 사용 안 함).

## 3. 무엇을, 어떻게 수집하나

### 수집 대상

- **항상(또는 거의 항상) align 성공하는 recipe** 들.
- recipe 당 **여러 장의 성공 측정** (정상 공정 변동을 포괄 — wafer/lot/시간 분산).
- 각 측정에 대해: rcp 참조(IMAP0001 OM / IMAP0002 SEM, 흰 box 포함) + msr 성공 이미지
  (crosshair = ground-truth align point).

### 수집 경로 (주의: fail 기반과 다름)

- 현재 `align_images/` 는 **align-fail 알람으로 트리거되어** 수집된다. 항상 성공하는 recipe 는
  알람을 울리지 않으므로 이 데이터셋에 **없다.**
- 따라서 golden set 은 **의도적으로 수집**해야 한다. 후보 경로:
  - MES/이력에서 선정한 "성공 recipe" 의 과거 성공 측정 이미지를 직접 받아온다.
  - 또는 정상 운영 중 성공 케이스를 별도로 캡처/저장한다.
- 저장 레이아웃은 기존 규약을 재사용:
  `align_images_golden/<eqp>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr}` (S 만).

### 규모 가이드 (2026-06-02 갱신 — 임계 calibration 기준)

두 축이 *서로 다른 이유로* 작동한다:

- **recipe당 S 장수** → per-recipe consensus·S-internal CV 의 신뢰도. median 은 ~5장부터 안정되지만
  **CV 추정은 표본이 적으면 출렁인다** → **recipe당 ≥ 8~10장**, 그리고 **wafer/lot/시간에 분산**.
  (한 wafer 에서 10장 ❌ — 정상 공정변동을 담아야 healthy 분포가 인위적으로 좁아지지 않는다.)
- **recipe 개수** → healthy 분포 percentile(=임계)의 정밀도. 단계적으로:
  - **부트스트랩 ~10 recipe** — 파이프라인 검증 + sanity. 임계는 *잠정*.
  - **사용 가능 ~30 recipe** — 하위 percentile(p10) 임계가 안정. fail 의 판정가능 recipe(~26)와
    균형 → success vs fail 비교 검정력 확보.
  - **견고 50+ recipe** — per-modality(OM/SEM 각각) 임계까지. modality별 분포가 다르면 각 ≥ 20~30.
- **균형 원칙:** golden recipe 수 ≥ fail 판정가능 수(~26) 여야 비교가 한쪽으로 안 기운다.
- **단계 진행:** 10개로 끝까지 한 번 돌려(파이프라인·self-test) 확인 → 30+ 로 확장해 실제 임계 확정.
  50 모일 때까지 착수를 막지 말 것.

## 4. golden set 으로 무엇을 정하나 (acceptance)

1. **healthy 분포 측정**: golden recipe 들의 rcp-vs-S(consensus) 유사도 분포 → 하위 percentile 을
   stale 임계로 설정 (단기 코드의 `RELATIVE_STALE_RATIO` 를 실측값으로 교체).
2. **거짓 양성 0 검증**: 방법이 golden rcp 를 stale 로 찍지 않아야 한다.
3. **S 내부 일관성 baseline**: S-vs-S 일관성(CV)의 정상 범위 → "S inconsistent (CV 판단 불가)"
   경계(`S_INCONSISTENT_CV`)를 실측으로 확정.

## 5. 단기(이번 주) 작업과의 관계

golden set 을 모으기 전에도 **기존 S 217장으로 부트스트랩**한다.

- `align_similarity.py` 의 staleness 를 **절대 점수 → 상대 기준**으로 교체:
  recipe 별 rcp-vs-S(consensus) 를 **S-vs-S 일관성**과 비교.
  - S 끼리 잘 뭉치는데 rcp 만 동떨어짐 → rcp outlier = **stale**(강한 신호, matcher 약함 상쇄).
  - S 끼리도 안 뭉침 → consensus 불신 → **"CV 판단 불가, golden/엔지니어 필요"** 로 분리.
- 이때 임계(`RELATIVE_STALE_RATIO`, `MIN_S_FOR_CONSENSUS`, `S_INCONSISTENT_CV`)는 cold-start.
  **golden set(다음 주)으로 실측 calibration** 하여 확정한다.
- **최종 교체 결정은 자동 단독으로 하지 않는다.** CV 는 후보를 거르고, 플래그된 것만
  엔지니어가 확인한다 (재등록 비용이 있으므로).

## 6. 산출/체크리스트

- [ ] 성공 recipe 후보 목록 선정 (부트스트랩 ~10 → 사용가능 ~30, recipe 당 ≥ 8~10 S, wafer/lot/시간 분산)
- [ ] `align_images_golden/<eqp>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr}` 적재 (사용자 직접)
- [x] `success_vs_fail_compare.py` 구현 (+ Mac 합성 self-test 통과, 2026-06-02)
- [ ] golden 실행 → healthy 상대 ratio·S-internal CV 분포 측정
- [ ] `RELATIVE_STALE_RATIO` / `S_INCONSISTENT_CV` 를 실측 percentile 로 확정
- [ ] 거짓 양성 0 검증 (golden rcp 가 stale 로 안 찍히는지)
- [ ] 확정 임계를 fail 데이터셋에 적용 → success vs fail 분포 비교 + drift recipe 가이드라인(`guideline.md`)

## 7. 처리 모듈 & 흐름 (2026-06-02 확정)

**모듈:** `poc/workflow_2/success_vs_fail_compare.py` (standalone, `uv run python ...`, CLI args 없음).
`align_similarity.py` 의 헬퍼(`_consensus`, `_mi`, `_edge_density`, `_lap_var`, 상대 staleness 로직)만
import 하고 그 파일 자체는 건드리지 않는다. golden 은 E 가 없어 align_similarity 의 S/E·truth-forced·
gt-in-topk 블록이 반쪽이므로, 이를 직접 실행하는 대신 전용 모듈이 healthy 기준선+비교만 담당한다.

**입력:** golden 루트 `align_images_golden/` (S 만) + 기존 fail 루트 `align_images/`. 둘 다
`align_fail_assets` leaf 글로브로 순회(golden 용 루트 인자만 추가).

**recipe별 지표(헬퍼 재사용):** S-at-crosshair crop → `_consensus`(median) → ① rcp_vs_consensus
(matcher score + MI) ② S_internal CV(MI) ③ **상대 ratio = rcp_vs_consensus / S_internal_median**
(recipe 정규화 — recipe 간 비교의 핵심) ④ sharpness(`_lap_var`).

**흐름:** golden → healthy 분포 percentile → 임계 확정(거짓양성 0 검증) → fail 에 적용 →
success vs fail 분포 나란히 비교 + drift recipe 플래그.

**출력:** stdout 요약 + `DEBUG_IMAGE_DIR/success_vs_fail/<ts>/{golden_rows.jsonl, fail_rows.jsonl,
compare_summary.json, guideline.md}`. `guideline.md` 는 한국어 — 엔지니어용 "재등록 검토 recipe" 가이드.

**원칙:** 절대 차이 점수는 recipe 가 다르면 비교 불가 → 항상 상대 ratio 로 비교. 최종 재등록 판정은
자동 단독 금지 — CV 는 후보를 거르고 플래그된 것만 엔지니어가 확인.

> **별도(우선) workstream:** align-fail 런타임의 contour reranker + VLM/CV align-point 선택은 이 문서
> 범위 밖. 근거·계획은 `journals/260602/260602_075313_mi-reranker-ruled-out-contour-next.md` 참조.
