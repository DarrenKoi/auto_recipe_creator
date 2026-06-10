# Tier 1.1 검증 설계 — cond-box-crop localization, miss-distance bin 층화 (2026-06-11)

> 로드맵 [2026-06-11-matching-improvement-roadmap](2026-06-11-matching-improvement-roadmap.md)
> §1.1 의 **validate-first** 단계. White-box unique-region cropping 을 production 으로
> 포팅하기 *전에*, cond-box-crop+offset 이 **far/very-far 구조적 miss 를 실제로 살리는지**를
> golden 데이터로 측정해 게이트한다. 본 세션 산출물은 lab 리포팅 확장 + office 실행 준비이며,
> production 코드는 없다(포팅은 게이트 통과 후 별도 spec).

---

## 1. 배경 / 게이트 질문

**게이트 질문:** cond.box_ltrb 로 crop 한 box template(+decoupled align_offset)이 center-crop
대비 align point localization 을 올리는가 — 특히 **far/very-far 구조적 displacement** 구간에서?

집계(aggregate) 단독으로는 답할 수 없다:
- 과거 box-crop 신호는 약했다(`free_best_box bACC 0.61`, 일부 "역전") —
  [[project_matcher_flat_chamfer_distinctiveness]].
- 로드맵 §0: 잔여 miss 의 ~89% 가 구조적 far/very-far displacement. 집계가 wash 여도
  far-bin lift 가 숨어 있을 수 있고, 그 반대도 가능 → **"집계 recall 단독 금지"**
  (journal 260608_133422 #4).

따라서 **miss-distance bin 별** box-vs-center 비교가 게이트의 핵심 산출물이다.

## 2. 현황 (이미 존재하는 것 — 무수정 유지)

`poc/workflow_2/golden_localization_eval_cond.py` + `golden_localization_eval.py`(=`gle`):

- `cond_template_crop(gray, cond)` — rcp template 을 cond.box_ltrb 로 crop(대칭 inset → crop
  중심 = box 중심).
- `cond_align_offset(box_ltrb, shape)` = image_center − box_center. crop 과 **분리**, cond.txt 만으로 결정.
- `_build_offset_templates_cond(...)` → `center_tpls`, `box_tpls` (modality 별, `align_offset_xy` 보유).
- `gle._localize(templates, frame, GT)` → `{mod, score, align_xy, dist_norm, hit, topk_rank, in_topk}`.
  match 중심 + offset = align point, GT(=cond.crosshair_xy, S frame 의 실제 정렬점)와 비교.
  `dist_norm` = |align point − GT| / template 짧은변 (= `GT_TOL_NORM` 과 동일 척도).
- cells = `arm × frame` = `{center, box} × {raw, inpaint}`. `gle._summarize(all_rows)` 가
  cell 별 `gt_in_topk_rate / rank1_hit_rate / topk_not_rank1_rate / n` 로 집계.
- `lever_verdict(cells["box__inpaint"])`, `_offset_diag_cond(...)` — 기존 진단(유지).

**bit-parity 규칙**([[feedback_ensemble_dev_in_workflow2_then_port]]): 본 변경은 **리포팅 전용**.
matcher / `_localize` / `_build_offset_templates_cond` 등 측정 로직은 건드리지 않는다.

## 3. 변경 (리포팅 전용)

### 3.1 ensemble 매처 hardcode

`_matcher_for_eval()` 의 env 기본값은 baseline 이다. Tier 1.1 검증은 항상 production
ensemble 매처여야 하므로 cond eval 에서 강제한다(공유 함수·테스트 무수정):

```python
FORCE_ENSEMBLE = True   # Tier 1.1 검증은 항상 production ensemble 매처로.
...
def run() -> str:
    if FORCE_ENSEMBLE:
        os.environ.setdefault("ALIGN_USE_ENSEMBLE", "1")   # 명시적 0 은 존중(escape hatch).
```

`setdefault` 라 `ALIGN_USE_ENSEMBLE=0` 명시 시 baseline 비교는 그대로 가능. office 실행
명령에서 env prefix 가 사라진다.

### 3.2 frame 크기 기록

`_process_msr_cond` 에서 row 에 `frame_hw=[h, w]`(=`gray_raw.shape[:2]`)를 1줄 추가한다.
binning A 가 frame 중심·짧은변을 알아야 하기 때문. 측정이 아니라 데이터 캡처라 bit-parity 유지.

### 3.3 bin 층화 리포트 (순수 함수)

신규 `_binned_localization_report(all_rows) -> dict`. `all_rows` 만 후처리(매칭 재실행 없음).
S row 중 `center__inpaint`·`box__inpaint` cell 이 있는 행을 두 방식으로 층화하고, 각 bin ×
arm(center/box) 에 대해 `{gt_in_topk, rank1, n}` 집계:

- **A — structural displacement**: bin 키 = `|GT − frame_center| / frame_short_side`.
  곧 "구조적 far/very-far displacement". box-crop 이 멀리 있는 정렬점을 살리는가에 직접 답.
- **B — rescue framing**: bin 키 = 해당 행 `center__inpaint` 의 `dist_norm`(center-crop 이
  얼마나 빗나갔나). 그 bin 안에서 box arm 이 얼마나 건지는가. 같은 row 데이터 재사용 → 저렴.

`summary["binned"] = {"by_displacement": {...}, "by_center_miss": {...}}` 로 싣고,
`bin × arm` 표를 콘솔에 출력(아래 digest 양식). 기존 cells / lever_verdict / offset_diag 유지.

> 두 층화 모두 `inpaint` frame(crosshair 제거) cell 로 계산한다 — `lever_verdict` 가
> `box__inpaint` 을 쓰는 것과 일관. `raw` cell 도 있으나 본 리포트는 clean 경로에 집중.

### 3.4 bin 임계 (상수, 튜닝 가능)

- **A** (frame 짧은변 비율): `near <0.10 · mid <0.20 · far <0.35 · veryfar ≥0.35`
- **B** (center `dist_norm`, `GT_TOL_NORM` 배수): `hit ≤1× · near <2× · far <4× · veryfar ≥4×`

near/mid/far/veryfar 라벨·경계는 모듈 상수로 두어 office 1차 결과 후 재조정 가능.

## 4. 테스트 (Mac, TDD — fab 데이터 불필요)

`_binned_localization_report` 와 bin 분류 헬퍼는 row dict 에 대한 순수 함수라 합성 입력으로
결정적 검증한다(`test_golden_localization_eval_cond.py` 에 추가):

- 알려진 `frame_hw` + GT 로 displacement bin 경계(near/mid/far/veryfar) 정확 분류.
- 알려진 center `dist_norm` 으로 rescue bin 경계(hit/near/far/veryfar) 정확 분류.
- 합성 cell(`in_topk`/`hit` 조합) → bin × arm 의 `gt_in_topk/rank1/n` 집계가 손으로 센 값과 일치.
- center/box cell 누락 행은 해당 arm 분모에서 제외(빈 bin no-crash).

`FORCE_ENSEMBLE` 의 `setdefault` 의미(미설정→ensemble, `=0`→baseline 유지)도 기존
`test_matcher_for_eval_toggle` 와 충돌 없음을 확인(공유 함수 무수정이므로 그 테스트는 그대로 통과).

## 5. Office 실행 + digest

- 실행(env prefix 불필요): `uv run python poc/workflow_2/golden_localization_eval_cond.py`
- 돌려줄 digest = `summary["binned"]` 표 (260608_133422 placeholder 를 채움):

```
[A] by structural displacement   |  center gt_in_topk / rank1   |  box gt_in_topk / rank1   |  n
  near     (<0.10)               |        __ / __               |        __ / __            | __
  mid      (<0.20)               |        __ / __               |        __ / __            | __
  far      (<0.35)               |        __ / __               |        __ / __            | __
  veryfar  (>=0.35)              |        __ / __               |        __ / __            | __
[B] by center-arm miss           |  (center=baseline)           |  box gt_in_topk / rank1   |  n
  hit / near / far / veryfar     |        ...                   |        ...                | ...
bin 비중: near=__% · far+veryfar=__%   ·   matcher=__   ·   S n=__
```

## 6. 결정 규칙

far+veryfar 구간에서 **box arm `gt_in_topk` 가 center arm 을 의미 있게(예: ≥ +0.05) 상회**하면
→ Tier 1.1 production 포팅 green-light(별도 spec: cond_template_crop+cond_align_offset 를
production vision 모듈로 승격 + `correct_align_fail` 에 crop→match→apply-offset 배선, flag-gated).
그렇지 않으면 → box-crop 은 이 데이터에서 레버가 아님. 결과를 기록하고 1.1 중단, Tier 1.2(anchor-relative)
/ 2.x 로 이동.

## 7. 범위 밖 (YAGNI)

- production 코드(`align_fail_correct`/`cycle` 배선) — 게이트 통과 후 별도 spec.
- matcher / offset 계산 로직 수정 — bit-parity 위반, 금지.
- offset-error 진단 확장 — 기존 `_offset_diag_cond` 로 충분.
- bin 임계 자동 최적화 — 1차 결과 보고 수동 조정.

## 참고

- 로드맵: `specs/2026-06-11-matching-improvement-roadmap.md` §0(병목 재정의), §1.1
- 기존 lab: `golden_localization_eval_cond.py`(cond_template_crop / cond_align_offset /
  _build_offset_templates_cond / run), `golden_localization_eval.py`(_localize / _summarize /
  _matcher_for_eval), `clean_align_image.py`(cursor_to_image / OVERSAMPLE)
- 근거: journal `260608_133422`(cond-eval offset-decouple 계획 + bin 요구), 메모
  [[project_matcher_flat_chamfer_distinctiveness]], [[project_rcp_white_box_unique_area]],
  [[project_align_cond_files_and_coords]]
