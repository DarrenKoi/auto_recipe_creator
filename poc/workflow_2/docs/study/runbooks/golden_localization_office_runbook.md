# Golden (S-only) Align-Point 위치추정 — 오피스 실행 런북

> 대상 스크립트: `poc/workflow_2/golden_localization_eval.py`
> 목적: 정답(crosshair)을 신뢰할 수 있는 **성공(S) 데이터**에서, 보정한 align point 가
> 실제 crosshair 에 떨어지는지(rank-1 hit)를 거리로 직접 재고,
> **NEW(흰 box crop + crosshair inpaint) vs OLD(center crop + raw)** A/B 를 판정한다.
> Mac 에서는 합성 self-test 만 가능 — 실판정은 오피스 실데이터로만 한다.

---

## 0. 한눈에 (TL;DR)

```
1) 수집:  poc/workflow_1/align_images_golden/<eqp>/<class>/<recipe>/
            ├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   ← 흰 box 포함 등록 이미지
            └─ align_img_from_msr/   S*.*                              ← 성공 측정 프레임(십자 있음)
2) 실행:  uv run python poc/workflow_2/golden_localization_eval.py
3) 회신:  stdout 의 [GT 위생] + [위치추정 지표 2×2 표] + [VERDICT] 블록을 텍스트로 복사해 전달
```

경로는 코드가 자동 인식한다(`GOLDEN_ROOT = align_images/.. / align_images_golden`). 환경변수 불필요.

---

## 1. 데이터 수집 (S-only)

`align_images`(fail 트리)와 **동일한 레이아웃**으로, 단 루트만 `align_images_golden` 으로 모은다.

```
poc/workflow_1/align_images_golden/
└─ <eqp_id>/<class>/<recipe>/
   ├─ align_img_from_rcp/
   │    IMAP0001.jpg   # OM 등록 align key (흰 unique-area box 포함)
   │    IMAP0002.jpg   # SEM 등록 align key (흰 unique-area box 포함)
   └─ align_img_from_msr/
        S01A0001.jpg   # 성공 측정 프레임 — 파일명 S 로 시작, 십자(crosshair) 보임
        S01A0002.jpg
        ...
```

### 수집 체크리스트 (이걸 어기면 표본이 조용히 버려진다)

| 항목 | 이유 | 어기면 |
|---|---|---|
| msr 파일명이 `S` 로 시작 | `_tool_label` 가 S 로 인식해야 채점 대상 | E/`?` 는 GT 불신 → 제외 |
| 각 S 프레임에 **십자(crosshair) 보임** | crosshair 가 곧 정답(GT) | crosshair 미검출 S 는 위치추정에서 제외(`n_S_no_crosshair` 로 집계) |
| rcp 이미지에 **흰 box** 있음 | `box` template 셀(=NEW)의 필수 입력 | box 셀 비어 verdict "비교 불가" — OLD baseline 만 돈다 |
| recipe 당 S **여러 장**(권장 ≥ 5) | 비율 지표 안정 | 표본 적으면 rank1_hit 변동 큼 |
| recipe **여러 개**(권장 ≥ 5) | recipe 간 일반화 확인 | 단일 recipe 면 과적합 우려 |

> **가장 흔한 실수:** rcp 에 흰 box 가 없으면 이 실험의 핵심인 NEW-vs-OLD A/B 자체가 성립하지 않는다.
> "흰 box 가 등록 이미지에 그려진 recipe" 를 우선 모을 것.

---

## 2. 실행

오피스 PC, 인자 없음:

```bash
uv run python poc/workflow_2/golden_localization_eval.py
```

### 실행 직후 확인 — 실데이터를 읽고 있나?

- 출력 첫 줄에 **`(SELF-TEST)` 가 없어야** 정상이다.
- `[WARNING] golden 데이터 없음 ... → 합성 self-test` 가 뜨면 **데이터를 못 읽은 것**.
  → 레이아웃/경로(특히 `align_img_from_rcp` / `align_img_from_msr` 디렉터리명)를 다시 확인.
- 정상이면 `[INFO] <recipe_id>: msr N장 (box template OK/없음(폴백))` 가 recipe 마다 찍힌다.

단일 recipe 만 빠르게 보려면(선택):

```bash
ALIGN_EQP_ID=<eqp> ALIGN_CLASS_NAME=<class> ALIGN_RECIPE_NAME=<recipe> \
  uv run python poc/workflow_2/golden_localization_eval.py
```

---

## 3. 회신할 내용 (fab-safe, 텍스트만)

stdout 의 아래 세 블록을 **그대로 복사**해 전달한다. 이미지 반출 불필요 — 숫자/표만으로 판정한다.

1. `=== GT 위생 ===` — S 장수, crosshair 검출률·평균 conf, E/`?` 오염 여부.
2. `=== 위치추정 지표 (2×2 표) ===` — `center__raw / center__inpaint / box__raw / box__inpaint`
   각 셀의 `rank1_hit / gt_topk / topk!=1 / med_dist / p90_dist`.
3. `=== VERDICT: NEW vs OLD ===` — rank1_hit / gt_in_topk 의 delta.

산출물은 오피스 PC 에도 저장된다:
`poc/workflow_2/debug_images/golden_localization_eval/<timestamp>/{rows.jsonl, summary.json, overlays/}`
(overlay JPG 는 GT vs 예측 align point 시각 확인용 — 반출 말고 오피스에서만 참고.)

---

## 4. 결과 읽는 법 (해석 가이드)

- **주 판정 = `rank1_hit_rate`** (생산의 "1발 명중"). `box__inpaint`(NEW) 가 `center__raw`(OLD)보다
  높으면 box+inpaint 전처리 교체 근거.
- **`gt_in_topk_rate`** = proposer recall. NEW 가 여기서만 높고 rank1 은 안 오르면 → 정답이 후보엔
  들지만 1등이 아님 → **reranker 로 메울 수 있는 갭**(`topk_not_rank1_rate` 가 그 크기).
- **`med/p90 dist_norm`** = 명중 정밀도. hit 라도 거리가 크면 경계상 위태로움.
- **GT 위생 게이트**: `crosshair_detect_rate` 가 낮으면(과거 실데이터 ~0.79) 위치추정 표본 자체가
  줄어 지표 신뢰도 하락 — 이 경우 crosshair 검출부터 손봐야 한다.
- **NEW worse [!]** 가 뜨면: box 검출이 오검(엉뚱한 박스)했거나 inpaint 가 패턴을 훼손했을 수
  있음 → 해당 recipe overlay 로 원인 확인 후 회신.

---

## 5. 사전 점검(Mac, 참고)

Mac 에서 합성 self-test 로 파이프라인이 도는 것은 이미 확인됨(코드 무결성 보장).
오피스 실패 시 원인은 **코드가 아니라 데이터**로 좁혀진다 — 본 런북의 §1 체크리스트부터 확인.
