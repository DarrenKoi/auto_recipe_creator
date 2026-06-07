# 평가 방법 입문 (3) — 실험 설계: 2×2 A/B, confound, leakage

> 시리즈 3번(마지막). 선행: [eval_guide_02_localization_metrics_ko](eval_guide_02_localization_metrics_ko.md)
> 이 문서의 한 줄: **지표 하나를 "옛 방식 vs 새 방식"으로 공정하게 비교하려면
> 변수를 하나씩 분리(2×2)하고, 정답 유출(leakage)을 막고, 같은 표본끼리(paired) 견줘야 한다.**

대상 코드: `golden_localization_eval.py` 의 `TEMPLATES`/`FRAMES`(2×2 정의),
`_print_summary._contrast()`(delta 출력), `_inpaint_crosshair`(leakage 방지).

---

## 1. 우리가 검증하려는 변화 두 가지

새 방식(NEW)은 옛 방식(OLD)에 두 가지를 바꿨다.

1. **template 을 바꿨다 (`center` → `box`)**
   - `center`(옛): 등록 이미지의 *정중앙을 면적 비율로* 잘라 template 으로 씀.
   - `box`(새): 등록 이미지에 그려진 **흰 unique-area box 안쪽만** 깨끗이 잘라 씀.
   - 가설: box 안쪽이 "이 위치만의 고유한 단서"라 더 잘 맞을 것이다.

2. **프레임을 바꿨다 (`raw` → `inpaint`)**
   - `raw`(옛): 십자선이 그려진 원본 프레임에서 매칭.
   - `inpaint`(새): 십자선을 *지운* 프레임에서 매칭.
   - 가설: 생산엔 십자선이 없으니, 십자선 지운 프레임이 더 현실에 충실하고 더 공정하다.

이제 질문: **"NEW 가 OLD 보다 낫다"를 어떻게 증명하나?** 그냥 OLD 한 번, NEW 한 번 돌려
비교하면 안 될까? — 안 된다. 변수를 *두 개 동시에* 바꿨기 때문이다.

---

## 2. 왜 변수를 하나씩 분리하나 — ablation 과 2×2 factorial

### 문제: 변수를 한꺼번에 바꾸면 "원인"을 모른다

OLD(center+raw)와 NEW(box+inpaint)만 비교해서 점수가 올랐다고 하자. 그럼 **무엇 덕분**인가?
template 덕분? 프레임 덕분? 둘 다? 알 수 없다. 두 변수가 **엉켜(confounded)** 있다.

> 비유: 운동 *그리고* 식단을 동시에 바꿔 살이 빠졌다. 운동 효과인가 식단 효과인가? 모른다.

### 해법: 2×2 factorial design (요인 설계)

변수 2개 × 각 2상태 = **4가지 조합을 전부** 만든다. 코드:

```python
TEMPLATES = ("center", "box")     # 변수 1
FRAMES    = ("raw", "inpaint")    # 변수 2
```

```
              frame=raw          frame=inpaint
tpl=center    OLD baseline       crosshair-only (frame 만 바꿈)
tpl=box       box-only           NEW (둘 다 바꿈)
            (template 만 바꿈)
```

이제 **한 칸씩 옆으로** 비교하면 *한 변수의 순효과*가 분리된다. 이게 **ablation**(절제 실험,
재료를 하나씩 빼보며 그 재료의 역할을 보는 것)이다.

- `center__raw → box__raw`: 프레임 고정, **template 만** 바꿈 → template 효과
- `center__raw → center__inpaint`: template 고정, **frame 만** 바꿈 → frame 효과
- `center__inpaint → box__inpaint`: 둘 다 honest 한 상태에서 template 효과 (**主판정**)

---

## 3. data leakage (정답 유출) — 왜 `raw` 끼리가 아니라 `inpaint` 끼리 비교하나

이게 이 설계에서 가장 미묘하고 중요한 부분이다.

### raw 프레임에는 정답이 그려져 있다

`raw` 프레임에는 **십자선이 보인다**. 그런데 그 십자선이 곧 **정답(GT)** 이다.
즉 입력 안에 답이 들어있다. 매처가 십자선의 강한 edge 에 *lock-on* 하면, 실력이 아니라
**정답을 베껴서** 점수가 올라간다. 이걸 **data leakage** 라고 한다.

> 비유: 시험지 귀퉁이에 정답이 인쇄돼 있는 시험. 점수가 높아도 실력의 증거가 못 된다.
> 게다가 생산 현장(live)에는 십자선이 *없다*(우리가 찍을 위치를 *찾는* 중이니까).
> 따라서 raw 점수는 "현장에서 절대 안 일어날 치팅을 포함한" 부풀린 점수다.

### inpaint: 정답을 먼저 기록하고, 입력에선 지운다

그래서 핵심 순서(`_process_msr`)는:

```
1) raw 에서 십자선 검출 → 그 좌표를 정답(GT)으로 "먼저 기록"
2) 그 자리를 inpaint 로 지움 (cv2.inpaint, 주변 픽셀로 자연스럽게 메움)
3) 십자선 없는 프레임에서 매칭 → 예측점을 1)의 GT 와 비교
```

**정답은 보존하되, 매칭은 정답이 안 보이는 프레임에서.** 이렇게 해야 점수가 "현장에서의 진짜 실력"이 된다.

### 결론: 主판정은 `inpaint` 끼리

```
center__inpaint  vs  box__inpaint     ← 둘 다 십자선 지운 honest 프레임. 公正.
```

- `raw` 셀(center__raw, box__raw)은 leakage 가 섞여 있어 **참고용**이다.
- 옛 verdict 가 한때 `center__raw`(OLD) vs `box__inpaint`(NEW)를 비교했는데,
  이건 template 과 frame 을 *동시에* 바꾼 데다 한쪽에 leakage 까지 끼어 혼동스러워 **폐기**됐다.
  (`_print_summary` 마지막 주석 참고.)

### frame 효과를 보는 법 — raw→inpaint 는 떨어지는 게 정상

```
center__raw → center__inpaint
```
이건 "십자선을 지웠더니 점수가 얼마나 *떨어지나*"를 본다. raw 가 치팅으로 부풀려져 있었다면
inpaint 에서 떨어지는 게 **정상**이다. 그 낙폭이 곧 "raw 가 정답으로 치팅하던 양"이다.

---

## 4. delta(Δ)와 verdict — 효과를 숫자로

두 셀을 비교할 때, 코드(`_contrast`)는 **delta = 새 − 옛** 을 찍는다.

```
rank1_hit:  base=0.55  new=0.70  delta=+0.15  => better
gt_in_topk: base=0.80  new=0.88  delta=+0.08  => better
```

- delta > 0 → `better`, delta = 0 → `동률`, delta < 0 → `worse [!]`
- delta = **effect size(효과 크기)**. "방향(좋아졌나)"과 "크기(얼마나)"를 동시에 본다.

主판정은 **`box__inpaint` − `center__inpaint` 의 rank1_hit delta** 다. 이게 양수이고 의미 있게
크면, "honest 한 조건에서 box template 이 위치추정을 더 잘한다" = 새 전처리 채택 근거.

---

## 5. paired vs unpaired — "같은 표본끼리 비교했나"

공정한 비교의 또 다른 조건: **같은 프레임 집합**을 두 방식에 통과시켜야 한다.
같은 학생들에게 두 시험을 보여야 난이도 차이가 아닌 *방식 차이*를 보는 것과 같다.

- **paired (쌍대)**: 두 셀이 *똑같은 프레임들*을 채점 → delta 가 순수하게 방식 차이.
- **unpaired (비-쌍대)**: 두 셀의 프레임 집합이 *다름* → delta 에 "표본 차이"가 섞여 해석 주의.

### 우리 경우의 함정

`box` 셀은 **흰 box 가 검출된 recipe 의 프레임만** 채점할 수 있다(box 가 없으면 template 을 못 만듦).
반면 `center` 셀은 모든 프레임을 채점한다. 그래서 `box` 셀의 n 이 `center` 셀보다 작으면,
둘은 *다른 프레임 집합* = unpaired 다. 코드가 이때 경고한다:

```
[!] 비-쌍대(unpaired): n center__inpaint=120 vs box__inpaint=78 — 같은 프레임 집합 아님. delta 해석 주의.
```

> 읽는 법: box 셀 n 이 center 셀 n 보다 많이 작으면, delta 를 "방식 효과"로 곧이곧대로 믿지 말고
> "box 가 검출된 쉬운 recipe 들만의 점수일 수 있다"는 가능성을 함께 의심한다.

---

## 6. confound 다시 보기 — 위치추정이 면역인 이유 (복습)

시리즈 1에서 bACC 의 **cross-recipe confound**(recipe 마다 점수 기준선이 달라 전역 임계가
오염되는 문제)를 봤다. 2×2 의 각 셀이 위치추정 지표를 쓰는 덕에, 이 confound 는 사라진다:

- hit/miss 는 *프레임 안에서 자기 십자선까지 거리*로 정해진다 → recipe 간 임계 공유 없음.
- 그래서 셀 간 delta 는 (paired 만 지키면) "방식 차이"로 깨끗이 읽힌다.

> 즉 2×2 의 구조는 `align_index_ablation` 에서 빌려왔지만, **셀 안의 지표를
> "confound 있는 bACC" → "confound 없는 위치추정"으로 갈아끼운 것**이 핵심 개선이다.

---

## 7. 전체를 한 장으로 — 표 읽는 순서

오피스 stdout 을 받았을 때 읽는 **권장 순서**:

```
① GT 위생 먼저
   crosshair_detect_rate 가 낮나? → 낮으면 STOP. 정답이 부실 → 아래 숫자 믿지 말 것.

② 主판정
   box__inpaint 의 rank1_hit_rate 가 매처의 진짜 "한 방 명중률".
   center__inpaint 대비 delta(+면 box 채택 근거). 둘 다 honest(십자 제거).

③ 분해(원인 진단)
   - center__raw → center__inpaint : raw 가 치팅하던 양 (떨어지는 게 정상)
   - center__inpaint → box__inpaint : template 순효과 (主판정)
   - center__raw → box__raw         : raw 에서의 template 효과 (치팅 포함, 참고만)

④ paired 점검
   box 셀 n 이 center 셀 n 보다 크게 작으면 unpaired 경고 → delta 해석 보수적으로.

⑤ 정밀도
   rank1 이 같아도 median/p90 dist_norm 낮은 쪽이 더 안정적(경계에서 덜 위태로움).

⑥ recall 갭
   rank1 낮은데 gt_in_topk 높으면 → 후보엔 정답 있음 → 재정렬로 메울 여지(topk_not_rank1).
```

---

## 8. 한 줄 치트시트

```
2×2 factorial   : 변수 2개를 하나씩 분리 → 어느 변화가 효과의 원인인지 안다
ablation        : 재료(변수)를 하나씩 켜고 끄며 그 역할을 측정
data leakage    : 정답(십자선)이 입력에 섞여 점수가 부풀려짐 → inpaint 로 제거
inpaint 끼리 비교 : 둘 다 honest(정답 안 보임) → 公正한 主판정
delta (Δ)       : 새 − 옛 = 효과의 방향+크기. 主판정 = box__inpaint − center__inpaint (rank1)
paired/unpaired : 같은 프레임 집합인가. box 셀 n < center 셀 n 이면 unpaired 주의
confound        : 위치추정 지표라 recipe 간 교란에서 자유로움(bACC 대비 개선점)
읽는 순서        : 위생 → 主판정 → 분해 → paired → 정밀도 → recall 갭
```

---

## 9. 시리즈를 마치며

- 0 [overview/glossary](eval_guide_00_overview_glossary_ko.md) — 전체 그림
- 1 [classification vs localization](eval_guide_01_classification_vs_localization_ko.md) — 과제 정의
- 2 [localization metrics](eval_guide_02_localization_metrics_ko.md) — 지표 낱낱이
- 3 (이 문서) — 실험 설계

이제 [golden_localization_office_runbook](../golden_localization_office_runbook.md) 의 §4 "결과 읽는 법"이
한 줄도 빠짐없이 이해될 것이다. 거기서 막히면 이 시리즈의 해당 절로 돌아오면 된다.
