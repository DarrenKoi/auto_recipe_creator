# Tool ID 정규화 매칭 — OCR 혼동 글자 보정 (deep dive)

> 대상: `tool_name_match.py`, `util/json_utils.py` (`normalize_tool_text`)
> 상위 개요: `automation_methods_intro.md` §4

---

## 1. 문제 — fuzzy 매칭은 고정 길이 ID 를 망친다

List 탭에서 Tool 행을 찾으려면 OCR/VLM 이 읽은 텍스트를 기대하는 Tool ID 와 비교해야 합니다. 그런데
Tool ID 가 `MCD916` 같은 **고정 길이 코드** 라서 까다롭습니다.

- **fuzzy(편집거리) 매칭** 을 쓰면 한 글자 차이로 다른 Tool 과 매칭됩니다 (`MCD916` vs `MCD915`).
- 반대로 OCR 은 **글자 모양이 비슷한 것을 자주 헷갈립니다** (O↔0, I↔1, B↔8, S↔5…).

두 문제를 한 번에 푸는 방법: **혼동 글자를 표준화한 뒤 정확(exact) 매칭.**

---

## 2. 해결 — confusion map 으로 정규화 후 exact 매칭

`tool_name_match.py`:

```python
_CONFUSION_MAP = {
    "O": "0", "Q": "0",
    "I": "1", "L": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "G": "6",
}

def canonicalize(text: str) -> str:
    # 1) 대문자화  2) 영숫자만 남김  3) 혼동 글자 치환
    ...
# "Tool_O1b" → "TOOLO1B" → "TOOL018"
```

핵심 발상은 이렇습니다. "어차피 OCR 이 O 를 0 으로 잘못 읽는다" → **기대값과 관측값을 둘 다 0 으로
보내** 같은 정규형으로 만든다. 그러면 OCR 오인식이 있어도 정확히 일치하게 됩니다.

### 보수성 — 일부러 안 하는 치환
`D→0`, `T→7`, `A→4` 같은 치환은 **하지 않습니다.** Tool ID 의 의미 있는 접두사(예: MCD 계열의 D)를
망가뜨려 오히려 충돌을 일으키기 때문입니다. "흔하고 안전한 혼동" 만 보정합니다.

---

## 3. best_match — 모호하면 자동 선택을 포기한다

`best_match(items, target_name)`:

1. target 을 canonicalize.
2. 각 spotting item 의 텍스트를 공백으로 토큰화해 각 토큰을 canonicalize.
3. `canonical_target in token_canons` 이면 매칭 후보.
4. **모호성 거부**: 매칭이 **2개 이상 서로 다른 행(row)** 에 걸치면 → `None` 반환 → VLM grounding 으로 위임.
5. **동행 tie-break**: 같은 행 내 복수 후보면 → **가장 작은 bbox**(가장 타이트한 검출)를 채택.

```python
if _distinct_row_count(matched) > 1:
    return None        # 모호 → 자동 매칭 포기
return min(candidates, key=lambda it: _bbox_area(it["bbox"]))   # 동행이면 타이트한 박스
```

`_distinct_row_count()` 는 bbox 의 세로 위치(`top`)로 행을 군집화해 셉니다.

### 왜 "모호하면 포기"가 중요한가 (상사 질문 포인트)
정규화 때문에 **서로 다른 ID 가 같은 정규형이 될 수도** 있습니다 (우연한 충돌). 이때 무리하게
하나를 고르면 **틀린 Tool 을 더블클릭** 할 위험이 있습니다. 그래서 "확신이 없으면 자동화를 멈추고 더
강한 근거(VLM)로 넘긴다"가 규칙입니다. workflow 전역의 "미확인 시 행동 금지" 정신과 같습니다.

---

## 4. List 탭에서의 위치 — 1차는 VLM, fallback 이 정규화 매칭

`workflow_select_tool.py` 의 흐름:

1. **1차: VLM(`ui-venus`→`mai-ui`)** 로 List 영역 crop 에서 Tool 행 클릭점 탐색 (coarse→fine, 최대
   `COARSE_FINE_MAX_ITERS=2` 회로 run-to-run 변동 흡수).
2. **fallback: OCR** — VLM 실패 시 List crop 에 OCR 을 돌려 라인 추출 →
   `normalize_tool_text()`(영숫자·대문자화) → 여기서 `tool_name_match` 의 정규화·매칭 사용.
3. **스크롤 재시도** — 못 찾으면 List 영역 안에서 아래로 스크롤(`_scroll_list_region_down`), 픽셀
   변화(`mean_diff > LIST_CHANGE_THRESHOLD=2.0`)로 목록이 바뀐 걸 감지, 최대 `MAX_SCROLL_ITERS=8`.

> 왜 List 전체에 OCR Spotting 을 1차로 쓰지 않나? 이 UI 의 전체 목록 Spotting 은 과거 **느리고
> garbage** 였습니다(프로젝트 메모리). 그래서 VLM 영역 제안을 1차로 쓰고, 정규화 OCR 매칭은 fallback 으로 둡니다.

---

## 5. 정규화 매칭 vs fuzzy 매칭 — 비교

| | confusion-map + exact (채택) | edit-distance fuzzy |
|---|---|---|
| 고정 길이 ID | 강함 (한 글자도 정확) | 약함 (인접 ID 와 혼동) |
| OCR 혼동(O↔0) | 정규화로 흡수 | threshold 튜닝 필요 |
| 모호성 처리 | 명시적 거부(행 수) | 점수로 뭉개짐 |
| 구현 난이도 | 단순(치환+exact) | threshold·정규화 복잡 |

---

## 6. 핵심 상수 한눈에

| 항목 | 값 | 의미 |
|---|---|---|
| `_CONFUSION_MAP` | 8쌍 | O/Q→0, I/L→1, B→8, S→5, Z→2, G→6 |
| `COARSE_FINE_MAX_ITERS` | 2 | VLM 좌표 재시도 |
| `LIST_CHANGE_THRESHOLD` | 2.0 | 스크롤 후 목록 변화 감지(mean pixel diff) |
| `MAX_SCROLL_ITERS` | 8 | 스크롤 재시도 상한 |
