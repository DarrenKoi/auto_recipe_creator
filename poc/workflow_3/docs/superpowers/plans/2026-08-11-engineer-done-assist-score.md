# engineer-done Assist Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** engineer-done 판정을 카운터 절대값 대신 Recipe Monitor Assist Window 의 score 색상(검정=정상/빨강=실패)과 watch 시작 이후 카운터 차분으로 바꾼다.

**Architecture:** `sem_monitor/assist_score.py` 를 새로 만들어 Assist 패널을 판독한다. 패널 위치와 score 셀 격자는 **VLM+OCR 로 1회만** 잡아 캐시하고, 이후 폴링은 **순수 CV** 로 각 셀의 잉크 색만 본다. `monitor/engineer_done_align_adjustment.py` 는 ROI grounding + 카운터 OCR 경로를 그대로 두고 판정식만 `(delta >= 6) and (streak >= 6)` 로 교체한다.

**Tech Stack:** Python 3.10+, numpy, Pillow, 기존 `poc.workflow_3.vlm.ui_venus_mai_locator` (2단계 로케이터) / `poc.workflow_3.vlm.ocr_spotting` (PaddleOCR spotting) / `poc.workflow_3.debug_artifacts`.

## Global Constraints

- 설계 문서: `poc/workflow_3/docs/superpowers/specs/2026-08-11-engineer-done-assist-score-design.md` — 충돌 시 스펙이 우선.
- **한국어 docstring** 전부. `[INFO]`/`[ERROR]`/`[WARNING]` print 로깅만 사용(`logging` 모듈 금지).
- **`from __future__ import annotations` 금지.**
- `print()` 문자열 안에 em-dash(U+2014) 금지 (오피스 콘솔 cp949). docstring 은 무방.
- 절대 임포트만: `from poc.workflow_3.xxx import ...`.
- **CLI 인자 금지** (argparse/flag 없음). 설정은 `Workflow3Settings` 또는 모듈 상수.
- 테스트는 `uv run python <path>` 로 직접 실행되는 스크립트 형태 (기존 `test_*.py` 관례). `main()` 이 통과 개수를 출력하고 `SystemExit(0/1)`.
- 판정의 대가는 비대칭이다. **모든 애매함은 "아직 아님"(False) 쪽으로 넘긴다.**
- 색 상태는 4가지: `"black"` / `"red"` / `"blank"` / `"unknown"`.
- 열 이름은 `"Addressing1"`, `"Addressing2"`, `"Measurement"` 문자열을 그대로 쓴다.
- 최신 행은 **맨 아래**(`ASSIST_NEWEST_ROW_AT = "bottom"`).
- `parse_spotting_items` 는 `{"text": str, "bbox": {"left","top","right","bottom"}}` 를 돌려준다. 키는 `bbox` 이지 `box` 가 아니다.

---

### Task 1: 잉크 색 분류 (`classify_ink`)

score 숫자 한 칸의 색을 판정하는 순수 CV 함수. 이후 모든 판정이 여기 얹힌다.

**Files:**
- Create: `poc/workflow_3/sem_monitor/assist_score.py`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: 없음 (numpy 만)
- Produces:
  - 상수 `ASSIST_ROWS = 7`, `ASSIST_NEWEST_ROW_AT = "bottom"`, `ASSIST_COLUMNS = ("Addressing1", "Addressing2", "Measurement")`
  - `classify_ink(cell_rgb: np.ndarray) -> str` — `"black"|"red"|"blank"|"unknown"`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/sem_monitor/test_assist_score.py` 를 새로 만든다:

```python
"""Assist Window score 판독 self-test (VLM/실이미지 불필요).

합성 이미지와 합성 OCR 항목만 쓰므로 Mac 에서 그대로 돈다.

    uv run python poc/workflow_3/sem_monitor/test_assist_score.py
"""

import numpy as np

from poc.workflow_3.sem_monitor.assist_score import classify_ink


def _cell(rgb=(240, 240, 240), *, ink=None, ink_px=40):
    """배경 40x20 셀에 잉크 픽셀을 ink_px 개 찍어 돌려준다."""
    cell = np.full((20, 40, 3), rgb, dtype=np.uint8)
    if ink is not None:
        flat = cell.reshape(-1, 3)
        flat[:ink_px] = ink
    return cell


def test_black_ink():
    ok = classify_ink(_cell(ink=(20, 20, 20))) == "black"
    print(f"[{'PASS' if ok else 'FAIL'}] black_ink")
    return ok


def test_red_ink():
    ok = classify_ink(_cell(ink=(200, 20, 20))) == "red"
    print(f"[{'PASS' if ok else 'FAIL'}] red_ink")
    return ok


def test_blank_cell():
    """잉크가 없으면 blank (측정 진행 중인 행)."""
    ok = classify_ink(_cell()) == "blank"
    print(f"[{'PASS' if ok else 'FAIL'}] blank_cell")
    return ok


def test_blank_when_ink_below_min_pixels():
    """안티에일리어싱 몇 픽셀은 잉크로 치지 않는다."""
    ok = classify_ink(_cell(ink=(20, 20, 20), ink_px=3)) == "blank"
    print(f"[{'PASS' if ok else 'FAIL'}] blank_when_ink_below_min_pixels")
    return ok


def test_mixed_ink_is_unknown():
    """흑/적이 반반이면 판정 불가. streak 을 끊어야 하므로 unknown."""
    cell = _cell(ink=(20, 20, 20), ink_px=40)
    flat = cell.reshape(-1, 3)
    flat[20:40] = (200, 20, 20)
    ok = classify_ink(cell) == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] mixed_ink_is_unknown")
    return ok


def main():
    print("[INFO] assist_score self-test 시작")
    results = [
        test_black_ink(),
        test_red_ink(),
        test_blank_cell(),
        test_blank_when_ink_below_min_pixels(),
        test_mixed_ink_is_unknown(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'classify_ink'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/sem_monitor/assist_score.py` 를 새로 만든다:

```python
"""Recipe Monitor Assist Window 의 score 색상을 읽어 측정 성부를 판정한다.

Assist Window 는 tool 창 내부 패널이다(별도 최상위 창이 아님). 3열(Addressing1 /
Addressing2 / Measurement) x 7행으로 최신 7회 측정의 썸네일과 score 가 실시간으로 쌓이고,
**score 숫자의 색이 곧 성부다** - 검정이면 정상 측정, 빨강이면 측정 실패. 측정이 진행
중인 행은 빈칸이다. 최신 행은 맨 아래에 쌓인다.

설계 경계(프로젝트 규칙): VLM 은 패널 *영역*만 1회 식별하고, 색 판정은 전부 CV 가 한다.
우리가 필요한 건 값이 아니라 색이므로 폴링마다 OCR 을 돌리지 않는다 - 읽지 않는 정보는
틀릴 수 없다.
"""

import numpy as np

# 표 형태.
ASSIST_ROWS = 7
ASSIST_NEWEST_ROW_AT = "bottom"  # tool 버전이 다르면 "top".
ASSIST_COLUMNS = ("Addressing1", "Addressing2", "Measurement")

# 색 분류 임계. 배경은 밝고 잉크(글자)는 어둡다는 전제.
INK_MEAN_MAX = 200      # 채널 평균이 이보다 어두우면 잉크로 본다.
INK_MIN_PIXELS = 6      # 잉크가 이보다 적으면 빈칸(안티에일리어싱 무시).
RED_CHROMA_MIN = 60     # max-min 이 이 이상이면 유채색.
RED_DOMINANCE_MIN = 40  # R - max(G,B) 가 이 이상이면 붉은 계열.
RED_RATIO_MIN = 0.30    # 잉크 중 빨강 비율이 이 이상이면 red.
BLACK_RATIO_MAX = 0.10  # 이 이하면 black. 사이는 unknown.


def classify_ink(cell_rgb: np.ndarray) -> str:
    """셀 하나의 잉크 색을 판정한다. "black"|"red"|"blank"|"unknown".

    입력은 RGB numpy 배열이다(PIL Image 를 np.array 로 바꾼 형태). 흑/적 비율이 어느
    쪽으로도 확실하지 않으면 "unknown" 을 돌려준다 - 호출부가 streak 을 끊게 해서
    애매함이 done 판정으로 새지 않게 한다.
    """
    if cell_rgb is None or cell_rgb.size == 0:
        return "blank"
    arr = cell_rgb.astype(np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return "blank"

    red_c = arr[:, :, 0]
    green_c = arr[:, :, 1]
    blue_c = arr[:, :, 2]
    mean = arr[:, :, :3].mean(axis=2)
    ink = mean < INK_MEAN_MAX
    ink_count = int(ink.sum())
    if ink_count < INK_MIN_PIXELS:
        return "blank"

    chroma = arr[:, :, :3].max(axis=2) - arr[:, :, :3].min(axis=2)
    dominance = red_c - np.maximum(green_c, blue_c)
    is_red = ink & (chroma >= RED_CHROMA_MIN) & (dominance >= RED_DOMINANCE_MIN)
    red_ratio = float(is_red.sum()) / float(ink_count)

    if red_ratio >= RED_RATIO_MIN:
        return "red"
    if red_ratio <= BLACK_RATIO_MAX:
        return "black"
    return "unknown"


__all__ = [
    "ASSIST_COLUMNS",
    "ASSIST_NEWEST_ROW_AT",
    "ASSIST_ROWS",
    "classify_ink",
]
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 5/5 cases passed`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "feat(workflow_3): Assist Window score 잉크 색 분류(classify_ink)

검정=정상 측정, 빨강=측정 실패. 흑/적 어느 쪽도 확실하지 않으면 unknown 을 돌려
호출부가 streak 을 끊게 한다 - 애매함이 done 판정으로 새면 엔지니어 작업 중에 창이
닫힌다."
```

---

### Task 2: 행 판정과 연속 정상 개수 (`row_verdict`, `ok_streak`)

셀 3개의 색에서 행 하나의 성부를 정하고, 행 목록에서 최신 연속 정상 개수를 센다. 둘 다 순수 함수라 경계 조건을 촘촘히 테스트한다.

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: Task 1 의 `ASSIST_COLUMNS`
- Produces:
  - 상수 `ASSIST_CRITICAL_COLUMNS = ("Addressing1", "Measurement")`
  - `RowState` 데이터클래스 — 필드 `cells: dict` (열 이름 -> 색), 프로퍼티 `verdict: str`
  - `row_verdict(cells: dict) -> str` — `"ok"|"fail"|"pending"|"unknown"`
  - `ok_streak(rows: list) -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

`test_assist_score.py` 의 import 를 늘리고 테스트를 추가한다:

```python
from poc.workflow_3.sem_monitor.assist_score import (
    RowState,
    classify_ink,
    ok_streak,
    row_verdict,
)


def _cells(addr1="black", addr2="blank", meas="black"):
    return {"Addressing1": addr1, "Addressing2": addr2, "Measurement": meas}


def test_verdict_ok_without_addressing2():
    """Addressing2 는 대개 비어 있다. 없어도 정상 판정이어야 한다."""
    ok = row_verdict(_cells()) == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_ok_without_addressing2")
    return ok


def test_verdict_fail_on_red_measurement():
    ok = row_verdict(_cells(meas="red")) == "fail"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_fail_on_red_measurement")
    return ok


def test_verdict_fail_on_red_addressing1():
    """Addressing1 이 빨강이어도 그 측정은 실패다."""
    ok = row_verdict(_cells(addr1="red")) == "fail"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_fail_on_red_addressing1")
    return ok


def test_verdict_pending_when_measurement_blank():
    ok = row_verdict(_cells(meas="blank")) == "pending"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_pending_when_measurement_blank")
    return ok


def test_verdict_ok_when_only_measurement_present():
    """Addressing1 이 없는 레시피도 Measurement 로 완료를 판정한다.

    없는 칸을 '진행 중' 으로 읽으면 그 레시피는 영영 done 이 되지 않는다.
    """
    ok = row_verdict(_cells(addr1="blank")) == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_ok_when_only_measurement_present")
    return ok


def test_verdict_unknown_beats_ok():
    ok = row_verdict(_cells(meas="unknown")) == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_unknown_beats_ok")
    return ok


def _rows(verdicts):
    """verdict 문자열 목록을 RowState 목록으로 (index 0 = 가장 오래된 행)."""
    mapping = {
        "ok": _cells(),
        "fail": _cells(meas="red"),
        "pending": _cells(meas="blank"),
        "unknown": _cells(meas="unknown"),
    }
    return [RowState(cells=dict(mapping[v])) for v in verdicts]


def test_streak_counts_from_newest():
    ok = ok_streak(_rows(["fail", "ok", "ok", "ok"])) == 3
    print(f"[{'PASS' if ok else 'FAIL'}] streak_counts_from_newest")
    return ok


def test_streak_skips_trailing_pending():
    """최신 행이 측정 진행 중(빈칸)이어도 그 앞의 연속 정상은 살아 있어야 한다."""
    ok = ok_streak(_rows(["ok", "ok", "ok", "pending"])) == 3
    print(f"[{'PASS' if ok else 'FAIL'}] streak_skips_trailing_pending")
    return ok


def test_streak_broken_by_fail_and_unknown():
    ok = (
        ok_streak(_rows(["ok", "ok", "fail", "ok"])) == 1
        and ok_streak(_rows(["ok", "ok", "unknown", "ok"])) == 1
    )
    print(f"[{'PASS' if ok else 'FAIL'}] streak_broken_by_fail_and_unknown")
    return ok


def test_streak_all_ok_is_full_length():
    ok = ok_streak(_rows(["ok"] * 7)) == 7
    print(f"[{'PASS' if ok else 'FAIL'}] streak_all_ok_is_full_length")
    return ok


def test_streak_empty_rows_is_zero():
    ok = ok_streak([]) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] streak_empty_rows_is_zero")
    return ok
```

`main()` 의 `results` 목록에 위 11개 함수 호출을 순서대로 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'RowState'`

- [ ] **Step 3: 최소 구현**

`assist_score.py` 에 추가한다 (`import numpy as np` 아래에 `from dataclasses import dataclass, field` 를 넣는다):

```python
# Addressing2 는 대개 비어 있어 판정에 쓰지 않는다.
ASSIST_CRITICAL_COLUMNS = ("Addressing1", "Measurement")


def row_verdict(cells: dict) -> str:
    """측정 1회(행 하나)의 성부. "ok"|"fail"|"pending"|"unknown".

    pending 을 **Measurement 기준으로만** 판정하는 이유: Addressing2 는 대개 비어 있고
    Addressing1 도 레시피에 따라 없을 수 있다. 없는 칸을 '진행 중' 으로 읽으면 그
    레시피는 영영 done 이 되지 않는다. Measurement 가 최종 결과이므로 그것으로 완료를
    판정하고, Addressing1 은 값이 있을 때만 실패 신호로 쓴다.
    """
    critical = [cells.get(name, "blank") for name in ASSIST_CRITICAL_COLUMNS]
    if any(state == "red" for state in critical):
        return "fail"
    if any(state == "unknown" for state in critical):
        return "unknown"
    if cells.get("Measurement", "blank") == "blank":
        return "pending"
    return "ok"


@dataclass
class RowState:
    """Assist Window 의 행 하나 - 열별 잉크 색과 그로부터 나온 성부."""

    cells: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return row_verdict(self.cells)


def ok_streak(rows: list) -> int:
    """최신 행부터 세어 연속 정상(ok) 개수. fail/unknown 을 만나면 멈춘다.

    최신 쪽의 pending(측정 진행 중)은 건너뛴다 - 아직 결과가 안 나온 행이 그 앞의 연속
    정상 기록을 지우면 안 된다. 목록은 index 0 이 가장 오래된 행이다.
    """
    idx = len(rows) - 1
    while idx >= 0 and rows[idx].verdict == "pending":
        idx -= 1
    streak = 0
    while idx >= 0 and rows[idx].verdict == "ok":
        streak += 1
        idx -= 1
    return streak
```

`__all__` 에 `"ASSIST_CRITICAL_COLUMNS"`, `"RowState"`, `"ok_streak"`, `"row_verdict"` 를 추가한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 16/16 cases passed`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "feat(workflow_3): Assist 행 성부 판정 + 연속 정상 개수(ok_streak)

pending 은 Measurement 기준으로만 본다. Addressing2 는 대개 없고 Addressing1 도 레시피에
따라 없을 수 있는데, 없는 칸을 '진행 중' 으로 읽으면 그 레시피는 영영 done 이 안 된다.

ok_streak 은 최신 쪽 pending 을 건너뛴다 - 지금 측정 중이라는 사실이 그 앞의 연속 정상
기록을 지우면 안 된다."
```

---

### Task 3: score 셀 격자 만들기 (`build_score_grid`)

OCR spotting 결과에서 표의 열 x-범위와 행 y-띠를 뽑아 7x3 격자를 만든다. 여기서 만든 격자는 캐시되어 이후 폴링에서 그대로 재사용된다(셀 위치는 고정, 내용만 바뀐다).

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: Task 1 의 `ASSIST_ROWS`, `ASSIST_COLUMNS`
- Produces:
  - `AssistLayout` 데이터클래스 — 필드 `panel_box: dict`, `grid: list` (행 x 열의 박스 dict), `columns: tuple`
  - `build_score_grid(items: list, panel_size: tuple, *, rows: int = ASSIST_ROWS) -> "AssistLayout | None"`

- [ ] **Step 1: 실패하는 테스트 작성**

`test_assist_score.py` 에 추가한다:

```python
from poc.workflow_3.sem_monitor.assist_score import AssistLayout, build_score_grid


def _item(text, left, top, right, bottom):
    return {"text": text, "bbox": {"left": left, "top": top, "right": right, "bottom": bottom}}


def _panel_items():
    """헤더 3개 + 숫자 4행(부분만 채워진 표)을 흉내낸 OCR 결과.

    행 pitch 30px, 첫 숫자행 top=40. 열: 10-60 / 110-160 / 210-260.
    """
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
    ]
    for idx in range(4):
        top = 40 + idx * 30
        items.append(_item("12", 20, top, 50, top + 18))
        items.append(_item("34", 220, top, 250, top + 18))
    return items


def test_grid_has_full_rows_and_columns():
    layout = build_score_grid(_panel_items(), (300, 260))
    ok = (
        layout is not None
        and len(layout.grid) == 7
        and all(len(row) == 3 for row in layout.grid)
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_has_full_rows_and_columns")
    return ok


def test_grid_extrapolates_missing_rows_by_pitch():
    """표가 부분만 차 있어도 pitch 로 7행을 채운다(행 간격 30px)."""
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_extrapolates_missing_rows_by_pitch: layout None")
        return False
    tops = [row[0]["top"] for row in layout.grid]
    diffs = {tops[i + 1] - tops[i] for i in range(len(tops) - 1)}
    ok = diffs == {30}
    print(f"[{'PASS' if ok else 'FAIL'}] grid_extrapolates_missing_rows_by_pitch: {sorted(diffs)}")
    return ok


def test_grid_columns_follow_headers():
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_columns_follow_headers: layout None")
        return False
    first = layout.grid[0]
    ok = (
        layout.columns == ("Addressing1", "Addressing2", "Measurement")
        and first[0]["left"] == 10 and first[0]["right"] == 60
        and first[2]["left"] == 210 and first[2]["right"] == 260
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_columns_follow_headers")
    return ok


def test_grid_none_without_headers():
    """헤더를 못 읽으면 어느 열이 무엇인지 알 수 없으므로 격자를 만들지 않는다."""
    items = [_item("12", 20, 40, 50, 58), _item("34", 220, 40, 250, 58)]
    ok = build_score_grid(items, (300, 260)) is None
    print(f"[{'PASS' if ok else 'FAIL'}] grid_none_without_headers")
    return ok


def test_grid_none_with_single_number_row():
    """행이 하나면 pitch 를 알 수 없다. 추정하지 않고 실패시킨다."""
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
        _item("12", 20, 40, 50, 58),
    ]
    ok = build_score_grid(items, (300, 260)) is None
    print(f"[{'PASS' if ok else 'FAIL'}] grid_none_with_single_number_row")
    return ok
```

`main()` 의 `results` 에 위 5개를 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'AssistLayout'`

- [ ] **Step 3: 최소 구현**

`assist_score.py` 에 추가한다:

```python
# 헤더 텍스트 매칭용 - 영숫자만 남기고 소문자 비교(OCR 공백/기호 흔들림 흡수).
def _normalize(text: str) -> str:
    """열 이름 비교용 정규화."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _is_score_text(text: str) -> bool:
    """score 로 볼 텍스트인지(숫자만)."""
    stripped = (text or "").strip()
    return bool(stripped) and all(ch.isdigit() for ch in stripped)


@dataclass
class AssistLayout:
    """Assist 패널의 score 셀 격자. 1회 만들어 캐시한다."""

    panel_box: dict
    grid: list        # grid[row][col] = {"left","top","right","bottom"} (패널 crop 좌표계)
    columns: tuple


def build_score_grid(items: list, panel_size: tuple, *, rows: int = ASSIST_ROWS):
    """OCR spotting 항목에서 score 셀 격자를 만든다. 실패 시 None.

    열은 헤더 텍스트로 잡는다(순서로 추정하지 않는다 - Addressing2 가 비어 있으면 숫자
    덩어리가 2개뿐이라 어느 것이 Measurement 인지 알 수 없다). 행은 숫자 항목의 y 중심을
    모아 pitch 를 구한 뒤 rows 개로 외삽한다.

    items 는 패널 crop 좌표계여야 한다. panel_size 는 (width, height).
    """
    if not items:
        return None

    # --- 열: 헤더 텍스트로 x 범위 확정 ---
    header_boxes = {}
    for item in items:
        name = _normalize(item.get("text", ""))
        for column in ASSIST_COLUMNS:
            if name == _normalize(column) and column not in header_boxes:
                header_boxes[column] = item.get("bbox") or {}
    if len(header_boxes) != len(ASSIST_COLUMNS):
        print(f"[WARNING] Assist 헤더 인식 부족({sorted(header_boxes)}) - 격자 생성 실패")
        return None

    # --- 행: 숫자 항목의 y 중심 -> pitch -> 외삽 ---
    header_bottom = max(int(box.get("bottom", 0)) for box in header_boxes.values())
    centers = []
    heights = []
    for item in items:
        if not _is_score_text(item.get("text", "")):
            continue
        box = item.get("bbox") or {}
        top = int(box.get("top", 0))
        bottom = int(box.get("bottom", 0))
        if bottom <= header_bottom:
            continue
        centers.append((top + bottom) / 2.0)
        heights.append(max(1, bottom - top))
    if not centers:
        print("[WARNING] Assist 숫자 항목 없음 - 격자 생성 실패")
        return None

    cell_h = int(round(sorted(heights)[len(heights) // 2]))
    band_centers = _cluster_1d(sorted(centers), tolerance=cell_h)
    if len(band_centers) < 2:
        print("[WARNING] Assist 행이 2개 미만 - pitch 를 알 수 없어 격자 생성 실패")
        return None

    pitch = (band_centers[-1] - band_centers[0]) / float(len(band_centers) - 1)
    if pitch <= 0:
        return None

    # 최신 행이 맨 아래이므로 가장 아래 띠를 마지막 행에 맞추고 위로 채운다.
    last_center = band_centers[-1]
    panel_h = panel_size[1]
    grid = []
    for row_idx in range(rows):
        center = last_center - pitch * (rows - 1 - row_idx)
        top = int(round(center - cell_h / 2.0))
        bottom = int(round(center + cell_h / 2.0))
        row_boxes = []
        for column in ASSIST_COLUMNS:
            box = header_boxes[column]
            row_boxes.append({
                "left": int(box.get("left", 0)),
                "right": int(box.get("right", 0)),
                "top": max(0, top),
                "bottom": min(panel_h, bottom),
            })
        grid.append(row_boxes)

    return AssistLayout(
        panel_box={"left": 0, "top": 0, "right": panel_size[0], "bottom": panel_size[1]},
        grid=grid,
        columns=tuple(ASSIST_COLUMNS),
    )


def _cluster_1d(values: list, *, tolerance: int) -> list:
    """정렬된 1D 값들을 tolerance 이내로 묶어 각 묶음의 평균을 돌려준다."""
    if not values:
        return []
    clusters = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / float(len(group)) for group in clusters]
```

`__all__` 에 `"AssistLayout"`, `"build_score_grid"` 를 추가한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 21/21 cases passed`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "feat(workflow_3): Assist score 셀 격자 생성(build_score_grid)

열은 헤더 텍스트로 잡는다. 순서로 추정하면 Addressing2 가 비었을 때 숫자 덩어리가 2개뿐
이라 어느 것이 Measurement 인지 알 수 없다.

행은 숫자 y 중심에서 pitch 를 구해 7행으로 외삽하되, 행이 2개 미만이면 pitch 를 모르므로
추정하지 않고 실패시킨다. 격자는 1회 만들어 캐시되고 이후 폴링은 그 박스를 그대로 쓴다
(셀 위치는 고정이고 내용만 바뀐다)."
```

---

### Task 4: 프레임에서 행 상태 읽기 (`read_row_states`)

캡처 이미지와 캐시된 격자로부터 폴링마다 행 상태를 만든다. VLM/OCR 없이 순수 CV.

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: Task 1 `classify_ink`, Task 2 `RowState`, Task 3 `AssistLayout`
- Produces: `read_row_states(image, layout: AssistLayout) -> list` (`RowState` 목록, index 0 = 가장 오래된 행)

- [ ] **Step 1: 실패하는 테스트 작성**

`test_assist_score.py` 에 추가한다 (파일 상단 import 에 `from PIL import Image` 추가):

```python
from poc.workflow_3.sem_monitor.assist_score import read_row_states


def _synth_panel(row_specs):
    """행별 (addr1, meas) 색 지정으로 합성 패널 이미지를 만든다.

    row_specs 는 길이 7. 각 원소는 ("black"|"red"|None, "black"|"red"|None).
    None 은 빈칸(잉크 없음).
    """
    image = Image.new("RGB", (300, 260), (240, 240, 240))
    pixels = image.load()
    ink = {"black": (20, 20, 20), "red": (200, 20, 20)}
    for row_idx, (addr1, meas) in enumerate(row_specs):
        top = 40 + row_idx * 30
        for column_left, state in ((20, addr1), (220, meas)):
            if state is None:
                continue
            for dx in range(20):
                for dy in range(10):
                    pixels[column_left + dx, top + dy] = ink[state]
    return image


def _layout_for_synth():
    return build_score_grid(_panel_items(), (300, 260))


def test_read_rows_marks_black_and_red():
    specs = [("black", "black")] * 6 + [("black", "red")]
    rows = read_row_states(_synth_panel(specs), _layout_for_synth())
    ok = len(rows) == 7 and rows[-1].verdict == "fail" and rows[0].verdict == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_marks_black_and_red: "
          f"{[r.verdict for r in rows]}")
    return ok


def test_read_rows_blank_is_pending():
    specs = [("black", "black")] * 6 + [("black", None)]
    rows = read_row_states(_synth_panel(specs), _layout_for_synth())
    ok = rows[-1].verdict == "pending" and ok_streak(rows) == 6
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_blank_is_pending: streak={ok_streak(rows)}")
    return ok


def test_read_rows_returns_empty_without_layout():
    ok = read_row_states(_synth_panel([("black", "black")] * 7), None) == []
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_returns_empty_without_layout")
    return ok
```

`main()` 의 `results` 에 위 3개를 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'read_row_states'`

- [ ] **Step 3: 최소 구현**

`assist_score.py` 에 추가한다:

```python
def read_row_states(image, layout) -> list:
    """패널 이미지에서 행별 상태를 읽는다. layout 이 없으면 빈 목록.

    image 는 패널 crop 된 PIL Image 다(격자 좌표계와 같아야 한다). 폴링마다 호출되며
    VLM/OCR 을 쓰지 않는다 - 셀 박스는 이미 격자에 있고 필요한 건 색뿐이다.
    """
    if layout is None or image is None:
        return []
    frame = np.array(image.convert("RGB"))
    height, width = frame.shape[:2]

    rows = []
    for row_boxes in layout.grid:
        cells = {}
        for column, box in zip(layout.columns, row_boxes):
            left = max(0, min(width, int(box["left"])))
            right = max(left, min(width, int(box["right"])))
            top = max(0, min(height, int(box["top"])))
            bottom = max(top, min(height, int(box["bottom"])))
            cells[column] = classify_ink(frame[top:bottom, left:right])
        rows.append(RowState(cells=cells))
    return rows
```

`__all__` 에 `"read_row_states"` 를 추가한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 24/24 cases passed`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "feat(workflow_3): 캡처 프레임에서 Assist 행 상태 판독(read_row_states)

폴링마다 도는 경로인데 VLM/OCR 을 쓰지 않는다. 셀 박스는 캐시된 격자에 이미 있고 필요한
건 색뿐이다. 값을 읽지 않으므로 OCR 환각이 판정에 낄 자리가 없다."
```

---

### Task 5: 패널 위치 잡기 (`locate_assist_layout`)

VLM 2단계 로케이터로 패널을 찾고, PaddleOCR spotting 1회로 격자를 만든다. watch 당 1회만 돈다.

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py`

**Interfaces:**
- Consumes: Task 3 `build_score_grid`; 기존 `poc.workflow_3.vlm.ui_venus_mai_locator.{TargetConfig, analyze_window_target}`, `poc.workflow_3.vlm.ocr_spotting.parse_spotting_items`, `poc.workflow_3.vlm.prompts.build_spotting_prompt`, `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient`, `poc.workflow_3.util.crop_image`
- Produces:
  - `assist_panel_target() -> TargetConfig`
  - `locate_assist_layout(window, window_title, backend, image) -> "tuple | None"` — `(panel_box, AssistLayout)` 또는 None

- [ ] **Step 1: 실패하는 테스트 작성**

`test_assist_score.py` 에 추가한다:

```python
from poc.workflow_3.sem_monitor.assist_score import assist_panel_target


def test_panel_target_uses_proven_button_geometry():
    """패널 로케이트도 오피스에서 검증된 2단계 로케이터 기하를 따른다.

    bench_tool_window_reader 가 같은 tool 창에서 acc=1.000 을 낸 설정이다. 여기서 임의로
    다른 값을 쓰면 '입증된 설정' 이라는 근거가 사라진다.
    """
    from poc.workflow_3.rcs import bench_tool_window_reader as bench

    mine = assist_panel_target()
    theirs = bench._button_target("Queue")
    ok = (
        mine.min_crop_width == theirs.min_crop_width
        and mine.vertical_pad_min_px == theirs.vertical_pad_min_px
        and "Assist" in mine.description
    )
    print(f"[{'PASS' if ok else 'FAIL'}] panel_target_uses_proven_button_geometry")
    return ok
```

`main()` 의 `results` 에 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'assist_panel_target'`

- [ ] **Step 3: 최소 구현**

`assist_score.py` 상단 import 에 추가:

```python
from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.util import crop_image
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
from poc.workflow_3.vlm.prompts import build_spotting_prompt
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

LOG_NAME = "assist_score"
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "assist_score"
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"

# 패널 crop 여유 - 로케이터가 준 점 주변을 넉넉히 잘라 표 전체를 담는다.
PANEL_LEFT_RATIO = 0.22
PANEL_RIGHT_RATIO = 0.22
PANEL_TOP_RATIO = 0.14
PANEL_BOTTOM_RATIO = 0.22
```

그리고 함수 두 개를 추가한다:

```python
def assist_panel_target() -> TargetConfig:
    """Assist 패널 grounding 타겟.

    기하는 bench_tool_window_reader._button_target 와 같은 계열이다 - 같은 tool 창에서
    오피스 acc=1.000 이 나온 설정이라 근거 없는 값을 새로 만들지 않는다.
    """
    return TargetConfig(
        key="assist_panel",
        description=(
            "the Recipe Monitor Assist panel inside this CD-SEM tool window - the table "
            "that lists recent measurements with Addressing1 / Addressing2 / Measurement "
            "thumbnails and their score numbers stacked vertically. Return a point at the "
            "centre of that table, not on the live SEM image and not on the button row."
        ),
        left_pad_ratio=0.8,
        right_pad_ratio=0.8,
        vertical_pad_ratio=0.8,
        min_crop_width=320,
        min_crop_height=96,
        vertical_pad_min_px=16,
    )


def locate_assist_layout(window, window_title: str, backend: str, image):
    """Assist 패널을 찾아 score 격자를 만든다. 실패 시 None.

    watch 당 1회만 돈다(이후 폴링은 read_row_states 가 캐시된 격자를 쓴다). 반환은
    (panel_box, AssistLayout) 이며 panel_box 는 창-이미지 좌표계다.
    """
    try:
        result = analyze_window_target(
            window, window_title, backend, assist_panel_target(),
            debug_image_dir=DEBUG_ARTIFACT_DIR,
            log_name=LOG_NAME,
            component_name=LOG_NAME,
            artifact_prefix="assist_panel",
            image=image,
            timeout_sec=15.0,
        )
    except Exception as exc:
        print(f"[WARNING] Assist 패널 grounding 실패: {exc}")
        return None

    point = getattr(result, "point", None)
    if not point:
        print("[WARNING] Assist 패널을 찾지 못함 - 감지 비활성(cap 대기)")
        return None

    width, height = image.size
    panel_box = {
        "left": max(0, int(point["x"] - width * PANEL_LEFT_RATIO)),
        "right": min(width, int(point["x"] + width * PANEL_RIGHT_RATIO)),
        "top": max(0, int(point["y"] - height * PANEL_TOP_RATIO)),
        "bottom": min(height, int(point["y"] + height * PANEL_BOTTOM_RATIO)),
    }
    panel = crop_image(image, panel_box)

    try:
        client = Workflow1VLMClient(OCR_SERVICE_SLUG, timeout_sec=30.0, log_name=LOG_NAME)
        image_b64, _w, _h = encode_image_webp(panel.convert("RGB"), quality=90)
        system_message, user_text = build_spotting_prompt()
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
        )
        items = parse_spotting_items(response.text)
    except Exception as exc:
        print(f"[WARNING] Assist 패널 OCR 실패: {exc}")
        return None

    layout = build_score_grid(items, panel.size)
    if layout is None:
        return None
    print(
        f"[INFO] Assist 격자 확보: panel={panel_box} rows={len(layout.grid)} "
        f"columns={layout.columns}"
    )
    return panel_box, layout
```

`__all__` 에 `"assist_panel_target"`, `"locate_assist_layout"` 를 추가한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 25/25 cases passed`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "feat(workflow_3): Assist 패널 로케이트 + 격자 확보(locate_assist_layout)

watch 당 1회만 도는 경로다. 2단계 로케이터로 패널 중심을 잡고 PaddleOCR spotting 1회로
헤더와 숫자 위치를 읽어 격자를 만든 뒤 캐시한다.

로케이터 기하는 bench_tool_window_reader 와 같은 계열로 맞췄다 - 같은 tool 창에서
acc=1.000 이 나온 설정이라 근거 없는 값을 새로 만들지 않는다."
```

---

### Task 6: 설정 필드 교체

절대값 기준을 없애고 차분/streak 기준을 넣는다.

**Files:**
- Modify: `poc/workflow_3/config.py:82` (`engineer_done_min_count` 정의), `poc/workflow_3/config.py:259` (env 로더)
- Test: `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`

**Interfaces:**
- Produces: `Workflow3Settings.engineer_done_ok_streak: int = 6`, `Workflow3Settings.engineer_done_min_delta: int = 6`; env `ALIGN_FAIL_ENGINEER_DONE_OK_STREAK`, `ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA`. `engineer_done_min_count` 는 제거된다.

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_engineer_done_align_adjustment.py` 에 추가한다:

```python
def test_settings_use_delta_and_streak_not_absolute_count():
    """절대값 기준(min_count)은 제거됐다.

    잔존 카운터 오탐의 근원이었다 - 이전 런의 350/350 이 떠 있으면 즉시 조건을 만족했다.
    """
    settings = load_workflow3_settings()
    ok = (
        settings.engineer_done_ok_streak == 6
        and settings.engineer_done_min_delta == 6
        and not hasattr(settings, "engineer_done_min_count")
    )
    print(f"[{'PASS' if ok else 'FAIL'}] settings_use_delta_and_streak_not_absolute_count")
    return ok
```

기존 파일의 실행 목록(`main()` 또는 `__main__` 블록)에 이 함수를 추가한다. 파일 상단에 `from poc.workflow_3.config import load_workflow3_settings` 가 없으면 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`
Expected: FAIL — `engineer_done_ok_streak` 속성이 없다

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/config.py` 에서 아래 줄을 삭제한다:

```python
    engineer_done_min_count: int = 6  # done(=watch 종료+tool 닫기 트리거) 최소 분자값 — N>5 까지 측정 확인 후 닫기.
```

그 자리에 넣는다:

```python
    # 판정: (watch 시작 이후 새 측정 >= min_delta) and (화면상 최신 연속 정상 >= ok_streak).
    # 앞 조건이 이전 런의 잔존 카운터를 걷어내고, 뒤 조건이 측정 품질을 본다.
    # 절대값 기준(옛 engineer_done_min_count)은 잔존 카운터를 통과시켜 제거했다.
    engineer_done_ok_streak: int = 6   # Assist score 연속 정상(검정) 요구 횟수.
    engineer_done_min_delta: int = 6   # watch 시작 이후 최소 새 측정 횟수.
```

로더에서 아래 줄을 삭제한다:

```python
        engineer_done_min_count=env_int("ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT", 6),
```

그 자리에 넣는다:

```python
        engineer_done_ok_streak=env_int("ALIGN_FAIL_ENGINEER_DONE_OK_STREAK", 6),
        engineer_done_min_delta=env_int("ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA", 6),
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`
Expected: 새 테스트 PASS. 기존 테스트 중 `engineer_done_min_count` 를 참조하는 것이 있으면 이 시점에 실패하므로, 해당 참조를 `engineer_done_min_delta` 로 바꾼다.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/config.py poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
git commit -m "feat(workflow_3)!: engineer-done 설정을 절대값에서 차분+streak 으로 교체

engineer_done_min_count 제거, engineer_done_ok_streak / engineer_done_min_delta 추가.

절대값 기준은 잔존 카운터 오탐의 근원이었다. N 이 낮게 시작해야 한다는 요구가 없어서
이전 런의 350/350 이 떠 있으면 watch 시작 즉시 조건을 만족했다."
```

---

### Task 7: 판정식 교체

`EngineerDoneDetector` 의 판정을 `(delta >= min_delta) and (streak >= ok_streak)` 로 바꾼다. ROI grounding 과 카운터 OCR 경로는 그대로 둔다.

**Files:**
- Modify: `poc/workflow_3/monitor/engineer_done_align_adjustment.py:126-181` (`__call__`), `:102-124` (`__init__`)
- Test: `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`

**Interfaces:**
- Consumes: Task 2 `ok_streak`, Task 6 의 설정 필드
- Produces: `EngineerDoneDetector(..., rows_fn=None)` — `rows_fn() -> list` 주입점 (테스트용, 실배선은 Task 8)

- [ ] **Step 1: 실패하는 테스트 작성**

`test_engineer_done_align_adjustment.py` 에 추가한다:

```python
from poc.workflow_3.sem_monitor.assist_score import RowState


def _rows_all_ok(count=7):
    cells = {"Addressing1": "black", "Addressing2": "blank", "Measurement": "black"}
    return [RowState(cells=dict(cells)) for _ in range(count)]


def _detector_with(counter_values, rows):
    """카운터 값을 순서대로 돌려주는 detector 를 만든다."""
    settings = load_workflow3_settings()
    state = {"i": 0}

    def ocr_fn(_crop):
        idx = min(state["i"], len(counter_values) - 1)
        state["i"] += 1
        return f"{counter_values[idx]}/350"

    detector = EngineerDoneDetector(
        None, settings,
        capture_fn=lambda: Image.new("RGB", (200, 100), (240, 240, 240)),
        ground_fn=lambda _img: (500, 500),
        ocr_fn=ocr_fn,
        rows_fn=lambda: rows,
    )
    return detector


def test_leftover_counter_does_not_fire():
    """watch 시작 시 7행 전부 검정 + 카운터가 안 움직이면 done 이 아니다.

    옛 판정(n >= 6 and n >= _last_n)이 즉시 True 를 내던 바로 그 상황이다.
    """
    detector = _detector_with([350, 350, 350], _rows_all_ok())
    results = [detector() for _ in range(3)]
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] leftover_counter_does_not_fire: {results}")
    return ok


def test_delta_reached_but_streak_short():
    """새 측정 6회를 채워도 연속 정상이 모자라면 done 이 아니다."""
    rows = _rows_all_ok(7)
    rows[-2].cells["Measurement"] = "red"   # 최신에서 두 번째가 실패 -> streak = 1
    detector = _detector_with([10, 20], rows)
    results = [detector(), detector()]
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] delta_reached_but_streak_short: {results}")
    return ok


def test_done_when_delta_and_streak_both_met():
    detector = _detector_with([10, 20], _rows_all_ok())
    first = detector()      # baseline 설정 -> False
    second = detector()     # delta=10, streak=7 -> True
    ok = (first is False) and (second is True)
    print(f"[{'PASS' if ok else 'FAIL'}] done_when_delta_and_streak_both_met: {first},{second}")
    return ok


def test_baseline_cleared_on_relocalize():
    """재grounding 하면 baseline 도 무효화한다.

    옛 구현은 _last_n 만 살려둬서 옛 ROI 값과 새 ROI 값을 비교했다. 같은 실수를 막는다.
    """
    detector = _detector_with([10, 20], _rows_all_ok())
    detector()
    detector._roi_ratios = None       # 재grounding 예약 상태를 흉내낸다
    detector._reset_baseline()
    ok = detector._baseline_n is None
    print(f"[{'PASS' if ok else 'FAIL'}] baseline_cleared_on_relocalize")
    return ok
```

`main()` 의 실행 목록에 위 4개를 추가한다. 파일 상단에 `from PIL import Image` 가 없으면 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`
Expected: FAIL — `EngineerDoneDetector` 가 `rows_fn` 을 받지 않는다 (`TypeError`)

- [ ] **Step 3: 최소 구현**

`engineer_done_align_adjustment.py` 상단 import 에 추가:

```python
from poc.workflow_3.sem_monitor.assist_score import ok_streak
```

`__init__` 시그니처에 `rows_fn=None` 을 추가하고 본문에 넣는다:

```python
        self._rows_fn = rows_fn
        self._baseline_n: int | None = None
```

기존 `self._last_n: int | None = None` 은 삭제한다.

`__call__` 의 판정 부분(기존 `self._ocr_miss_streak = 0` 이후 `return is_done` 까지)을 다음으로 교체한다:

```python
        self._ocr_miss_streak = 0

        if self._baseline_n is None:
            # watch 시작 기준점. 이전 런의 잔존 카운터를 여기서 흡수한다.
            self._baseline_n = n
            self.last_debug["baseline_n"] = n
            return False

        delta = n - self._baseline_n
        rows = self._read_rows()
        streak = ok_streak(rows)
        self.last_debug.update({"n": n, "delta": delta, "streak": streak})

        is_done = (
            delta >= self.s.engineer_done_min_delta
            and streak >= self.s.engineer_done_ok_streak
        )
        if is_done:
            print(
                f"[INFO] 측정 진행 확인: 새 측정 {delta}회, 연속 정상 {streak}회 "
                f"(>= {self.s.engineer_done_min_delta}/{self.s.engineer_done_ok_streak}) "
                f"- align 완료 판정, watch 조기 종료 후 tool 창 닫기 진행"
            )
        return is_done
```

메서드 두 개를 추가한다:

```python
    def _read_rows(self) -> list:
        """Assist 행 상태를 읽는다. 실패는 빈 목록(= streak 0 = 아직 아님)."""
        if self._rows_fn is None:
            return []
        try:
            return self._rows_fn() or []
        except Exception as exc:
            print(f"[WARNING] Assist 행 판독 실패(이번 회차 미판정): {exc}")
            return []

    def _reset_baseline(self) -> None:
        """baseline 을 무효화한다. ROI 가 바뀌면 반드시 함께 호출한다.

        옛 구현은 재grounding 때 _last_n 만 살려둬서 옛 ROI 값과 새 ROI 값을 비교했다.
        폴링 간 상태를 baseline 하나로 줄이고, 그 하나를 여기서 확실히 지운다.
        """
        self._baseline_n = None
```

재grounding 분기(기존 `self._roi_ratios = None` 을 세팅하는 곳, 대략 160행)에 `self._reset_baseline()` 을 함께 호출하도록 추가한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`
Expected: 새 4개 PASS. `engineer_done_min_count` / `_last_n` 을 참조하던 기존 테스트가 실패하면 새 의미(`_baseline_n`, `engineer_done_min_delta`)로 고친다.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/engineer_done_align_adjustment.py poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
git commit -m "feat(workflow_3)!: engineer-done 판정을 차분+Assist streak 으로 교체

옛 판정 'n >= 6 and n >= _last_n' 을 '(delta >= 6) and (streak >= 6)' 으로 바꾼다.
delta 는 watch 시작 이후 새 측정 횟수, streak 은 Assist 화면상 최신 연속 정상 개수다.
앞이 잔존 카운터를 걷어내고 뒤가 품질을 본다.

streak 은 매 폴링 화면에서 새로 계산되므로 폴링 간 누적 상태가 없다. 남는 상태는
_baseline_n 하나뿐이고 재grounding 시 _reset_baseline 이 지운다 - 옛 _last_n 이
재grounding 을 건너 살아남아 옛 ROI 값과 새 ROI 값을 비교하던 버그의 구조적 해법이다."
```

---

### Task 8: 배선과 디버그 오버레이

`build_engineer_done_detector` 가 실제 `rows_fn` 을 만들어 넣고, 판정이 바뀔 때만 오버레이 1장을 남긴다.

**Files:**
- Modify: `poc/workflow_3/monitor/engineer_done_align_adjustment.py:302-320` (`build_engineer_done_detector`)
- Modify: `poc/workflow_3/sem_monitor/assist_score.py` (오버레이 함수 추가)
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: Task 4 `read_row_states`, Task 5 `locate_assist_layout`
- Produces: `save_assist_overlay(image, layout, rows, out_path) -> None`;
  `engineer_done_align_adjustment._make_rows_fn(tool_window, *, debug_dir=None) -> callable`;
  모듈 상수 `ALL_BLANK_RELOCATE_AFTER = 3` (`engineer_done_align_adjustment.py` 최상단)

- [ ] **Step 1: 실패하는 테스트 작성**

`test_assist_score.py` 에 추가한다 (상단에 `import tempfile`, `from pathlib import Path` 추가):

```python
def test_overlay_writes_a_file():
    """오버레이는 오피스가 행 방향/열 매핑/색 임계를 한 장으로 검증하는 수단이다."""
    layout = _layout_for_synth()
    image = _synth_panel([("black", "black")] * 7)
    rows = read_row_states(image, layout)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "overlay.jpg"
        save_assist_overlay(image, layout, rows, out)
        ok = out.exists() and out.stat().st_size > 0
    print(f"[{'PASS' if ok else 'FAIL'}] overlay_writes_a_file")
    return ok
```

import 에 `save_assist_overlay` 를 추가하고 `main()` 목록에도 넣는다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `ImportError: cannot import name 'save_assist_overlay'`

- [ ] **Step 3: 최소 구현**

`assist_score.py` 에 추가한다 (상단 import 에 `from poc.workflow_3.debug_artifacts import save_debug_jpeg` 와 `from PIL import ImageDraw` 추가):

```python
_VERDICT_COLORS = {
    "ok": (0, 200, 0),
    "fail": (255, 0, 0),
    "pending": (128, 128, 128),
    "unknown": (255, 160, 0),
}


def save_assist_overlay(image, layout, rows: list, out_path) -> None:
    """판독 결과를 패널 이미지 위에 그려 저장한다 (실패 무시).

    오피스가 행 방향(최신이 아래인지) / 열 매핑 / 색 임계를 한 장으로 검증할 수 있게
    한다. 폴링마다가 아니라 판정이 바뀔 때만 부른다.
    """
    try:
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for row_idx, row_boxes in enumerate(layout.grid):
            verdict = rows[row_idx].verdict if row_idx < len(rows) else "unknown"
            color = _VERDICT_COLORS.get(verdict, (255, 160, 0))
            for box in row_boxes:
                draw.rectangle(
                    [box["left"], box["top"], box["right"], box["bottom"]],
                    outline=color, width=2,
                )
            label = f"{row_idx}:{verdict}"
            draw.text((row_boxes[0]["left"], max(0, row_boxes[0]["top"] - 12)), label, fill=color)
        save_debug_jpeg(canvas, out_path)
    except Exception as exc:
        print(f"[WARNING] Assist 오버레이 저장 실패: {exc}")
```

`__all__` 에 `"save_assist_overlay"` 를 추가한다.

`engineer_done_align_adjustment.py` 최상단(`_INT_RE` 정의 근처)에 상수를 추가한다:

```python
# Assist 전 행이 연속으로 이만큼 빈칸이면 패널이 이동한 것으로 보고 격자를 다시 잡는다.
ALL_BLANK_RELOCATE_AFTER = 3
```

이어서 `build_engineer_done_detector` 를 고친다. `ocr_fn = _make_ocr_fn(settings)` 다음에 넣는다:

```python
    rows_fn = _make_rows_fn(tool_window, debug_dir=debug_dir)
```

그리고 `EngineerDoneDetector(...)` 호출에 `rows_fn=rows_fn` 을 추가한다. 같은 파일에 헬퍼를 추가한다:

```python
def _make_rows_fn(tool_window, *, debug_dir=None):
    """Assist 행 판독 클로저. 격자는 첫 성공 때 1회만 만들고 캐시한다.

    로케이트에 실패하면 None 을 캐시하지 않고 매번 재시도하되, 실패 로그는 1회만 낸다
    (watch 내내 같은 경고가 반복되면 콘솔이 쓸모없어진다).
    """
    from poc.workflow_3.sem_monitor.assist_score import (
        locate_assist_layout,
        read_row_states,
        save_assist_overlay,
    )
    from poc.workflow_3.util import capture_window, crop_image

    state = {"panel_box": None, "layout": None, "warned": False, "last_verdicts": None,
             "seq": 0, "all_blank_streak": 0}

    def rows_fn():
        image = capture_window(tool_window)
        if state["layout"] is None:
            # window_title/backend 는 빈 문자열로 넘긴다. image 를 함께 주면
            # analyze_window_target 이 창 활성화/재캡처를 건너뛰므로 쓰이지 않는다.
            located = locate_assist_layout(tool_window, "", "", image)
            if located is None:
                if not state["warned"]:
                    print("[WARNING] Assist 격자 확보 실패 - 이번 watch 는 done 판정 없이 cap 대기")
                    state["warned"] = True
                return []
            state["panel_box"], state["layout"] = located
            state["all_blank_streak"] = 0

        panel = crop_image(image, state["panel_box"])
        rows = read_row_states(panel, state["layout"])

        # 패널이 이동/리사이즈되면 빈 영역을 샘플링해 모든 행이 pending 으로 나온다.
        # 실제로 전 행이 비는 일은 거의 없으므로, 연속으로 그러면 격자를 버리고 다시 잡는다.
        if rows and all(row.verdict == "pending" for row in rows):
            state["all_blank_streak"] += 1
            if state["all_blank_streak"] >= ALL_BLANK_RELOCATE_AFTER:
                print("[INFO] Assist 전 행이 계속 빈칸 - 패널 이동 가능성, 격자 재확보 예약")
                state["layout"] = None
                state["panel_box"] = None
                state["all_blank_streak"] = 0
                return []
        else:
            state["all_blank_streak"] = 0

        verdicts = [row.verdict for row in rows]
        if debug_dir is not None and verdicts != state["last_verdicts"]:
            state["seq"] += 1
            save_assist_overlay(
                panel, state["layout"], rows,
                debug_dir / f"assist_{state['seq']:03d}.jpg",
            )
            state["last_verdicts"] = verdicts
        return rows

    return rows_fn
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인**

Run: `uv run python poc/workflow_3/sem_monitor/test_assist_score.py`
Expected: `[INFO] 26/26 cases passed`

Run: `uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`
Expected: 전부 PASS

- [ ] **Step 5: 전체 회귀**

```bash
for t in $(find poc/workflow_3 poc/workflow_3e -name "test_*.py" | sort); do
  uv run python "$t" >/dev/null 2>&1 || echo "RED: $t"
done
uv run pytest -q poc/workflow_3/recording_filter
```

Expected: `RED:` 는 `poc/workflow_3/align/diagnostics/test_match_on_captured_frames.py` 하나만 (오피스 캡처 픽스처 필요, 기존부터 그러함). pytest 는 전부 통과.

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py poc/workflow_3/monitor/engineer_done_align_adjustment.py
git commit -m "feat(workflow_3): Assist 판독 배선 + 판정 변화 시 오버레이

build_engineer_done_detector 가 rows_fn 을 만들어 넣는다. 격자는 첫 성공 때 1회만 만들고
캐시하며, 로케이트 실패 경고는 1회만 낸다(watch 내내 반복되면 콘솔이 쓸모없어진다).

오버레이는 폴링마다가 아니라 판정이 바뀔 때만 남긴다. 오피스가 행 방향(최신이 아래인지)
열 매핑, 색 임계를 한 장으로 검증할 수 있다."
```

---

## 오피스 검증 (구현 후)

`engineer_done_detect_enabled` 는 기본 False 다. 오피스에서 켜고 확인할 것:

1. `ALIGN_FAIL_ENGINEER_DONE_DETECT=1` 로 align_fail_monitor 실행, 실알람 1건.
2. `debug_images/engineer_done/<eqp>_<tag>/assist_*.jpg` 오버레이 확인:
   - 행 번호 0 이 **맨 위**(가장 오래된 행)에 붙었는가 → 아니면 `ASSIST_NEWEST_ROW_AT = "top"`
   - 열 박스가 Addressing1 / Addressing2 / Measurement 에 각각 맞는가
   - 빨간 측정이 `fail`, 빈칸이 `pending` 으로 찍히는가
3. 콘솔의 `새 측정 N회, 연속 정상 M회` 값이 화면과 맞는가.
