# EngineerDoneDetector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recipe Monitor 측정 카운터 분자(N/M 의 N)로 "측정 시작=엔지니어 align 보정 완료"를 감지해, 미보정 engineer watch 녹화를 고정 timeout 대신 조기 종료한다. + 측정 중인 tool 에 바로 접속해 전 체인을 즉시 검증하는 오피스 캘리브레이션 단독 실행 모드.

**Architecture:** hybrid 파이프라인 — VLM grounding 1회(분자 위치, tool-window 상대 ROI 캐시) → CV 변화감지 gate(정적이면 OCR 0회) → 변화 시에만 OCR confirm(연속 2회 읽기로 done 판정). 모든 실패는 False 로 degrade(기존 `engineer_watch_sec` cap 이 안전망). spec: `poc/workflow_2/docs/superpowers/specs/2026-06-11-multi-tool-serial-queue-design.md` §4-B/4-C/4-D/4-E.

**Tech Stack:** Python 3.10+/uv, PIL+numpy(CV gate, `recording.py` 로직 재사용), `Workflow1VLMClient`(ui-venus grounding / paddleocr-vl OCR), pywinauto 창 캡처(`util.capture_window`), print-기반 테스트 스크립트(레포 규약).

**spec 대비 의도적 강화 1건:** done 판정을 "N ≥ min_count 且 비감소(첫 읽기 허용)" 대신 **"N ≥ min_count 且 직전 읽기 존재 且 N ≥ 직전"**(연속 2회 확인)으로 한다. 단일 프레임 OCR 오독(예: 분모 350 을 한 번 잡음)으로 녹화가 끊기는 것을 막는다. 비용은 poll 1회(~8s) 지연.

**레포 규약 주의(모든 task 공통):**
- 한국어 docstring, print 로깅(`[INFO]`/`[WARNING]`/`[ERROR]`), `logging` 모듈 금지.
- print() 문자열에 em-dash(U+2014)·이모지 금지(office cp949). docstring 은 허용.
- `from __future__` import 금지. CLI 인자(argparse) 금지 — env/모듈 상수만.
- 절대 import(`from poc.workflow_3...`).
- **commit 은 명시된 파일만 `git add`** — 작업트리에 무관한 수정(`vision/align_fail_correct.py`, `vision/test_align_fail_correct.py`)이 있으므로 절대 `git add -A` 금지.
- main 직접 commit(브랜치 안 만듦).

---

## File Structure

| 파일 | 상태 | 책임 |
|---|---|---|
| `poc/workflow_3/config.py` | 수정 | `Workflow3Settings` engineer_done_* 필드 8개 + env 배선 |
| `poc/workflow_3/vlm/prompts/prompt_recipe_monitor_counter.py` | 신규 | 분자 grounding 프롬프트(ui-venus 공식 단일요소 형식 재사용) |
| `poc/workflow_3/monitor/engineer_done.py` | 신규 | 순수 헬퍼(parse/roi/extract) + `EngineerDoneDetector` + `build_engineer_done_detector` + 캘리브레이션 `main()` |
| `poc/workflow_3/monitor/test_engineer_done.py` | 신규 | Mac 합성 테스트(주입 seam, RCS/VLM 불요) |
| `poc/workflow_3/monitor/cycle.py` | 수정 | `_engineer_watch` done-detector 조기종료 + watch 호출부 + `context["vlm_client"]` |
| `poc/workflow_3/README.md` | 수정 | env 표에 engineer_done_* 추가 |

---

### Task 1: `Workflow3Settings` engineer_done_* 필드 + env 배선

**Files:**
- Modify: `poc/workflow_3/config.py`
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (신규 생성, 이후 task 들이 누적)

- [ ] **Step 1: Write the failing test**

`poc/workflow_3/monitor/test_engineer_done.py` 신규 생성:

```python
"""engineer_done 감지기 합성 테스트 (Mac/dev, RCS·VLM 불요).

`uv run python poc/workflow_3/monitor/test_engineer_done.py` 로 실행한다.
"""

import sys

from poc.workflow_3.config import Workflow3Settings


def _check(name: str, condition: bool) -> bool:
    """단건 검증 결과를 출력하고 통과 여부를 반환한다."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def test_settings_defaults() -> bool:
    """engineer_done_* 필드가 기본값과 함께 존재한다 (기본 비활성)."""
    s = Workflow3Settings()
    ok = True
    ok &= _check("detect_enabled default False", s.engineer_done_detect_enabled is False)
    ok &= _check("poll_sec default 8.0", s.engineer_done_poll_sec == 8.0)
    ok &= _check("min_count default 2", s.engineer_done_min_count == 2)
    ok &= _check("change_min_px default 4", s.engineer_done_change_min_px == 4)
    ok &= _check("relocalize_after_miss default 3", s.engineer_done_relocalize_after_miss == 3)
    ok &= _check("roi_pad_x default 0.03", s.engineer_done_roi_pad_x == 0.03)
    ok &= _check("roi_pad_y default 0.02", s.engineer_done_roi_pad_y == 0.02)
    ok &= _check("vlm_service default ui-venus", s.engineer_done_vlm_service == "ui-venus-1.5-8b")
    ok &= _check("ocr_service default paddleocr", s.engineer_done_ocr_service == "paddleocr-vl-1.5")
    return ok


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_settings_defaults,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `AttributeError: 'Workflow3Settings' object has no attribute 'engineer_done_detect_enabled'` (또는 FAIL 출력)

- [ ] **Step 3: Add settings fields + env wiring**

`poc/workflow_3/config.py` 의 `Workflow3Settings` 에서 `# --- 모호 키 재등록 알림 ---` 블록 **앞에** 추가:

```python
    # --- engineer watch 측정-시작 감지 (Recipe Monitor 카운터) ---
    # 미보정 watch 중 측정 카운터 분자(N/M 의 N)가 증가하면 align 완료로 보고
    # 녹화를 조기 종료한다. VLM grounding 1회 + CV gate + OCR confirm(연속 2회).
    engineer_done_detect_enabled: bool = False  # 오피스 캘리브레이션 검증 전 기본 off.
    engineer_done_poll_sec: float = 8.0  # watch 안 detector 호출 간격.
    engineer_done_min_count: int = 2  # done 으로 보는 최소 분자값.
    engineer_done_change_min_px: int = 4  # CV gate 변화 픽셀 임계(다운샘플).
    engineer_done_relocalize_after_miss: int = 3  # 변화 후 OCR 연속 미검출 시 재grounding.
    engineer_done_roi_pad_x: float = 0.03  # grounding 점 -> crop 확장 비율(가로, 창 대비).
    engineer_done_roi_pad_y: float = 0.02  # grounding 점 -> crop 확장 비율(세로, 창 대비).
    engineer_done_vlm_service: str = "ui-venus-1.5-8b"  # grounding 서비스 slug.
    engineer_done_ocr_service: str = "paddleocr-vl-1.5"  # 분자 OCR 서비스 slug.
```

`load_workflow3_settings()` 의 `reregister_second_ratio_threshold=` 줄 **앞에** 추가:

```python
        engineer_done_detect_enabled=env_flag("ALIGN_FAIL_ENGINEER_DONE_DETECT", default=False),
        engineer_done_poll_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_POLL_SEC", 8.0),
        engineer_done_min_count=env_int("ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT", 2),
        engineer_done_change_min_px=env_int("ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX", 4),
        engineer_done_relocalize_after_miss=env_int("ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS", 3),
        engineer_done_roi_pad_x=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_X", 0.03),
        engineer_done_roi_pad_y=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_Y", 0.02),
        engineer_done_vlm_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE", "ui-venus-1.5-8b"),
        engineer_done_ocr_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE", "paddleocr-vl-1.5"),
```

추가로 `engineer_watch_sec: float = 600.0` 줄의 의미가 바뀌므로 주석을 갱신:

```python
    engineer_watch_sec: float = 600.0  # 미보정 watch 상한(cap) — done 감지 시 조기 종료.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `[PASS]` x9, `engineer_done 테스트: 1/1 통과`, exit 0

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): Workflow3Settings 측정-시작 감지 필드 + env 배선"
```

---

### Task 2: 분자 grounding 프롬프트 빌더

**Files:**
- Create: `poc/workflow_3/vlm/prompts/prompt_recipe_monitor_counter.py`
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (추가)

- [ ] **Step 1: Write the failing test**

`test_engineer_done.py` 에 import 와 테스트 추가, `main()` 의 `tests` 리스트에 `test_counter_prompt` 추가:

```python
from poc.workflow_3.vlm.prompts.prompt_recipe_monitor_counter import (
    RECIPE_MONITOR_NUMERATOR_INSTRUCTION,
    build_recipe_monitor_counter_prompt,
)


def test_counter_prompt() -> bool:
    """ui-venus 공식 단일요소 형식([x,y], [-1,-1] 거부)을 따른다."""
    system_message, user_text = build_recipe_monitor_counter_prompt()
    ok = True
    ok &= _check("system empty (official format)", system_message == "")
    ok &= _check("instruction embedded", RECIPE_MONITOR_NUMERATOR_INSTRUCTION in user_text)
    ok &= _check("point format requested", "[x,y]" in user_text)
    ok &= _check("refusal format requested", "[-1,-1]" in user_text)
    return ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `ModuleNotFoundError: ... prompt_recipe_monitor_counter`

- [ ] **Step 3: Write the prompt builder**

`poc/workflow_3/vlm/prompts/prompt_recipe_monitor_counter.py` 신규:

```python
"""Recipe Monitor 측정 카운터(분자) grounding 프롬프트 빌더.

tool 창의 Recipe Monitor 패널에서 측정 진행 카운터(예: 2/350)의 분자 위치를
ui-venus 공식 단일 요소 grounding 형식으로 요청한다. 창이 드래그로 움직일 수
있어 고정 ROI 가 불가하므로, 사이클당 1회 이 grounding 으로 위치를 캐시한다.
"""

from poc.workflow_3.vlm.prompts.prompt_login_rcs_ui_venus import (
    UI_VENUS_OFFICIAL_PROMPT_TEMPLATE,
)

# 오피스 캘리브레이션에서 문구를 조정할 수 있도록 instruction 을 상수로 분리한다.
# 첫-글자 anchoring 원칙: 'Recipe Monitor' 텍스트를 먼저 찾게 한 뒤 행 -> 카운터 순.
RECIPE_MONITOR_NUMERATOR_INSTRUCTION = (
    "Find the visible text 'Recipe Monitor' first, then inside that Recipe Monitor "
    "area find the row showing Port, Slot and Recipe. In that row, locate the "
    "measurement progress counter that looks like 'N/M' (for example '2/350'). "
    "Output the center point of the numerator N only (the integer BEFORE the slash '/')"
)


def build_recipe_monitor_counter_prompt() -> tuple[str, str]:
    """ui-venus 공식 단일 요소 형식으로 분자 중심점을 요청한다."""
    user_text = UI_VENUS_OFFICIAL_PROMPT_TEMPLATE.format(
        instruction=RECIPE_MONITOR_NUMERATOR_INSTRUCTION
    )
    return "", user_text


__all__ = [
    "RECIPE_MONITOR_NUMERATOR_INSTRUCTION",
    "build_recipe_monitor_counter_prompt",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `2/2 통과`, exit 0

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/vlm/prompts/prompt_recipe_monitor_counter.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): Recipe Monitor 분자 grounding 프롬프트"
```

---

### Task 3: 순수 헬퍼 — `parse_point_1000` / `point_to_roi_ratios` / `extract_numerator`

**Files:**
- Create: `poc/workflow_3/monitor/engineer_done.py` (헬퍼부터; detector 는 Task 4)
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (추가)

- [ ] **Step 1: Write the failing tests**

`test_engineer_done.py` 에 추가 (`tests` 리스트에도 3개 추가):

```python
from poc.workflow_3.monitor.engineer_done import (
    extract_numerator,
    parse_point_1000,
    point_to_roi_ratios,
)


def test_parse_point_1000() -> bool:
    """ui-venus [x,y] 응답 파싱 — 거부/범위밖/없음은 None."""
    ok = True
    ok &= _check("valid point", parse_point_1000("[525, 550]") == (525, 550))
    ok &= _check("point in prose", parse_point_1000("the point is [10,20].") == (10, 20))
    ok &= _check("refusal -> None", parse_point_1000("[-1,-1]") is None)
    ok &= _check("out of range -> None", parse_point_1000("[1500, 200]") is None)
    ok &= _check("no point -> None", parse_point_1000("cannot find it") is None)
    ok &= _check("empty -> None", parse_point_1000("") is None)
    return ok


def test_point_to_roi_ratios() -> bool:
    """grounding 점(0-1000) -> 상대비율 ROI 확장 + 경계 clamp."""
    ok = True
    roi = point_to_roi_ratios(500, 500, 0.05, 0.05)
    ok &= _check("center roi", roi == (0.45, 0.45, 0.55, 0.55))
    roi = point_to_roi_ratios(0, 0, 0.05, 0.05)
    ok &= _check("corner clamped", roi is not None and roi[0] == 0.0 and roi[1] == 0.0)
    ok &= _check("corner still has span", roi is not None and roi[2] > 0.0 and roi[3] > 0.0)
    return ok


def test_extract_numerator() -> bool:
    """OCR 텍스트에서 분자 정수 추출 (첫 연속 숫자열)."""
    ok = True
    ok &= _check("'2/350' -> 2", extract_numerator("2/350") == 2)
    ok &= _check("' 13 / 350 ' -> 13", extract_numerator(" 13 / 350 ") == 13)
    ok &= _check("bare '7' -> 7", extract_numerator("7") == 7)
    ok &= _check("no digits -> None", extract_numerator("abc") is None)
    ok &= _check("empty -> None", extract_numerator("") is None)
    return ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `ModuleNotFoundError: ... engineer_done`

- [ ] **Step 3: Write the helpers**

`poc/workflow_3/monitor/engineer_done.py` 신규 (모듈 docstring + 헬퍼만; detector 는 Task 4 에서 같은 파일에 추가):

```python
"""엔지니어 수동 align 보정 완료 감지 — Recipe Monitor 측정 카운터 기반.

미보정 engineer watch 동안 "측정이 시작됐다"(= align 완료)를 감지해 녹화를 조기
종료한다. 신호는 tool 창 Recipe Monitor 의 측정 점 카운터 분자(N/M 의 N)가
증가하는 것 (1/350 → 2/350 → ...).

hybrid 파이프라인 (사이클당 VLM 1회):
  1. grounding(1회 캐시): VLM(ui-venus)으로 분자 위치를 찾아 tool-window
     상대비율 ROI 로 캐시한다 (tool 마다/드래그로 위치가 달라 고정 ROI 불가).
  2. CV gate(매 호출): ROI crop 변화감지 — align-fix 중엔 카운터가 정적이라
     OCR 호출이 0회로 유지된다 (recording.py 의 다운샘플+delta 로직 재사용).
  3. OCR confirm(변화 시에만): paddleocr 로 분자 N 을 읽어
     N >= min_count 이고 직전 읽기 대비 비감소(연속 2회 확인)면 done.
     연속 2회 확인은 단일 프레임 OCR 오독(분모 등)으로 끊기는 것을 막는다.

실패는 전부 graceful: grounding 거부/OCR 실패/예외 → False → watch 의
engineer_watch_sec cap 이 안전망. (CLAUDE.md 규칙: VLM 은 위치만, 전이 판정의
정량 근거는 CV 변화 + OCR 숫자.)

오피스 캘리브레이션 (단독 실행):
  지금 측정 중인 tool 의 Remote Monitoring 창을 열어 두고 실행하면 — 측정
  중에는 분자가 실제로 증가하므로 — grounding/gate/OCR 전 체인을 align fail
  없이 즉시 검증할 수 있다.

  uv run python poc/workflow_3/monitor/engineer_done.py
"""

import re

_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_INT_RE = re.compile(r"\d+")


def parse_point_1000(text: str) -> tuple[int, int] | None:
    """ui-venus 응답에서 첫 [x,y](0-1000)를 파싱한다.

    거부([-1,-1])·범위 밖·미검출은 None.
    """
    match = _POINT_RE.search(text or "")
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def point_to_roi_ratios(
    x_1000: int, y_1000: int, pad_x: float, pad_y: float
) -> tuple[float, float, float, float] | None:
    """grounding 점(0-1000)을 분자 crop 용 상대비율 ROI (l,t,r,b) 로 확장한다."""
    cx, cy = x_1000 / 1000.0, y_1000 / 1000.0
    left = min(max(cx - pad_x, 0.0), 1.0)
    right = min(max(cx + pad_x, 0.0), 1.0)
    top = min(max(cy - pad_y, 0.0), 1.0)
    bottom = min(max(cy + pad_y, 0.0), 1.0)
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None
    return left, top, right, bottom


def extract_numerator(text: str) -> int | None:
    """OCR 텍스트에서 분자 정수를 뽑는다 ('2/350' → 2, 첫 연속 숫자열)."""
    match = _INT_RE.search(text or "")
    if match is None:
        return None
    return int(match.group(0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `5/5 통과`, exit 0

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/engineer_done.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): 점/ROI/분자 순수 헬퍼"
```

---

### Task 4: `EngineerDoneDetector` (주입 seam + 상태 기계)

**Files:**
- Modify: `poc/workflow_3/monitor/engineer_done.py` (detector 클래스 추가)
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (추가)

- [ ] **Step 1: Write the failing tests**

`test_engineer_done.py` 상단 import 에 추가:

```python
import numpy as np
from PIL import Image

from poc.workflow_3.monitor.engineer_done import EngineerDoneDetector
```

테스트용 fixture 헬퍼 + 테스트 5개 추가 (`tests` 리스트에도 추가):

```python
def _frame(counter_value: int) -> Image.Image:
    """카운터 영역 픽셀이 counter_value 에 따라 달라지는 합성 tool 창 프레임.

    창 400x200. 카운터 셀은 x 190..230, y 100..120 부근 — grounding 점
    (525, 550) + pad (0.05, 0.05) 의 ROI 와 일치시킨다.
    """
    arr = np.zeros((200, 400, 3), dtype=np.uint8)
    arr[100:120, 190:190 + 4 * (counter_value + 1)] = 255
    return Image.fromarray(arr)


class _SeqCapture:
    """호출마다 프레임 시퀀스를 차례로 반환한다 (끝나면 마지막 프레임 반복)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def __call__(self):
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame


class _CountingFn:
    """반환값 시퀀스를 차례로 내놓으며 호출 횟수를 기록한다."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, *args):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def _settings():
    """테스트용 설정 — ROI pad 를 합성 프레임 카운터 셀에 맞춘다."""
    return Workflow3Settings(
        engineer_done_detect_enabled=True,
        engineer_done_roi_pad_x=0.05,
        engineer_done_roi_pad_y=0.05,
        engineer_done_min_count=2,
        engineer_done_relocalize_after_miss=3,
    )


def test_detector_static_no_ocr() -> bool:
    """정적 프레임(첫 샘플 포함)에서는 OCR 을 호출하지 않는다."""
    # grounding 1회 캡처 + 정적 crop 3회.
    capture = _SeqCapture([_frame(1), _frame(1), _frame(1), _frame(1)])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("all False on static", results == [False, False, False])
    ok &= _check("ground called once", ground.calls == 1)
    ok &= _check("ocr never called", ocr.calls == 0)
    return ok


def test_detector_two_read_confirm() -> bool:
    """변화 + OCR 2 -> 3: 첫 읽기는 확인 대기(False), 두 번째에 done."""
    capture = _SeqCapture([
        _frame(1),            # grounding 캡처
        _frame(1),            # baseline (첫 샘플, OCR 안 함)
        _frame(2),            # 변화 1 -> OCR '2' (last 없음 -> 확인 대기)
        _frame(3),            # 변화 2 -> OCR '3' (3>=2, 3>=2 -> done)
    ])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350", "3/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("baseline False", results[0] is False)
    ok &= _check("first read waits", results[1] is False)
    ok &= _check("second read done", results[2] is True)
    ok &= _check("ocr called twice", ocr.calls == 2)
    return ok


def test_detector_below_min_not_done() -> bool:
    """N < min_count 면 변화가 있어도 done 아님."""
    capture = _SeqCapture([_frame(0), _frame(0), _frame(1), _frame(2)])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["1/350", "1/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    return _check("below min stays False", results == [False, False, False])


def test_detector_ground_refusal() -> bool:
    """grounding 거부(None) -> 항상 False, 재시도 안 함."""
    capture = _SeqCapture([_frame(1), _frame(2), _frame(3)])
    ground = _CountingFn([None])
    ocr = _CountingFn(["2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("refusal -> all False", results == [False, False, False])
    ok &= _check("ground called once only", ground.calls == 1)
    ok &= _check("ocr never called", ocr.calls == 0)
    return ok


def test_detector_relocalize_after_miss() -> bool:
    """변화 후 OCR 연속 미검출이 임계에 닿으면 1회 재grounding 한다."""
    # 매 호출 프레임이 달라(계속 변화) OCR 이 그때마다 불리지만 빈 텍스트.
    capture = _SeqCapture([
        _frame(0),                       # grounding 1 캡처
        _frame(0), _frame(1), _frame(2), _frame(3),  # baseline + 변화 3회 (miss 3)
        _frame(4),                       # 재grounding 캡처
        _frame(4), _frame(5),            # 새 baseline + 변화
    ])
    ground = _CountingFn([(525, 550), (525, 550)])
    ocr = _CountingFn(["", "", "", "2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    for _ in range(6):
        detector()
    return _check("ground called twice (relocalize)", ground.calls == 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `ImportError: cannot import name 'EngineerDoneDetector'`

- [ ] **Step 3: Implement the detector**

`engineer_done.py` 의 import 를 다음으로 갱신하고:

```python
import re

from poc.workflow_3.monitor.recording import _frame_changed, _to_diff_gray
from poc.workflow_3.util import capture_window
```

`extract_numerator` 아래에 detector 클래스 추가:

```python
class EngineerDoneDetector:
    """Recipe Monitor 분자 기반 측정-시작 감지기 (watch iteration 마다 호출).

    capture_fn/ground_fn/ocr_fn 은 테스트 주입점 (RecordingSession 의 capture_fn
    패턴). 실배선은 build_engineer_done_detector 가 담당한다.

      capture_fn() -> PIL.Image          (기본: util.capture_window(tool_window))
      ground_fn(image) -> (x,y) 0-1000 | None   (VLM grounding, 거부 시 None)
      ocr_fn(crop_image) -> str          (분자 crop OCR 텍스트)
    """

    def __init__(
        self,
        tool_window,
        settings,
        *,
        capture_fn=None,
        ground_fn=None,
        ocr_fn=None,
        debug_dir=None,
    ):
        self.tool_window = tool_window
        self.s = settings
        self._capture_fn = capture_fn or (lambda: capture_window(self.tool_window))
        self._ground_fn = ground_fn
        self._ocr_fn = ocr_fn
        self.debug_dir = debug_dir
        self._roi_ratios: tuple[float, float, float, float] | None = None
        self._localize_tried = False
        self._prev_gray = None
        self._last_n: int | None = None
        self._ocr_miss_streak = 0
        self._debug_seq = 0
        self.last_debug: dict = {}

    def __call__(self) -> bool:
        """측정 시작이 확인되면 True. 모든 실패/미확정은 False (cap 이 안전망)."""
        self.last_debug = {}
        if self._roi_ratios is None and not self._localize_tried:
            self._localize_tried = True
            self._roi_ratios = self._localize()
        if self._roi_ratios is None:
            return False

        crop = self._crop_numerator()
        if crop is None:
            return False

        gray = _to_diff_gray(crop)
        first_sample = self._prev_gray is None
        changed = (not first_sample) and _frame_changed(
            self._prev_gray, gray, self.s.engineer_done_change_min_px
        )
        self._prev_gray = gray
        self.last_debug.update({"changed": changed, "first_sample": first_sample})
        if not changed:
            return False

        self._save_debug_crop(crop)
        n = self._read_numerator(crop)
        if n is None:
            self._ocr_miss_streak += 1
            self.last_debug["ocr_miss_streak"] = self._ocr_miss_streak
            if self._ocr_miss_streak >= self.s.engineer_done_relocalize_after_miss:
                print("[INFO] OCR 연속 미검출 - ROI 재grounding 예약(패널 이동 가능성)")
                self._roi_ratios = None
                self._localize_tried = False
                self._ocr_miss_streak = 0
                self._prev_gray = None
            return False

        self._ocr_miss_streak = 0
        # 연속 2회 확인: 직전 읽기가 있어야 하고 비감소 + min_count 이상.
        is_done = (
            n >= self.s.engineer_done_min_count
            and self._last_n is not None
            and n >= self._last_n
        )
        self.last_debug["n"] = n
        self._last_n = n
        if is_done:
            print(
                f"[INFO] 측정 카운터 확인: N={n} "
                f"(>= {self.s.engineer_done_min_count}, 연속 2회) - 측정 시작 판정"
            )
        return is_done

    # ---- 내부 ----

    def _localize(self):
        """VLM grounding 1회 — 분자 위치를 상대비율 ROI 로. 실패/거부 시 None."""
        if self._ground_fn is None:
            print("[WARNING] engineer-done grounding fn 없음 - 감지 비활성(cap 대기)")
            return None
        try:
            image = self._capture_fn()
            point = self._ground_fn(image)
        except Exception as exc:
            print(f"[WARNING] engineer-done grounding 실패: {exc}")
            return None
        if point is None:
            print("[INFO] engineer-done grounding 거부/미발견 - 감지 비활성(cap 대기)")
            return None
        ratios = point_to_roi_ratios(
            point[0], point[1],
            self.s.engineer_done_roi_pad_x, self.s.engineer_done_roi_pad_y,
        )
        if ratios is None:
            print(f"[WARNING] engineer-done ROI 생성 실패: point={point}")
            return None
        print(
            f"[INFO] engineer-done ROI 캐시: point={point}, "
            f"ratios=({ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f},{ratios[3]:.3f})"
        )
        return ratios

    def _crop_numerator(self):
        """tool 창 재캡처 후 캐시된 상대비율 ROI 로 분자 셀을 crop 한다."""
        try:
            image = self._capture_fn()
        except Exception as exc:
            print(f"[WARNING] engineer-done 캡처 실패(회차 skip): {exc}")
            return None
        left, top, right, bottom = self._roi_ratios
        width, height = image.size
        box = (
            int(left * width),
            int(top * height),
            max(int(left * width) + 1, int(right * width)),
            max(int(top * height) + 1, int(bottom * height)),
        )
        return image.crop(box)

    def _read_numerator(self, crop) -> int | None:
        """분자 crop 을 OCR 해 정수 N 을 얻는다. 실패는 None."""
        if self._ocr_fn is None:
            return None
        try:
            text = self._ocr_fn(crop)
        except Exception as exc:
            print(f"[WARNING] engineer-done OCR 실패(회차 미판정): {exc}")
            return None
        return extract_numerator(text)

    def _save_debug_crop(self, crop) -> None:
        """debug_dir 설정 시 변화-발화 crop 을 저장한다 (실패 무시)."""
        if self.debug_dir is None:
            return
        try:
            from poc.workflow_3.debug_artifacts import save_debug_jpeg

            self._debug_seq += 1
            save_debug_jpeg(crop, self.debug_dir / f"numerator_{self._debug_seq:03d}.jpg")
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `10/10 통과`, exit 0

검증 포인트(실패 시 디버깅 힌트):
- `test_detector_static_no_ocr` 실패 → `_frame_changed` 의 첫-샘플 처리(`first_sample` 가드) 확인.
- `test_detector_relocalize_after_miss` 실패 → 재grounding 시 `_prev_gray = None` 리셋 확인
  (리셋 안 하면 새 ROI 첫 crop 이 곧장 changed 로 잡혀 호출 수가 어긋난다).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/engineer_done.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): EngineerDoneDetector (VLM 1회 + CV gate + OCR 연속확인)"
```

---

### Task 5: 실배선 builder + 오피스 캘리브레이션 단독 실행

**Files:**
- Modify: `poc/workflow_3/monitor/engineer_done.py` (builder + main)
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (gate 테스트 추가)

- [ ] **Step 1: Write the failing test**

`test_engineer_done.py` 에 추가 (`tests` 리스트에도 추가):

```python
from poc.workflow_3.monitor.engineer_done import build_engineer_done_detector


def test_builder_gates() -> bool:
    """설정 off / tool_window 없음 -> None (고정 timeout 폴백)."""
    ok = True
    off = Workflow3Settings(engineer_done_detect_enabled=False)
    ok &= _check("disabled -> None", build_engineer_done_detector(object(), off) is None)
    on = _settings()
    ok &= _check("no window -> None", build_engineer_done_detector(None, on) is None)
    return ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `ImportError: cannot import name 'build_engineer_done_detector'`

- [ ] **Step 3: Implement builder + calibration main**

`engineer_done.py` 모듈 상단 import 에 추가:

```python
import os
import time
from dataclasses import replace
```

detector 클래스 아래에 추가:

```python
# ------------------------------------------------------------------
# 실배선 builder.
# ------------------------------------------------------------------


def _make_ground_fn(settings, vlm_client=None):
    """grounding closure — ui-venus 단일요소 프롬프트 + [x,y] 파싱."""
    from poc.workflow_3.vlm.prompts.prompt_recipe_monitor_counter import (
        build_recipe_monitor_counter_prompt,
    )
    from poc.workflow_3.util import encode_image_webp

    client = vlm_client
    if client is None or getattr(client, "service_slug", "") != settings.engineer_done_vlm_service:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.engineer_done_vlm_service)

    def ground(image):
        system_message, user_text = build_recipe_monitor_counter_prompt()
        image_b64, _, _ = encode_image_webp(image)
        response = client.chat_with_image_b64(
            image_b64=image_b64, system_message=system_message, user_text=user_text
        )
        return parse_point_1000(response.text)

    return ground


def _make_ocr_fn(settings):
    """분자 crop OCR closure — paddleocr `OCR:` 태스크 (tight crop 만, 환각 회피)."""
    from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_ocr_assist_prompt
    from poc.workflow_3.util import encode_image_webp
    from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

    client = Workflow1VLMClient(settings.engineer_done_ocr_service)

    def ocr(crop):
        system_message, user_text = build_ocr_assist_prompt(crop.size[0], crop.size[1])
        image_b64, _, _ = encode_image_webp(crop)
        response = client.chat_with_image_b64(
            image_b64=image_b64, system_message=system_message, user_text=user_text
        )
        return response.text

    return ocr


def build_engineer_done_detector(tool_window, settings, *, vlm_client=None, debug_dir=None):
    """설정 게이트 확인 후 실 VLM/OCR 배선된 detector 를 만든다.

    비활성/창 없음/클라이언트 생성 실패 → None (호출부는 고정 timeout 폴백).
    cycle 의 OK-버튼용 vlm_client 가 같은 서비스면 재사용한다.
    """
    if not settings.engineer_done_detect_enabled:
        return None
    if tool_window is None:
        return None
    try:
        ground_fn = _make_ground_fn(settings, vlm_client=vlm_client)
        ocr_fn = _make_ocr_fn(settings)
    except Exception as exc:
        print(f"[WARNING] engineer-done 클라이언트 생성 실패(고정 timeout 폴백): {exc}")
        return None
    return EngineerDoneDetector(
        tool_window, settings, ground_fn=ground_fn, ocr_fn=ocr_fn, debug_dir=debug_dir
    )


# ------------------------------------------------------------------
# 오피스 캘리브레이션 단독 실행.
# ------------------------------------------------------------------

# 비우면 env ALIGN_DONE_CALIB_EQP_ID 폴백 (그것도 비면 아무 Remote Monitoring 창).
EQP_ID_OVERRIDE = ""


def run_calibration() -> bool:
    """지금 측정 중인 tool 창에 대해 grounding/gate/OCR 전 체인을 즉시 검증한다.

    측정 중에는 분자가 실제로 증가하므로, align fail 을 기다리지 않고 done 감지
    성공까지의 전 경로(위치 grounding 정확성 포함)를 확인할 수 있다.
    Remote Monitoring 창은 미리 열어 둔다 (직접 또는 workflow_select_tool).
    """
    try:
        from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
    except Exception as exc:
        print(f"[ERROR] RCS 모듈 로드 실패 - 캘리브레이션은 office Windows 전용: {exc}")
        return False

    from poc.workflow_3 import DEBUG_IMAGE_DIR
    from poc.workflow_3.config import load_workflow3_settings

    eqp_id = (EQP_ID_OVERRIDE or "").strip() or os.getenv("ALIGN_DONE_CALIB_EQP_ID", "").strip()
    duration_sec = float(os.getenv("ALIGN_DONE_CALIB_SEC", "120"))

    settings = load_workflow3_settings()
    if not settings.engineer_done_detect_enabled:
        # 캘리브레이션은 게이트를 무시하고 강제 활성(검증이 목적이므로).
        settings = replace(settings, engineer_done_detect_enabled=True)

    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] Remote Monitoring 창 없음 (eqp_id={eqp_id!r}) - 먼저 tool 을 여세요.")
        return False
    print(f"[INFO] 캘리브레이션 대상 창: {title!r}")

    debug_dir = DEBUG_IMAGE_DIR / "engineer_done_calib"
    detector = build_engineer_done_detector(window, settings, debug_dir=debug_dir)
    if detector is None:
        print("[ERROR] detector 생성 실패 - VLM 서비스 설정을 확인하세요.")
        return False

    print(
        f"[INFO] 캘리브레이션 시작: 최대 {duration_sec:.0f}s, "
        f"poll={settings.engineer_done_poll_sec}s, min_count={settings.engineer_done_min_count}, "
        f"debug={debug_dir}"
    )
    deadline = time.time() + duration_sec
    tick = 0
    while time.time() < deadline:
        tick += 1
        done = detector()
        dbg = detector.last_debug
        print(
            f"[INFO] tick {tick}: changed={dbg.get('changed')}, "
            f"n={dbg.get('n')}, miss={dbg.get('ocr_miss_streak', 0)}, done={done}"
        )
        if done:
            print("[INFO] 캘리브레이션 성공: 측정 중 tool 에서 done 감지 체인 검증 완료")
            print("[INFO] 운영 활성화: ALIGN_FAIL_ENGINEER_DONE_DETECT=1")
            return True
        time.sleep(settings.engineer_done_poll_sec)

    print(
        "[WARNING] duration 내 done 미감지 - debug crop 으로 ROI 를 확인하고 "
        "grounding 문구(RECIPE_MONITOR_NUMERATOR_INSTRUCTION)/ROI pad 를 조정하세요."
    )
    return False


if __name__ == "__main__":
    raise SystemExit(0 if run_calibration() else 1)
```

마지막에 `__all__` 추가:

```python
__all__ = [
    "EngineerDoneDetector",
    "build_engineer_done_detector",
    "extract_numerator",
    "parse_point_1000",
    "point_to_roi_ratios",
]
```

- [ ] **Step 4: Run tests to verify they pass (+ Mac 단독 실행 경로 확인)**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `11/11 통과`, exit 0

Run: `uv run python poc/workflow_3/monitor/engineer_done.py`
Expected (Mac): pywinauto 부재로 `[ERROR] RCS 모듈 로드 실패 - 캘리브레이션은 office Windows 전용: ...` 후 exit 1 (크래시 아님)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/engineer_done.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): 실배선 builder + 측정중 tool 캘리브레이션 단독 실행"
```

---

### Task 6: `cycle.py` watch 배선 (done-detector 조기종료)

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py` (`_engineer_watch`, `_exec_run_correction`, `run_alarm_cycle`)
- Test: `poc/workflow_3/monitor/test_engineer_done.py` (watch 단위 테스트 추가)

- [ ] **Step 1: Write the failing tests**

`test_engineer_done.py` 에 추가 (`tests` 리스트에도 추가):

```python
import time as _time

from poc.workflow_3.monitor.cycle import _engineer_watch


class _FakeRecording:
    """is_alive 만 흉내내는 fake (n번째 확인 후 사망 옵션)."""

    def __init__(self, alive_checks: int = 10**6):
        self.alive_checks = alive_checks
        self.checks = 0

    def is_alive(self) -> bool:
        self.checks += 1
        return self.checks <= self.alive_checks


def test_watch_early_exit_on_done() -> bool:
    """detector True -> cap 보다 훨씬 일찍 종료."""
    detector = _CountingFn([False, True])
    started = _time.time()
    _engineer_watch(_FakeRecording(), 60.0, done_detector=detector, poll_sec=0.0)
    elapsed = _time.time() - started
    ok = True
    ok &= _check("early exit well under cap", elapsed < 30.0)
    ok &= _check("detector called twice", detector.calls == 2)
    return ok


def test_watch_detector_exception_safe() -> bool:
    """detector 예외 -> 삼키고 recording 사망/cap 으로 정상 종료."""

    def boom():
        raise RuntimeError("detector crash")

    _engineer_watch(_FakeRecording(alive_checks=2), 60.0, done_detector=boom, poll_sec=0.0)
    return _check("watch survived detector exception", True)


def test_watch_no_detector_unchanged() -> bool:
    """detector 없음 -> 기존 동작(recording 사망 시 종료)."""
    recording = _FakeRecording(alive_checks=3)
    _engineer_watch(recording, 60.0, done_detector=None, poll_sec=0.0)
    return _check("exits on recording death", recording.checks >= 3)
```

주의: `_engineer_watch` 의 내부 sleep 은 2.0s 고정이므로 `_FakeRecording` 사망/done 을
2~3회 loop 안에 끝나게 설계했다 (위 테스트 총 소요 약 10s 내).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `TypeError: _engineer_watch() got an unexpected keyword argument 'done_detector'`

- [ ] **Step 3: Modify cycle.py**

(a) `_engineer_watch` 교체 (`cycle.py` 의 기존 함수 전체를 다음으로):

```python
def _engineer_watch(
    recording: RecordingSession,
    watch_sec: float,
    *,
    done_detector=None,
    poll_sec: float = 8.0,
) -> None:
    """미보정 시 엔지니어 수동 조작 구간 대기 — 녹화 스레드가 계속 캡처한다.

    종료 조건(첫 충족 시): ① 녹화 스레드 자체 종료(창 닫힘=window_gone/max_sec)
    ② done_detector() True (측정 시작 = align 완료, engineer_done 모듈)
    ③ watch_sec 경과 (이제 backstop cap). detector 예외는 ②만 무력화한다.
    """
    if watch_sec <= 0:
        return
    print(
        f"[INFO] engineer watch 시작: 최대 {watch_sec:.0f}s "
        f"(창 닫힘/측정시작 감지/녹화 종료 시 조기 종료, "
        f"감지={'on' if done_detector is not None else 'off'})"
    )
    deadline = time.time() + watch_sec
    next_check = 0.0
    while time.time() < deadline and recording.is_alive():
        if done_detector is not None and time.time() >= next_check:
            try:
                if done_detector():
                    print("[INFO] 측정 시작 감지(align 완료 추정) - engineer watch 조기 종료")
                    log_work2_event(
                        component=LOG_COMPONENT, message="engineer_done_detected"
                    )
                    break
            except Exception as exc:
                print(f"[WARNING] done detector 예외(무시, cap 으로 진행): {exc}")
            next_check = time.time() + max(poll_sec, 0.0)
        time.sleep(2.0)
    print("[INFO] engineer watch 종료")
```

(b) `_exec_run_correction` 에서 vlm_client 생성 직후(`except` 블록 뒤가 아니라
`vlm_client = Workflow1VLMClient(...)` 성공/실패와 무관하게 try/except 블록 **다음 줄**)에 추가:

```python
    context["vlm_client"] = vlm_client
```

(c) `run_alarm_cycle` 의 watch 호출부 교체. 기존:

```python
        # 미보정이면 엔지니어 수동 조작을 녹화하며 대기.
        if recording is not None and (outcome is None or outcome.status != "corrected"):
            _engineer_watch(recording, settings.engineer_watch_sec)
```

신규:

```python
        # 미보정이면 엔지니어 수동 조작을 녹화하며 대기 (측정 시작 감지 시 조기 종료).
        if recording is not None and (outcome is None or outcome.status != "corrected"):
            done_detector = None
            if settings.engineer_done_detect_enabled and context.get("tool_window") is not None:
                try:
                    from poc.workflow_3.monitor.engineer_done import (
                        build_engineer_done_detector,
                    )

                    done_detector = build_engineer_done_detector(
                        context["tool_window"], settings,
                        vlm_client=context.get("vlm_client"),
                        debug_dir=DEBUG_IMAGE_DIR / "engineer_done" / tag,
                    )
                except Exception as exc:
                    print(f"[WARNING] done detector 생성 실패(고정 timeout 으로 진행): {exc}")
            _engineer_watch(
                recording, settings.engineer_watch_sec,
                done_detector=done_detector,
                poll_sec=settings.engineer_done_poll_sec,
            )
```

(import 는 함수 안 지연 import — RCS 부재 dev 환경에서 cycle import 가 죽지 않게 하는
기존 패턴(`correct_align_fail_auto` 지연 import)과 동일. `DEBUG_IMAGE_DIR` 는 cycle.py
상단에 이미 import 되어 있음.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python poc/workflow_3/monitor/test_engineer_done.py`
Expected: `14/14 통과`, exit 0

기존 사이클 영향 없음 회귀 확인:

Run: `uv run python poc/workflow_3/vision/test_align_fail_correct.py`
Expected: 기존과 동일 통과 (작업트리에 사용자 수정이 있으므로 그 결과 기준)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_engineer_done.py
git commit -m "workflow_3(engineer-done): engineer watch done-detector 조기종료 배선"
```

---

### Task 7: README env 표 갱신 + 전체 테스트

**Files:**
- Modify: `poc/workflow_3/README.md`

- [ ] **Step 1: env 표 위치 확인**

Run: `grep -n "ENGINEER_WATCH\|ALIGN_FAIL_RECORDING" poc/workflow_3/README.md`
Expected: env 변수 표/목록의 줄 번호. (없으면 env 섹션을 grep 으로 찾아 그 끝에 추가.)

- [ ] **Step 2: 다음 행 추가** (기존 표 형식에 맞춰; `ALIGN_FAIL_ENGINEER_WATCH_SEC` 항목 근처)

```markdown
| `ALIGN_FAIL_ENGINEER_DONE_DETECT` | `0` | 측정-시작(Recipe Monitor 분자) 감지로 engineer watch 조기 종료. 캘리브레이션(`monitor/engineer_done.py` 단독 실행, 측정 중 tool 대상) 검증 후 `1`. |
| `ALIGN_FAIL_ENGINEER_DONE_POLL_SEC` | `8.0` | watch 안 감지기 호출 간격. |
| `ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT` | `2` | done 으로 보는 최소 분자값(연속 2회 확인). |
| `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE` | `ui-venus-1.5-8b` | 분자 위치 grounding 서비스. |
| `ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE` | `paddleocr-vl-1.5` | 분자 OCR 서비스. |
| `ALIGN_DONE_CALIB_EQP_ID` | (빈값) | 캘리브레이션 대상 tool (빈값=열려있는 아무 Remote Monitoring 창). |
| `ALIGN_DONE_CALIB_SEC` | `120` | 캘리브레이션 최대 실행 시간. |
```

(README 가 표가 아니라 bullet 형식이면 같은 내용을 bullet 로.)

- [ ] **Step 3: 전체 테스트 일괄 확인**

```bash
uv run python poc/workflow_3/monitor/test_engineer_done.py
uv run python poc/workflow_3/vision/test_align_key_match.py
uv run python poc/workflow_3/rcs/test_tool_name_match.py
```

Expected: 각각 전부 통과 (`14/14`, `10/10`, `9/9`).

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/README.md
git commit -m "docs(workflow_3): engineer-done 감지 env + 캘리브레이션 절차"
```

---

## 오피스 검증 절차 (구현 후, 사용자 수행)

1. `git pull` 후 측정이 **정상 진행 중**인 tool 의 Remote Monitoring 창을 연다.
2. `uv run python poc/workflow_3/monitor/engineer_done.py` 실행.
3. 기대: grounding ROI 로그 → tick 마다 `changed=True` + `n` 증가 → `done=True` 로 종료.
4. 실패 시: `debug_images/engineer_done_calib/` 의 numerator crop 확인 →
   crop 이 분자를 안 담으면 `RECIPE_MONITOR_NUMERATOR_INSTRUCTION` 문구 또는
   `ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_X/_Y` 조정 후 재실행.
5. 성공 후 운영 `.env` 에 `ALIGN_FAIL_ENGINEER_DONE_DETECT=1`.
