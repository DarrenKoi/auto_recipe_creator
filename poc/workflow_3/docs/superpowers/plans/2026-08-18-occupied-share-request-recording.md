# 점유 tool 화면 공유 요청 + 엔지니어 작업 녹화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 다른 엔지니어가 점유한 tool 에 대해 RCS `Select` 팝업에서 화면 공유를 요청하고,
승낙되면 그 엔지니어의 수동 align 작업을 녹화한다.

**Architecture:** 검출(`occupied_popup.py`, fail-open)과 클릭(`share_request.py`, fail-closed)을
분리한다. 점유는 3-상태(`occupied_by_other` / `free` / `unknown`)로 판별하며, `unknown` 은
보정을 막는 대신 outcome 을 `corrected_unverified` 로 표시해 알림이 반드시 나가게 한다. 새
outcome status 는 어디서도 성공으로 등록되지 않고 cooldown 재시도 경로로 간다.

**Tech Stack:** Python 3.10+, uv, PIL, pywinauto(Windows 전용), mai-ui VLM 2단계 로케이터,
paddleocr-vl-1.5 OCR 확인.

**Spec:** `poc/workflow_3/docs/superpowers/specs/2026-08-18-occupied-share-request-recording-design.md`

## Global Constraints

- **Korean docstrings** 를 모든 모듈/함수에 쓴다.
- **print 기반 로깅**: `[INFO]` / `[ERROR]` / `[WARNING]` 접두. `logging` 모듈 금지.
  `print()` 문자열 안에 em-dash(U+2014) 금지 — 오피스 콘솔이 cp949 라 인코딩 불가.
  (docstring 안에서는 허용)
- **`from __future__ import annotations` 금지.**
- **CLI 인자 금지.** 설정은 `Workflow3Settings`(`poc/workflow_3/config.py`) 또는 env.
- **절대 임포트**: `from poc.workflow_3.xxx import ...`.
- VLM service 는 **route slug** 를 쓴다 (`"mai-ui"`, `"paddleocr-vl-1.5"`). 모델명 아님.
- 새 테스트는 **Mac 에서 VLM/실장비 없이** 돌아야 한다. `uv run pytest <경로>` 로 실행.
- 실클릭은 `SAFE_MODE=0` 을 요구한다 (`settings.action_enabled`).

---

## File Structure

| 파일 | 책임 |
|---|---|
| `poc/workflow_3/monitor/share_request.py` (신규) | 팝업 actuator. 라디오/버튼 로케이트 → OCR 확인 게이트 → 클릭 → 승낙 대기 |
| `poc/workflow_3/vlm/prompts/prompt_share_options.py` (신규) | 라디오/버튼 로케이트용 `TargetConfig` 설명 문자열 |
| `poc/workflow_3/rcs/row_occupant.py` (신규) | List 행 점유자 컬럼 전용 crop + OCR → 3-상태 |
| `poc/workflow_3/monitor/test_share_request.py` (신규) | 확인 게이트/정책/승낙 대기 단위 테스트 |
| `poc/workflow_3/rcs/test_row_occupant.py` (신규) | 점유 3-상태 판별 단위 테스트 |
| `poc/workflow_3/monitor/test_share_cycle_wiring.py` (신규) | occupancy → outcome → notify → retry 배선 회귀 |
| `poc/workflow_3/config.py` (수정) | `share_*` 설정 4개 |
| `poc/workflow_3/monitor/cycle.py` (수정) | 점유 분기 확장, occupancy → correction 라우팅 |
| `poc/workflow_3/monitor/notify.py` (수정) | 새 status 별 cube 문구 |
| `poc/workflow_3/monitor/align_fail_monitor.py` (수정) | outcome 기반 재시도 집합 + 시도 상한 |

**의존 방향:** `row_occupant`/`share_request` → `vlm`,`util` (leaf 방향). `cycle` 이 둘을
호출한다. 역방향 임포트 없음.

---

## Task 1: 설정 필드 추가

**Files:**
- Modify: `poc/workflow_3/config.py`
- Test: `poc/workflow_3/monitor/test_share_request.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `Workflow3Settings.share_request_enabled: bool`,
  `.share_confirm_policy: str`, `.share_wait_sec: float`, `.share_max_attempts: int`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_request.py` 를 새로 만든다.

```python
"""share_request 단위 테스트 - VLM/실장비 없이 Mac 에서 실행."""

import os

from poc.workflow_3.config import load_workflow3_settings


def test_share_settings_defaults(monkeypatch):
    """기본값: 켜짐, strict, 45초, 2회."""
    for name in (
        "ALIGN_FAIL_SHARE_REQUEST",
        "ALIGN_FAIL_SHARE_CONFIRM",
        "ALIGN_FAIL_SHARE_WAIT_SEC",
        "ALIGN_FAIL_SHARE_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_workflow3_settings()
    assert settings.share_request_enabled is True
    assert settings.share_confirm_policy == "strict"
    assert settings.share_wait_sec == 45.0
    assert settings.share_max_attempts == 2


def test_share_settings_env_override(monkeypatch):
    """env 가 기본값을 이긴다."""
    monkeypatch.setenv("ALIGN_FAIL_SHARE_REQUEST", "0")
    monkeypatch.setenv("ALIGN_FAIL_SHARE_CONFIRM", "off")
    monkeypatch.setenv("ALIGN_FAIL_SHARE_WAIT_SEC", "10")
    monkeypatch.setenv("ALIGN_FAIL_SHARE_MAX_ATTEMPTS", "5")
    settings = load_workflow3_settings()
    assert settings.share_request_enabled is False
    assert settings.share_confirm_policy == "off"
    assert settings.share_wait_sec == 10.0
    assert settings.share_max_attempts == 5
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: FAIL — `AttributeError: 'Workflow3Settings' object has no attribute 'share_request_enabled'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/config.py` 의 `Workflow3Settings` dataclass 에 필드 4개를 추가한다.
`occupied_retry_cooldown_sec`(106행) 근처, 점유 관련 필드 뒤에 둔다.

```python
    # --- 점유 tool 화면 공유 요청 (2026-08-18) ---
    share_request_enabled: bool = True     # Select 팝업에서 화면 공유 요청 발송.
    share_confirm_policy: str = "strict"   # strict | lenient | off - 클릭 전 라벨 OCR 확인.
    share_wait_sec: float = 45.0           # 상대 승낙 대기 상한(초). 블로킹이므로 짧게.
    share_max_attempts: int = 2            # EQP 별 연속 view-only 재시도 상한.
```

같은 파일 `load_workflow3_settings()` 안, `occupied_retry_cooldown_sec=`(286행) 근처에
env 판독을 추가한다.

```python
        share_request_enabled=env_flag("ALIGN_FAIL_SHARE_REQUEST", default=True),
        share_confirm_policy=os.environ.get(
            "ALIGN_FAIL_SHARE_CONFIRM", "strict"
        ).strip().lower() or "strict",
        share_wait_sec=env_float("ALIGN_FAIL_SHARE_WAIT_SEC", 45.0),
        share_max_attempts=env_int("ALIGN_FAIL_SHARE_MAX_ATTEMPTS", 2),
```

`os` 가 이미 임포트되어 있는지 확인하고, 없으면 파일 상단에 `import os` 를 추가한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/config.py poc/workflow_3/monitor/test_share_request.py
git commit -m "feat(config): 점유 tool 화면 공유 요청 설정 4개 추가"
```

---

## Task 2: 확인 게이트 판정 로직

클릭도 VLM 도 없는 순수 판정 함수부터 만든다. 이것이 이 기능의 안전 장치이므로 가장 먼저,
가장 촘촘히 테스트한다.

**Files:**
- Create: `poc/workflow_3/monitor/share_request.py`
- Test: `poc/workflow_3/monitor/test_share_request.py`

**Interfaces:**
- Consumes: Task 1 의 `settings.share_confirm_policy`
- Produces:
  - `SHARE_SCREEN_REQUIRED: tuple[str, ...]`, `SHARE_SCREEN_FORBIDDEN: tuple[str, ...]`
  - `REQUEST_BTN_REQUIRED: tuple[str, ...]`, `REQUEST_BTN_FORBIDDEN: tuple[str, ...]`
  - `classify_label(tokens: list[str], required: tuple[str, ...], forbidden: tuple[str, ...]) -> str`
    반환값 ∈ `{"confirmed", "forbidden", "unreadable"}`
  - `accepts_label(status: str, policy: str) -> bool`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_request.py` 에 아래를 **추가**한다 (Task 1 테스트는 유지).

```python
from poc.workflow_3.monitor.share_request import (
    REQUEST_BTN_FORBIDDEN,
    REQUEST_BTN_REQUIRED,
    SHARE_SCREEN_FORBIDDEN,
    SHARE_SCREEN_REQUIRED,
    accepts_label,
    classify_label,
)


def _radio(tokens):
    return classify_label(tokens, SHARE_SCREEN_REQUIRED, SHARE_SCREEN_FORBIDDEN)


def _button(tokens):
    return classify_label(tokens, REQUEST_BTN_REQUIRED, REQUEST_BTN_FORBIDDEN)


def test_radio_confirmed_english():
    assert _radio(["Request", "to", "share", "the", "screen"]) == "confirmed"


def test_radio_confirmed_case_insensitive():
    assert _radio(["SHARE", "SCREEN"]) == "confirmed"


def test_radio_confirmed_korean():
    assert _radio(["화면", "공유", "요청"]) == "confirmed"


def test_radio_rejects_share_control():
    """제어 공유는 화면 공유가 아니다 - 요청 성격이 다르다."""
    assert _radio(["Request", "to", "share", "the", "control"]) == "forbidden"


def test_radio_rejects_terminate():
    """최악의 오클릭. forbidden 이 required 보다 우선해야 한다."""
    assert _radio(["Request", "termination", "of", "existant", "user"]) == "forbidden"


def test_radio_forbidden_wins_over_required():
    """'share' 와 'terminate' 가 같이 읽혀도 거부여야 한다."""
    assert _radio(["share", "screen", "terminate"]) == "forbidden"


def test_radio_partial_is_unreadable():
    """'share' 만 있고 'screen' 이 없으면 확인된 것이 아니다."""
    assert _radio(["Request", "to", "share"]) == "unreadable"


def test_radio_empty_is_unreadable():
    assert _radio([]) == "unreadable"


def test_button_confirmed():
    assert _button(["Request"]) == "confirmed"


def test_button_rejects_cancel():
    assert _button(["Cancel"]) == "forbidden"


def test_strict_requires_confirmation():
    assert accepts_label("confirmed", "strict") is True
    assert accepts_label("unreadable", "strict") is False
    assert accepts_label("forbidden", "strict") is False


def test_lenient_passes_unreadable_but_never_forbidden():
    """lenient 도 forbidden 은 절대 통과시키지 않는다."""
    assert accepts_label("confirmed", "lenient") is True
    assert accepts_label("unreadable", "lenient") is True
    assert accepts_label("forbidden", "lenient") is False


def test_off_still_blocks_forbidden():
    """off 는 진단용이지만 terminate 오클릭까지 허용하지는 않는다."""
    assert accepts_label("confirmed", "off") is True
    assert accepts_label("unreadable", "off") is True
    assert accepts_label("forbidden", "off") is False


def test_unknown_policy_falls_back_to_strict():
    assert accepts_label("unreadable", "typo-policy") is False
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.monitor.share_request'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/monitor/share_request.py` 를 만든다.

```python
"""점유 'select' 팝업 actuator - 화면 공유를 요청한다.

`occupied_popup.py` 는 팝업을 **검출만** 하는 fail-open detector 다(예외를 False 로
흡수해 검출 실패가 접속을 막지 않게 한다). 이 모듈은 반대로 **클릭하는** fail-closed
actuator 다 - 확신이 없으면 누르지 않는다. 두 정책이 정반대라 파일을 나눴다.

세 라디오(제어 공유 / 화면 공유 / 기존 사용자 강제 종료)가 세로로 붙어 있어 fine 단계가
한 칸 어긋나는 것이 가장 현실적인 실패이고, 그 어긋남의 최악이 강제 종료다. 그래서 좌표를
찍은 뒤 그 자리의 라벨을 좁은 crop 으로 OCR 해 확인하고, 확인되지 않으면 클릭하지 않는다.
"""

# 확인 게이트 토큰. 대소문자 무시 부분 일치로 비교하며 국문 표기도 함께 받는다.
# 오피스 실제 문구는 첫 실행의 진단 산출물로 확인한 뒤 조정한다.
SHARE_SCREEN_REQUIRED = ("share", "screen", "공유", "화면")
SHARE_SCREEN_FORBIDDEN = ("control", "terminat", "제어", "종료")
REQUEST_BTN_REQUIRED = ("request", "요청")
REQUEST_BTN_FORBIDDEN = ("cancel", "취소")

CONFIRM_STRICT = "strict"
CONFIRM_LENIENT = "lenient"
CONFIRM_OFF = "off"


def _normalize(token: str) -> str:
    """비교용 정규화 - 소문자로 낮추고 양끝 공백/구두점을 턴다."""
    return (token or "").strip().strip(".,:;()[]").lower()


def _any_token_contains(tokens: list[str], needles: tuple) -> bool:
    """토큰 중 하나라도 needles 의 어떤 문자열을 포함하는지."""
    normalized = [_normalize(token) for token in tokens]
    for needle in needles:
        target = needle.lower()
        if any(target in token for token in normalized if token):
            return True
    return False


def classify_label(tokens: list[str], required: tuple, forbidden: tuple) -> str:
    """읽은 토큰이 기대 라벨인지 판정한다.

    "confirmed"  : required 를 만족하고 forbidden 이 없다.
    "forbidden"  : forbidden 토큰이 읽혔다 (required 여부와 무관하게 우선).
    "unreadable" : 어느 쪽도 확정할 수 없다.

    forbidden 을 required 보다 **먼저** 본다. 'share screen' 과 'terminate' 가 함께
    읽히는 상황은 crop 이 옆 라디오까지 삼켰다는 뜻이라 클릭해서는 안 된다.
    """
    if _any_token_contains(tokens, forbidden):
        return "forbidden"

    # required 는 영문 쌍(share+screen)과 국문 쌍(공유+화면)을 각각 만족시켜야 한다.
    # 절반만 읽힌 것은 확인이 아니다 - 'share' 만으로는 제어 공유와 구분되지 않는다.
    english = [needle for needle in required if needle.isascii()]
    korean = [needle for needle in required if not needle.isascii()]
    if english and all(_any_token_contains(tokens, (needle,)) for needle in english):
        return "confirmed"
    if korean and all(_any_token_contains(tokens, (needle,)) for needle in korean):
        return "confirmed"
    return "unreadable"


def accepts_label(status: str, policy: str) -> bool:
    """확인 판정과 정책으로 클릭 허용 여부를 정한다.

    forbidden 은 **어떤 정책에서도** 통과하지 않는다. off 는 좌표 진단용이지 강제 종료
    오클릭까지 허용하라는 뜻이 아니다.
    """
    if status == "forbidden":
        return False
    if policy in (CONFIRM_LENIENT, CONFIRM_OFF):
        return True
    # strict 및 오타 정책 - 확인된 것만 통과(안전한 쪽으로 폴백).
    return status == "confirmed"


__all__ = [
    "CONFIRM_LENIENT",
    "CONFIRM_OFF",
    "CONFIRM_STRICT",
    "REQUEST_BTN_FORBIDDEN",
    "REQUEST_BTN_REQUIRED",
    "SHARE_SCREEN_FORBIDDEN",
    "SHARE_SCREEN_REQUIRED",
    "accepts_label",
    "classify_label",
]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/share_request.py poc/workflow_3/monitor/test_share_request.py
git commit -m "feat(share): 클릭 전 라벨 확인 게이트 판정 로직"
```

---

## Task 3: 점유 3-상태 판별 (`row_occupant`)

**Files:**
- Create: `poc/workflow_3/rcs/row_occupant.py`
- Test: `poc/workflow_3/rcs/test_row_occupant.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `OCCUPIED_BY_OTHER = "occupied_by_other"`, `FREE = "free"`, `UNKNOWN = "unknown"`
  - `looks_like_occupant(token: str) -> bool`
  - `classify_occupancy(read_ok: bool, tokens: list[str]) -> str`
  - `build_occupant_box(row_point: dict, image_width: int, image_height: int) -> dict`

**중요 (spec ③):** 이 모듈은 **자기 crop 과 자기 OCR 호출**을 가진다.
`rcs/tool_row_verify.py` 의 strip 을 넓혀 재사용하면 안 된다 —
`_looks_like_tool_id` 가 점유자 ID(`KIM0234`)를 장비 ID 로 오인해
`classify_tokens` 가 `mismatch` 를 내고, `accepts()` 는 `lenient` 에서도 `mismatch` 를
거부하므로 **정상 행의 클릭이 거부된다.** `tool_row_verify.py` 는 이 태스크에서 수정하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/rcs/test_row_occupant.py` 를 만든다.

```python
"""row_occupant 단위 테스트 - VLM/실장비 없이 Mac 에서 실행."""

from poc.workflow_3.rcs.row_occupant import (
    FREE,
    OCCUPIED_BY_OTHER,
    UNKNOWN,
    build_occupant_box,
    classify_occupancy,
    looks_like_occupant,
)


def test_occupant_id_shape():
    """사번형 ID: 영숫자 + 글자·숫자 혼재."""
    assert looks_like_occupant("KIM0234") is True
    assert looks_like_occupant("HYN1A2B") is True


def test_pure_word_is_not_occupant():
    """상태 문자열은 점유자가 아니다."""
    assert looks_like_occupant("Idle") is False
    assert looks_like_occupant("Status") is False


def test_pure_number_is_not_occupant():
    assert looks_like_occupant("12345") is False


def test_punctuated_is_not_occupant():
    """IP 나 시각은 점 콜론 때문에 탈락한다."""
    assert looks_like_occupant("10.1.2.3") is False
    assert looks_like_occupant("12:30:05") is False


def test_too_short_is_not_occupant():
    assert looks_like_occupant("A1") is False


def test_occupied_when_occupant_token_present():
    assert classify_occupancy(True, ["KIM0234"]) == OCCUPIED_BY_OTHER


def test_free_when_read_ok_and_no_occupant_token():
    """읽기는 성공했는데 점유자 모양 토큰이 없으면 비어 있는 것이다."""
    assert classify_occupancy(True, []) == FREE
    assert classify_occupancy(True, ["-"]) == FREE


def test_unknown_when_read_failed():
    """읽기 실패는 '비어 있음' 이 아니라 '모름' 이다. 이 구분이 이 모듈의 존재 이유다."""
    assert classify_occupancy(False, []) == UNKNOWN
    assert classify_occupancy(False, ["KIM0234"]) == UNKNOWN


def test_occupant_box_extends_right_of_row_point():
    """점유자 컬럼은 장비 ID 컬럼 오른쪽에 있다."""
    box = build_occupant_box({"x": 100, "y": 200}, 1920, 1080)
    assert box["left"] >= 100
    assert box["right"] > box["left"]
    assert box["top"] < 200 < box["bottom"]


def test_occupant_box_clamped_to_image():
    """행이 오른쪽 끝에 있어도 이미지 밖으로 나가지 않는다."""
    box = build_occupant_box({"x": 1900, "y": 5}, 1920, 1080)
    assert box["right"] <= 1920
    assert box["top"] >= 0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/rcs/test_row_occupant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.rcs.row_occupant'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/rcs/row_occupant.py` 를 만든다.

```python
"""List 행의 점유자 컬럼을 읽어 점유를 3-상태로 판별한다.

점유는 참/거짓이 아니라 3-상태다. "모른다" 를 "비어 있다" 로 접으면, 화면 공유(view-only)
세션에서 먹지 않는 클릭을 하고도 '보정 완료' 로 보고하는 조용한 오보가 된다.

**이 모듈은 자기 crop 과 자기 OCR 호출을 가진다.** `tool_row_verify` 의 행 strip 을 넓혀
한 번에 읽으면 안 된다: 그쪽 `_looks_like_tool_id` 가 점유자 ID(KIM0234 등)를 장비 ID 로
오인해 `mismatch` 를 내고, `accepts()` 는 lenient 에서도 mismatch 를 거부하므로 정상 행의
클릭이 거부된다. 지금은 무해한 unreadable 이 파괴적인 mismatch 로 승격되는 셈이다.
"""

from poc.workflow_3.vlm.label_verify import crop_box_around_point

OCCUPIED_BY_OTHER = "occupied_by_other"
FREE = "free"
UNKNOWN = "unknown"

# 점유자 ID 길이 범위. 사번/계정 형태를 받되 한두 글자 잡음은 배제한다.
OCCUPANT_MIN_LEN = 4
OCCUPANT_MAX_LEN = 16

# 점유자 컬럼 crop 기하 - 장비 ID 점 기준 오른쪽으로 뻗는다.
# 오피스 캘리브레이션 대상: 첫 실행에서 crop 산출물을 보고 조정한다.
OCCUPANT_LEFT_RATIO = 0.02
OCCUPANT_RIGHT_RATIO = 0.30
OCCUPANT_HALF_HEIGHT_RATIO = 0.010
OCCUPANT_MIN_HALF_HEIGHT_PX = 8


def looks_like_occupant(token: str) -> bool:
    """토큰이 사람 ID 모양인지 - 영숫자만 + 글자와 숫자를 모두 포함 + 길이 범위.

    IP(10.1.2.3)나 시각(12:30:05)은 구두점 때문에 isalnum 에서 탈락하고,
    'Idle'/'Status' 는 숫자가 없어서, 순수 카운트 숫자는 글자가 없어서 탈락한다.
    """
    cleaned = (token or "").strip()
    if not cleaned or not cleaned.isalnum():
        return False
    if not (OCCUPANT_MIN_LEN <= len(cleaned) <= OCCUPANT_MAX_LEN):
        return False
    return any(ch.isdigit() for ch in cleaned) and any(ch.isalpha() for ch in cleaned)


def classify_occupancy(read_ok: bool, tokens: list) -> str:
    """OCR 성공 여부와 토큰으로 점유 3-상태를 정한다.

    read_ok=False 는 UNKNOWN 이다. 읽지 못한 것을 '비어 있음' 으로 접으면 view-only
    세션이 보정 가능 세션으로 오인된다.
    """
    if not read_ok:
        return UNKNOWN
    if any(looks_like_occupant(token) for token in (tokens or [])):
        return OCCUPIED_BY_OTHER
    return FREE


def build_occupant_box(row_point: dict, image_width: int, image_height: int) -> dict:
    """장비 ID 점 기준으로 점유자 컬럼 crop box 를 만든다 (경계 clamp 포함).

    좌표를 새로 만들지 않는다 - 이미 확정된 행 좌표에서 오른쪽으로 뻗을 뿐이다.
    """
    return crop_box_around_point(
        row_point,
        image_width,
        image_height,
        left_ratio=OCCUPANT_LEFT_RATIO,
        right_ratio=OCCUPANT_RIGHT_RATIO,
        half_height_ratio=OCCUPANT_HALF_HEIGHT_RATIO,
        min_half_height_px=OCCUPANT_MIN_HALF_HEIGHT_PX,
    )


__all__ = [
    "FREE",
    "OCCUPIED_BY_OTHER",
    "UNKNOWN",
    "build_occupant_box",
    "classify_occupancy",
    "looks_like_occupant",
]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/rcs/test_row_occupant.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 기존 게이트 무회귀 확인**

Run: `uv run python poc/workflow_3/rcs/test_tool_row_verify.py`
Expected: 42/42 통과 (`tool_row_verify.py` 를 건드리지 않았음을 확인)

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/rcs/row_occupant.py poc/workflow_3/rcs/test_row_occupant.py
git commit -m "feat(rcs): List 행 점유자 컬럼 3-상태 판별 (전용 crop)"
```

---

## Task 4: 승낙 대기 판정

**Files:**
- Modify: `poc/workflow_3/monitor/share_request.py`
- Test: `poc/workflow_3/monitor/test_share_request.py`

**Interfaces:**
- Consumes: Task 2 의 `share_request` 모듈
- Produces:
  - `ACCEPTED = "accepted"`, `DENIED_OR_TIMEOUT = "denied_or_timeout"`
  - `wait_share_response(eqp_id, wait_sec, *, find_window_fn, sleep_fn=time.sleep, now_fn=time.monotonic, poll_sec=1.0) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_request.py` 에 추가한다.

```python
from poc.workflow_3.monitor.share_request import (
    ACCEPTED,
    DENIED_OR_TIMEOUT,
    wait_share_response,
)


class _FakeClock:
    """단조 시계 대역 - 실제로 자지 않고 시간만 흘린다."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_accepted_when_window_appears_immediately():
    clock = _FakeClock()
    result = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=lambda eqp_id: object(),
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert result == ACCEPTED


def test_accepted_when_window_appears_late():
    """상대가 뒤늦게 수락하는 경우."""
    clock = _FakeClock()
    calls = {"n": 0}

    def find(eqp_id):
        calls["n"] += 1
        return object() if calls["n"] >= 4 else None

    result = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=find, sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert result == ACCEPTED


def test_timeout_when_window_never_appears():
    clock = _FakeClock()
    result = wait_share_response(
        "MCD427", 5.0,
        find_window_fn=lambda eqp_id: None,
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert result == DENIED_OR_TIMEOUT
    assert clock.t >= 5.0


def test_window_lookup_exception_does_not_abort_wait():
    """탐색 1회 실패가 대기 전체를 죽이면 안 된다 - 창은 뒤에 뜰 수 있다."""
    clock = _FakeClock()
    calls = {"n": 0}

    def find(eqp_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("window enum failed")
        return object()

    result = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=find, sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert result == ACCEPTED


def test_zero_wait_returns_timeout_without_sleeping():
    clock = _FakeClock()
    result = wait_share_response(
        "MCD427", 0.0,
        find_window_fn=lambda eqp_id: None,
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert result == DENIED_OR_TIMEOUT
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: FAIL — `ImportError: cannot import name 'ACCEPTED'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/monitor/share_request.py` 에 추가한다. 파일 상단에 `import time` 을 넣는다.

```python
ACCEPTED = "accepted"
DENIED_OR_TIMEOUT = "denied_or_timeout"


def wait_share_response(
    eqp_id: str,
    wait_sec: float,
    *,
    find_window_fn,
    sleep_fn=time.sleep,
    now_fn=time.monotonic,
    poll_sec: float = 1.0,
) -> str:
    """공유 요청 후 상대의 승낙을 기다린다.

    승낙 신호는 '제목에 eqp_id 를 가진 Remote Monitoring 창의 등장' 하나뿐이다.
    거절과 무응답은 하나로 합친다 - 거절 시 RCS 화면이 확정되지 않았고, 어느 쪽이든
    결론은 '그 엔지니어가 점유하는 동안 접근 불가' 로 같아 동작이 갈리지 않는다.

    이 대기는 **블로킹**이며 단일 RCS 커서를 모든 tool 의 알람이 직렬로 공유하므로,
    wait_sec 은 짧게 둔다(기본 45초).

    시계와 창 탐색은 주입받는다 - 실장비 없이 테스트하기 위해서다.
    """
    deadline = now_fn() + max(0.0, wait_sec)
    while now_fn() < deadline:
        try:
            if find_window_fn(eqp_id) is not None:
                print(f"[INFO] 화면 공유 승낙됨 - tool 창 등장: EQP_ID={eqp_id}")
                return ACCEPTED
        except Exception as exc:
            # 탐색 1회 실패로 대기를 끝내지 않는다 - 창은 다음 poll 에 뜰 수 있다.
            print(f"[WARNING] 공유 대기 중 창 탐색 실패(계속 대기): {exc}")
        sleep_fn(poll_sec)
    print(f"[INFO] 화면 공유 무응답/거절: EQP_ID={eqp_id} ({wait_sec:.0f}s 경과)")
    return DENIED_OR_TIMEOUT
```

`__all__` 에 `"ACCEPTED"`, `"DENIED_OR_TIMEOUT"`, `"wait_share_response"` 를 추가한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: PASS (21 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/share_request.py poc/workflow_3/monitor/test_share_request.py
git commit -m "feat(share): 승낙 대기 판정 (거절/무응답 통합)"
```

---

## Task 5: outcome 상태 상수와 notify 문구

배선 전에 소비자 쪽을 먼저 만든다. 여기가 조용한 오보를 막는 지점이다.

**Files:**
- Modify: `poc/workflow_3/align/correction.py:100-117` (docstring 주석만)
- Modify: `poc/workflow_3/monitor/notify.py`
- Create: `poc/workflow_3/monitor/test_share_cycle_wiring.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `poc.workflow_3.monitor.notify.VIEW_ONLY_OBSERVATION = "view_only_observation"`
  - `poc.workflow_3.monitor.notify.CORRECTED_UNVERIFIED = "corrected_unverified"`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_cycle_wiring.py` 를 만든다.

```python
"""점유 공유 요청 기능의 배선 회귀 - occupancy → outcome → notify → retry."""

from dataclasses import dataclass

from poc.workflow_3.monitor.notify import (
    CORRECTED_UNVERIFIED,
    VIEW_ONLY_OBSERVATION,
    notify_correction_outcome,
)


@dataclass
class _Outcome:
    """CorrectionOutcome 의 notify 가 실제로 읽는 필드만 갖는 대역."""

    status: str
    path: str = "primary"
    key_decision: str = ""
    best_xy: object = None
    ok_screen_xy: object = None
    fallback: object = None
    error: object = None
    second_ratio: object = None


def _sent_calls(monkeypatch, outcome):
    """notify_correction_outcome 이 office 발송을 호출했는지 기록한다."""
    calls = []
    monkeypatch.setattr(
        "poc.workflow_3.monitor.notify.send_rich_notify_async",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    notify_correction_outcome("MCD427", "CLS/RCP", outcome, enabled=True)
    return calls


def test_corrected_suppresses_cube(monkeypatch):
    """기존 동작 - 성공은 알리지 않는다."""
    assert _sent_calls(monkeypatch, _Outcome(status="corrected")) == []


def test_view_only_observation_still_notifies(monkeypatch):
    """점유자와 알람 담당자는 다른 사람일 수 있다. 생략하면 아무도 모른다."""
    assert _sent_calls(monkeypatch, _Outcome(status=VIEW_ONLY_OBSERVATION)) != []


def test_corrected_unverified_still_notifies(monkeypatch):
    """보정 반영 여부가 미확인이면 반드시 알린다 - 조용한 성공 금지."""
    assert _sent_calls(monkeypatch, _Outcome(status=CORRECTED_UNVERIFIED)) != []


def test_new_statuses_are_not_equal_to_corrected():
    """정확 비교(==)에 걸리지 않아야 watch/cube 기존 분기가 유지된다."""
    assert VIEW_ONLY_OBSERVATION != "corrected"
    assert CORRECTED_UNVERIFIED != "corrected"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name 'VIEW_ONLY_OBSERVATION'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/monitor/notify.py` 상단(`LOG_COMPONENT` 정의 근처)에 상수를 추가한다.

```python
# 점유 tool 관련 outcome status (2026-08-18). 둘 다 "corrected" 가 아니므로
# engineer watch(cycle.py) 와 cube 발송 분기를 기존 정확 비교 그대로 통과한다.
VIEW_ONLY_OBSERVATION = "view_only_observation"   # 다른 엔지니어 점유 - 관전·녹화만.
CORRECTED_UNVERIFIED = "corrected_unverified"     # 점유 미상 - 보정했으나 반영 미확인.
```

`build_outcome_summary()` 안에서 status 별 사람이 읽을 문구를 붙인다. 함수 안의 status
분기 지점에 아래를 추가한다 (기존 분기는 그대로 둔다).

```python
    if status == VIEW_ONLY_OBSERVATION:
        summary = f"다른 엔지니어 점유 중 - 관전·녹화만 수행 | {summary}"
    elif status == CORRECTED_UNVERIFIED:
        summary = f"점유 여부 확인 불가 - 보정 시도했으나 반영 여부 미확인 | {summary}"
```

`notify_correction_outcome` 의 `if status == "corrected":` 분기는 **수정하지 않는다.**
두 새 status 는 정확 비교에 걸리지 않아 자동으로 발송 경로로 간다 — 그것이 의도다.

`__all__` 이 있으면 두 상수를 추가한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 기존 notify 무회귀 확인**

Run: `uv run python poc/workflow_3/monitor/test_notify.py`
Expected: 기존 통과 수 유지

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/monitor/notify.py poc/workflow_3/monitor/test_share_cycle_wiring.py
git commit -m "feat(notify): view_only/unverified status 는 cube 를 생략하지 않는다"
```

---

## Task 6: 상위 루프 재시도 규칙

**Files:**
- Modify: `poc/workflow_3/monitor/align_fail_monitor.py:280-297` (분류 상수),
  `:400-424` (분기)
- Test: `poc/workflow_3/monitor/test_share_cycle_wiring.py`

**Interfaces:**
- Consumes: Task 5 의 `VIEW_ONLY_OBSERVATION`, `CORRECTED_UNVERIFIED`
- Produces:
  - `_RETRY_LATER_OUTCOME_STATUSES: set[str]`
  - `_should_retry_later(cycle) -> bool`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_cycle_wiring.py` 에 추가한다.

```python
from poc.workflow_3.monitor.align_fail_monitor import (
    _RETRY_LATER_OUTCOME_STATUSES,
    _should_retry_later,
)
from poc.workflow_3.monitor.cycle import CycleResult


def _cycle(outcome_status="", run_status="completed", failed_step=""):
    cycle = CycleResult(eqp_id="MCD427", recipe_id="CLS/RCP", tag="t")
    cycle.run_status = run_status
    cycle.failed_step = failed_step
    cycle.outcome_status = outcome_status
    return cycle


def test_view_only_never_registers_success():
    """완주했더라도 active_tools 로 가면 tool 이 풀려도 영영 돌아오지 않는다."""
    assert _should_retry_later(_cycle(outcome_status=VIEW_ONLY_OBSERVATION)) is True


def test_corrected_unverified_never_registers_success():
    assert _should_retry_later(_cycle(outcome_status=CORRECTED_UNVERIFIED)) is True


def test_corrected_registers_success():
    assert _should_retry_later(_cycle(outcome_status="corrected")) is False


def test_awaiting_engineer_ok_registers_success():
    """반자동 정상 경로는 기존대로 active - 알람 해제까지 머문다."""
    assert _should_retry_later(_cycle(outcome_status="awaiting_engineer_ok")) is False


def test_retry_set_contents():
    assert _RETRY_LATER_OUTCOME_STATUSES == {
        VIEW_ONLY_OBSERVATION,
        CORRECTED_UNVERIFIED,
    }
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name '_RETRY_LATER_OUTCOME_STATUSES'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/monitor/align_fail_monitor.py` 의 `_MISCLICK_FAILURE_CLASSES`(284행) 아래에
추가한다.

```python
# 확인 게이트가 막아 공유 요청을 못 보낸 경우 - 장비 탓이 아니라 우리 인식 실패라
# 오클릭과 같은 성격이다(cooldown 후 재시도).
_SHARE_FAILURE_CLASSES = {"rcs_share_confirm_failed"}
_RETRY_LATER_FAILURE_CLASSES = (
    _OCCUPIED_FAILURE_CLASSES | _MISCLICK_FAILURE_CLASSES | _SHARE_FAILURE_CLASSES
)

# outcome 기반 재시도 - 사이클이 **완주했더라도** 성공으로 등록하면 안 되는 status.
# _cycle_failed 는 run_status/failed_step 만 보므로 이 둘은 그대로 두면 active_tools 에
# 등록되어 알람 해제까지 재시도되지 않는다. 점유자가 tool 을 놓아준 뒤에도 우리는 돌아가지
# 않게 되므로, 실제 보정이 돌 기회를 잃는다.
_RETRY_LATER_OUTCOME_STATUSES = {VIEW_ONLY_OBSERVATION, CORRECTED_UNVERIFIED}
```

기존 `_RETRY_LATER_FAILURE_CLASSES = _OCCUPIED_FAILURE_CLASSES | _MISCLICK_FAILURE_CLASSES`
줄은 위 블록으로 대체되므로 삭제한다.

임포트에 추가한다 (`from poc.workflow_3.monitor.notify import (...)` 블록).

```python
    CORRECTED_UNVERIFIED,
    VIEW_ONLY_OBSERVATION,
```

`_cycle_failed` 함수 아래에 판정 함수를 추가한다.

```python
def _should_retry_later(cycle) -> bool:
    """이 사이클을 active 로 굳히지 않고 cooldown 재시도로 보내야 하는가.

    사이클이 완주했어도(run_status='completed') outcome 이 '보정이 실제로 반영되었다'
    를 보장하지 못하면 성공으로 등록하지 않는다. 그래야 점유가 풀렸을 때 다시 붙는다.
    """
    return (cycle.outcome_status or "") in _RETRY_LATER_OUTCOME_STATUSES
```

`process_fail_rows` 안, `elif _cycle_failed(cycle):`(414행) **앞에** 분기를 넣는다.
시도 상한을 위해 함수 시그니처에 `view_only_attempts: dict | None = None` 을 추가하고
`if view_only_attempts is None: view_only_attempts = {}` 로 기본값을 잡는다.

```python
            elif _should_retry_later(cycle):
                attempts = view_only_attempts.get(eqp_id, 0) + 1
                view_only_attempts[eqp_id] = attempts
                if attempts >= settings.share_max_attempts:
                    # 상한 도달 - 더 재시도하면 cooldown 마다 cube 가 나가고 단일 RCS
                    # 커서를 계속 점유한다. 엔지니어는 이미 상한만큼 통보받았다.
                    active_tools.add(eqp_id)
                    print(
                        f"[WARNING] EQP_ID={eqp_id} {cycle.outcome_status} "
                        f"{attempts}회 - 상한 도달, 재시도 중단(알람 해제까지 대기)"
                    )
                else:
                    occupied_cooldown[eqp_id] = (
                        time.time() + settings.failure_retry_cooldown_sec
                    )
                    print(
                        f"[INFO] EQP_ID={eqp_id} {cycle.outcome_status} "
                        f"({attempts}/{settings.share_max_attempts}) - active 미등록, "
                        f"{settings.failure_retry_cooldown_sec:.0f}s 후 재시도"
                    )
```

cooldown 정리 루프(328-331행) 옆에서 카운터도 함께 정리한다. 알람이 해제되면
(`eqp_id not in current_tools`) 카운터를 지운다.

```python
    for eqp_id in list(view_only_attempts):
        if eqp_id not in current_tools:
            del view_only_attempts[eqp_id]
```

`monitor_loop` 안에서 dict 를 만들어 넘긴다. `occupied_cooldown: dict = {}`(452행) 아래에
`view_only_attempts: dict = {}  # {eqp_id: 연속 view-only/unverified 횟수}` 를 추가하고,
`process_fail_rows(fails, active_tools, settings, occupied_cooldown)`(511행) 호출을
`process_fail_rows(fails, active_tools, settings, occupied_cooldown, view_only_attempts)` 로
바꾼다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: 기존 cooldown 무회귀 확인**

Run: `uv run python poc/workflow_3/monitor/test_failure_cooldown.py`
Expected: 기존 통과 수 유지

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/monitor/align_fail_monitor.py poc/workflow_3/monitor/test_share_cycle_wiring.py
git commit -m "feat(monitor): view_only/unverified 는 active 가 아니라 cooldown 재시도 (상한 포함)"
```

---

## Task 7: 팝업 클릭 실행 경로

VLM/pynput 을 실제로 부르는 유일한 태스크. Mac 에서는 주입된 대역으로만 검증한다.

**Files:**
- Modify: `poc/workflow_3/monitor/share_request.py`
- Create: `poc/workflow_3/vlm/prompts/prompt_share_options.py`
- Test: `poc/workflow_3/monitor/test_share_request.py`

**Interfaces:**
- Consumes: Task 2 의 `classify_label`/`accepts_label`, Task 4 의 `wait_share_response`
- Produces:
  - `@dataclass ShareRequestResult(status: str, radio_verdict: str = "", button_verdict: str = "", error: str = "")`
  - `STATUS_REQUESTED/CONFIRM_FAILED/NOT_FOUND/BLOCKED_SAFE_MODE/ERROR`
  - `request_screen_share(settings, *, locate_fn, read_tokens_fn, click_fn, capture_fn, find_popup_fn) -> ShareRequestResult`
  - `SHARE_SCREEN_TARGET: TargetConfig`, `REQUEST_BUTTON_TARGET: TargetConfig`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_request.py` 에 추가한다.

```python
from poc.workflow_3.monitor.share_request import (
    STATUS_BLOCKED_SAFE_MODE,
    STATUS_CONFIRM_FAILED,
    STATUS_NOT_FOUND,
    STATUS_REQUESTED,
    ShareRequestResult,
    request_screen_share,
)


class _Settings:
    """request_screen_share 가 읽는 필드만 갖는 대역."""

    def __init__(self, policy="strict", action_enabled=True):
        self.share_confirm_policy = policy
        self.action_enabled = action_enabled


def _run(*, radio_tokens, button_tokens, policy="strict", action_enabled=True,
         popup=object(), point=None):
    """클릭 기록을 돌려주는 공통 실행기."""
    clicks = []
    point = point if point is not None else {"x": 10, "y": 20}
    result = request_screen_share(
        _Settings(policy, action_enabled),
        locate_fn=lambda image, target: point,
        read_tokens_fn=lambda image, box, label: (
            radio_tokens if label == "share_screen_radio" else button_tokens
        ),
        click_fn=lambda x, y, label: clicks.append(label),
        capture_fn=lambda window: object(),
        find_popup_fn=lambda: popup,
    )
    return result, clicks


def test_requests_when_both_labels_confirmed():
    result, clicks = _run(
        radio_tokens=["Request", "to", "share", "the", "screen"],
        button_tokens=["Request"],
    )
    assert result.status == STATUS_REQUESTED
    assert clicks == ["share_screen_radio", "request_button"]


def test_no_click_at_all_when_radio_reads_terminate():
    """최악의 경우. 라디오조차 누르지 않아야 한다."""
    result, clicks = _run(
        radio_tokens=["Request", "termination", "of", "existant", "user"],
        button_tokens=["Request"],
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_no_click_when_button_reads_cancel():
    """버튼 확인은 라디오 클릭 **전에** 끝나야 한다."""
    result, clicks = _run(
        radio_tokens=["share", "screen"],
        button_tokens=["Cancel"],
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_strict_blocks_unreadable():
    result, clicks = _run(radio_tokens=[], button_tokens=[])
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_lenient_allows_unreadable():
    result, clicks = _run(radio_tokens=[], button_tokens=[], policy="lenient")
    assert result.status == STATUS_REQUESTED
    assert clicks == ["share_screen_radio", "request_button"]


def test_lenient_still_blocks_terminate():
    result, clicks = _run(
        radio_tokens=["terminate"], button_tokens=["Request"], policy="lenient",
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_popup_missing_is_not_found():
    result, clicks = _run(
        radio_tokens=["share", "screen"], button_tokens=["Request"], popup=None,
    )
    assert result.status == STATUS_NOT_FOUND
    assert clicks == []


def test_safe_mode_blocks_click():
    result, clicks = _run(
        radio_tokens=["share", "screen"], button_tokens=["Request"],
        action_enabled=False,
    )
    assert result.status == STATUS_BLOCKED_SAFE_MODE
    assert clicks == []


def test_locate_failure_is_confirm_failed():
    result, clicks = _run(
        radio_tokens=["share", "screen"], button_tokens=["Request"], point=None,
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_verdicts_are_reported_for_diagnosis():
    """오피스 진단을 위해 판정이 결과에 실려야 한다."""
    result, _ = _run(radio_tokens=["control"], button_tokens=["Request"])
    assert result.radio_verdict == "forbidden"
```

주의: `test_locate_failure_is_confirm_failed` 는 `point=None` 을 넘기지만 `_run` 의
기본값 처리가 `None` 을 대체하므로, 이 테스트만 `locate_fn` 이 `None` 을 돌려주도록
별도로 작성한다.

```python
def test_locate_failure_is_confirm_failed():
    clicks = []
    result = request_screen_share(
        _Settings(),
        locate_fn=lambda image, target: None,
        read_tokens_fn=lambda image, box, label: ["share", "screen"],
        click_fn=lambda x, y, label: clicks.append(label),
        capture_fn=lambda window: object(),
        find_popup_fn=lambda: object(),
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []
```

앞서 쓴 중복 정의는 지운다.

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: FAIL — `ImportError: cannot import name 'request_screen_share'`

- [ ] **Step 3: 프롬프트/타겟 정의 작성**

`poc/workflow_3/vlm/prompts/prompt_share_options.py` 를 만든다.

```python
"""점유 'select' 팝업의 라디오/버튼 로케이트용 타겟 정의.

2단계 로케이터(`vlm/ui_venus_mai_locator.py`)에 넘길 TargetConfig 다. 세 라디오가 세로로
촘촘히 붙어 있으므로 vertical_pad 하한을 낮춰 위아래 항목을 crop 이 삼키지 않게 한다 -
tool List 행에서 배운 것과 같은 이유다.
"""

from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

SHARE_SCREEN_TARGET = TargetConfig(
    key="share_screen_radio",
    description=(
        "the radio button or checkbox for requesting to SHARE THE SCREEN "
        "(view/observe only), in the occupied-tool selection dialog. "
        "Do NOT pick the option about sharing CONTROL, and do NOT pick the "
        "option about TERMINATING the existing user."
    ),
    vertical_pad_ratio=0.6,
    vertical_pad_min_px=10,
    min_crop_height=56,
)

REQUEST_BUTTON_TARGET = TargetConfig(
    key="request_button",
    description=(
        "the button labelled 'Request' that submits the selected option "
        "in the occupied-tool selection dialog. Not the 'Cancel' button."
    ),
    vertical_pad_ratio=0.8,
    vertical_pad_min_px=12,
    min_crop_height=56,
)

__all__ = ["REQUEST_BUTTON_TARGET", "SHARE_SCREEN_TARGET"]
```

- [ ] **Step 4: 실행 경로 구현**

`poc/workflow_3/monitor/share_request.py` 에 추가한다. 상단에
`from dataclasses import dataclass` 를 넣는다.

```python
STATUS_REQUESTED = "requested"
STATUS_CONFIRM_FAILED = "confirm_failed"
STATUS_NOT_FOUND = "not_found"
STATUS_BLOCKED_SAFE_MODE = "blocked_safe_mode"
STATUS_ERROR = "error"


@dataclass
class ShareRequestResult:
    """공유 요청 시도 결과. 판정을 함께 실어 오피스 진단에 쓴다."""

    status: str
    radio_verdict: str = ""
    button_verdict: str = ""
    error: str = ""


def request_screen_share(
    settings,
    *,
    locate_fn,
    read_tokens_fn,
    click_fn,
    capture_fn,
    find_popup_fn,
) -> ShareRequestResult:
    """'select' 팝업에서 화면 공유를 선택하고 Request 를 누른다.

    **두 라벨을 모두 확인한 뒤에야 클릭을 시작한다.** 라디오를 먼저 누르고 버튼을
    확인하면, 버튼 확인이 실패했을 때 이미 남의 팝업 상태를 바꿔 놓은 뒤가 된다.

    협력자는 전부 주입받는다 - 실장비/VLM 없이 판정 로직을 시험하기 위해서다.
      locate_fn(image, target) -> point dict | None
      read_tokens_fn(image, box, label) -> list[str]
      click_fn(x, y, label) -> None
      capture_fn(window) -> image
      find_popup_fn() -> window | None
    """
    from poc.workflow_3.vlm.prompts.prompt_share_options import (
        REQUEST_BUTTON_TARGET,
        SHARE_SCREEN_TARGET,
    )

    policy = getattr(settings, "share_confirm_policy", CONFIRM_STRICT)
    try:
        popup = find_popup_fn()
        if popup is None:
            return ShareRequestResult(status=STATUS_NOT_FOUND)

        if not getattr(settings, "action_enabled", False):
            print("[INFO] SAFE_MODE - 공유 요청 클릭 차단(판정까지만 수행)")
            return ShareRequestResult(status=STATUS_BLOCKED_SAFE_MODE)

        image = capture_fn(popup)

        plan = []
        verdicts = {}
        for target, required, forbidden in (
            (SHARE_SCREEN_TARGET, SHARE_SCREEN_REQUIRED, SHARE_SCREEN_FORBIDDEN),
            (REQUEST_BUTTON_TARGET, REQUEST_BTN_REQUIRED, REQUEST_BTN_FORBIDDEN),
        ):
            point = locate_fn(image, target)
            if point is None:
                print(f"[WARNING] 공유 팝업 요소 좌표 미검출: {target.key}")
                verdicts[target.key] = "unreadable"
                return ShareRequestResult(
                    status=STATUS_CONFIRM_FAILED,
                    radio_verdict=verdicts.get(SHARE_SCREEN_TARGET.key, ""),
                    button_verdict=verdicts.get(REQUEST_BUTTON_TARGET.key, ""),
                )
            tokens = read_tokens_fn(image, point, target.key)
            verdict = classify_label(tokens, required, forbidden)
            verdicts[target.key] = verdict
            if not accepts_label(verdict, policy):
                print(
                    f"[WARNING] 공유 팝업 라벨 확인 실패 - 클릭하지 않음: "
                    f"{target.key} verdict={verdict} policy={policy} tokens={tokens!r}"
                )
                return ShareRequestResult(
                    status=STATUS_CONFIRM_FAILED,
                    radio_verdict=verdicts.get(SHARE_SCREEN_TARGET.key, ""),
                    button_verdict=verdicts.get(REQUEST_BUTTON_TARGET.key, ""),
                )
            plan.append((point, target.key))

        for point, label in plan:
            click_fn(int(point["x"]), int(point["y"]), label)
        print("[INFO] 화면 공유 요청 발송 완료 - 상대 승낙 대기")
        return ShareRequestResult(
            status=STATUS_REQUESTED,
            radio_verdict=verdicts.get(SHARE_SCREEN_TARGET.key, ""),
            button_verdict=verdicts.get(REQUEST_BUTTON_TARGET.key, ""),
        )
    except Exception as exc:
        # actuator 는 예외를 삼키지 않는다 - 조용한 성공을 만들지 않기 위해서다.
        print(f"[ERROR] 공유 요청 중 예외: {exc}")
        return ShareRequestResult(status=STATUS_ERROR, error=str(exc))
```

`__all__` 에 새 이름들을 추가한다.

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_request.py -v`
Expected: PASS (31 passed)

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/monitor/share_request.py poc/workflow_3/vlm/prompts/prompt_share_options.py poc/workflow_3/monitor/test_share_request.py
git commit -m "feat(share): 확인 게이트 통과 후에만 라디오+Request 클릭"
```

---

## Task 8: cycle 배선

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py:348-420` (`_exec_wait_tool_window`),
  `:455-480` (`_exec_run_correction` 근처)
- Test: `poc/workflow_3/monitor/test_share_cycle_wiring.py`

**Interfaces:**
- Consumes: Task 3 상수, Task 5 상수, Task 7 `request_screen_share`
- Produces: `context["occupancy"]` ∈ `{occupied_by_other, free, unknown}`

- [ ] **Step 1: 실패하는 테스트 작성**

`poc/workflow_3/monitor/test_share_cycle_wiring.py` 에 추가한다.

```python
from poc.workflow_3.monitor.cycle import resolve_correction_outcome_status
from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER, UNKNOWN


def test_free_keeps_original_status():
    assert resolve_correction_outcome_status(FREE, "corrected") == "corrected"


def test_unknown_downgrades_corrected():
    """반영 여부를 보장 못 하므로 성공으로 보고하지 않는다."""
    assert (
        resolve_correction_outcome_status(UNKNOWN, "corrected")
        == CORRECTED_UNVERIFIED
    )


def test_unknown_leaves_non_corrected_alone():
    """이미 실패/인계 경로면 그대로 둔다 - 정보를 덮어쓰지 않는다."""
    assert (
        resolve_correction_outcome_status(UNKNOWN, "awaiting_engineer_ok")
        == "awaiting_engineer_ok"
    )


def test_occupied_is_view_only_regardless_of_input():
    assert (
        resolve_correction_outcome_status(OCCUPIED_BY_OTHER, "corrected")
        == VIEW_ONLY_OBSERVATION
    )
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_correction_outcome_status'`

- [ ] **Step 3: 최소 구현**

`poc/workflow_3/monitor/cycle.py` 에 순수 함수를 추가한다 (`_exec_run_correction` 위).

```python
def resolve_correction_outcome_status(occupancy: str, status: str) -> str:
    """점유 상태를 반영해 최종 outcome status 를 정한다.

    occupied_by_other  : 보정 자체를 건너뛰었으므로 관전 status.
    unknown + corrected: 클릭이 먹었는지 확인할 수 없다. correct_align_fail_auto 는
                         open-loop 라 반영 여부를 되읽지 않으므로, 'corrected' 로 두면
                         cube 가 생략되어 아무도 모르는 미보정이 남는다.
    그 외              : 그대로 둔다(실패/인계 경로의 정보를 덮어쓰지 않는다).
    """
    from poc.workflow_3.rcs.row_occupant import OCCUPIED_BY_OTHER, UNKNOWN

    if occupancy == OCCUPIED_BY_OTHER:
        return VIEW_ONLY_OBSERVATION
    if occupancy == UNKNOWN and status == "corrected":
        return CORRECTED_UNVERIFIED
    return status
```

`cycle.py` 상단 임포트에 추가한다.

```python
from poc.workflow_3.monitor.notify import CORRECTED_UNVERIFIED, VIEW_ONLY_OBSERVATION
```

(이미 `notify` 에서 다른 이름을 가져오고 있으면 그 블록에 합친다.)

`_exec_run_correction` 안에서, 보정 호출 **전에** 점유를 확인하고, **후에** status 를
치환한다.

```python
    occupancy = context.get("occupancy", UNKNOWN)
    if occupancy == OCCUPIED_BY_OTHER:
        # view-only 세션에서는 클릭이 장비에 먹지 않는다. 시도하지 않고 관전만 한다.
        print("[INFO] 다른 엔지니어 점유 중 - 보정 건너뜀, 관전·녹화만 수행")
        context["outcome"] = CorrectionOutcome(
            status=VIEW_ONLY_OBSERVATION,
            path="observation",
            key_decision="",
            best_xy=None,
            ok_screen_xy=None,
            fallback=None,
        )
        return _make_result(step, "success", started_at, settings)
```

보정이 끝나고 `context["outcome"] = outcome` 을 넣기 직전에 status 를 치환한다.

```python
    outcome.status = resolve_correction_outcome_status(occupancy, outcome.status)
```

`_exec_wait_tool_window` 의 점유 분기(385-393행)를 확장한다. `occupied["select"]` 가
참일 때 기존의 즉시 실패 대신 아래를 수행한다.

```python
    if occupied["select"]:
        if not settings.share_request_enabled:
            return _make_result(
                step, "failed", started_at, settings,
                failure_class="rcs_occupied_select",
                error_message="점유 'select' 팝업 감지 - 공유 요청 비활성(설정)",
            )

        from poc.workflow_3.monitor.share_request import (
            ACCEPTED,
            STATUS_CONFIRM_FAILED,
            STATUS_REQUESTED,
            request_screen_share,
            wait_share_response,
        )

        share = _run_share_request(settings, popup_client)
        if share.status == STATUS_REQUESTED:
            responded = wait_share_response(
                eqp_id, settings.share_wait_sec,
                find_window_fn=lambda tool: _find_tool_window(tool),
            )
            if responded == ACCEPTED:
                context["occupancy"] = OCCUPIED_BY_OTHER
                window, title, backend = wait_for_remote_monitoring_window(
                    eqp_id, max_attempts=settings.rcs_window_max_trials,
                )
                if window is not None:
                    context["tool_window"] = window
                    context["tool_window_title"] = title
                    context["tool_window_backend"] = backend
                    return _make_result(step, "success", started_at, settings)
        _close_select_popup()
        failure_class = (
            "rcs_share_confirm_failed"
            if share.status == STATUS_CONFIRM_FAILED
            else "rcs_occupied_select"
        )
        return _make_result(
            step, "failed", started_at, settings,
            failure_class=failure_class,
            error_message=(
                f"화면 공유 요청 결과: {share.status} "
                f"(radio={share.radio_verdict or '-'}, button={share.button_verdict or '-'})"
            ),
        )
```

보조 함수 3개를 `cycle.py` 에 추가한다. `_run_share_request` 는 Task 7 의 주입점을 실제
구현으로 채우고, `_close_select_popup` 은 **좌표 클릭이 아니라 `close_window()`** 로 닫는다.

```python
def _find_tool_window(eqp_id: str):
    """제목에 eqp_id 를 가진 Remote Monitoring 창을 1회 탐색한다 (없으면 None)."""
    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window

    window, _title, _backend = find_remote_monitoring_window(eqp_id)
    return window


def _close_select_popup() -> None:
    """점유 'select' 팝업을 닫는다 - 좌표 클릭이 아니라 창 핸들로.

    Cancel 을 좌표로 누르는 것은 로케이션이 방금 실패했을 수도 있는 시점에 같은 팝업을
    다시 겨냥하는 것이라, 확인 게이트를 통과하지 않은 클릭을 최악의 순간에 내보내게 된다.
    """
    from poc.workflow_3.monitor.occupied_popup import SELECT_TITLE
    from poc.workflow_3.util import close_window, find_window_by_title_prefix

    try:
        popup = find_window_by_title_prefix(SELECT_TITLE)
        if popup is not None and close_window is not None:
            close_window(popup, debug_label="select popup")
    except Exception as exc:
        print(f"[WARNING] select 팝업 닫기 실패(수동 확인 필요): {exc}")


def _run_share_request(settings: Workflow3Settings, popup_client):
    """share_request 의 주입점을 실제 VLM/OCR/클릭 구현으로 채워 호출한다."""
    from poc.workflow_3.monitor.occupied_popup import SELECT_TITLE
    from poc.workflow_3.monitor.share_request import request_screen_share
    from poc.workflow_3.rcs.row_occupant import build_occupant_box  # noqa: F401
    from poc.workflow_3.util import capture_window, find_window_by_title_prefix
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import analyze_window_target

    debug_dir = DEBUG_IMAGE_DIR / "share_request"

    def _locate(image, target):
        result = analyze_window_target(
            None, "select", "uia", target,
            debug_image_dir=debug_dir,
            log_name="share_request",
            component_name="share_request",
            artifact_prefix=target.key,
            image=image,
        )
        return result.point

    def _read_tokens(image, point, label):
        box = crop_box_around_point(
            point, image.width, image.height,
            left_ratio=0.30, right_ratio=0.30, half_height_ratio=0.06,
        )
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(time.time()),
            artifact_label=label,
            log_name="share_request",
        )
        return tokens_from_text(read.raw_text) if read.ok else []

    def _click(x, y, label):
        click_at(x, y, debug_label=f"share_request {label}")

    return request_screen_share(
        settings,
        locate_fn=_locate,
        read_tokens_fn=_read_tokens,
        click_fn=_click,
        capture_fn=capture_window,
        find_popup_fn=lambda: find_window_by_title_prefix(SELECT_TITLE),
    )
```

`click_at` / `make_timestamp_tag` / `DEBUG_IMAGE_DIR` 이 `cycle.py` 에 이미 임포트되어
있는지 확인하고, 없으면 기존 임포트 블록에 추가한다 (`poc.workflow_3.util` 및
`poc.workflow_3` 에서 가져온다).

정상 경로(팝업 없음)에서는 `_exec_connect_tool` 이 행 좌표를 확정한 직후
`context["occupancy"]` 를 채운다. `select_tool_from_main_window` 결과에서 행 좌표를 얻어
`build_occupant_box` + `read_text_near_point` + `classify_occupancy` 를 호출하고, 예외는
`UNKNOWN` 으로 흡수한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: 임포트 무결성 확인**

Run: `uv run python -c "import poc.workflow_3.monitor.cycle; import poc.workflow_3.monitor.align_fail_monitor; print('ok')"`
Expected: `ok`

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_share_cycle_wiring.py
git commit -m "feat(cycle): 점유 시 공유 요청 분기 + occupancy 기반 보정 라우팅"
```

---

## Task 9: 전체 회귀 + 문서

**Files:**
- Modify: `poc/workflow_3/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 전체 테스트 실행**

```bash
uv run pytest poc/workflow_3/monitor/test_share_request.py \
              poc/workflow_3/monitor/test_share_cycle_wiring.py \
              poc/workflow_3/rcs/test_row_occupant.py -v
uv run python poc/workflow_3/rcs/test_tool_row_verify.py
uv run python poc/workflow_3/monitor/test_notify.py
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
uv run python poc/workflow_3/align/test_correction.py
```

Expected: 전부 통과. 하나라도 실패하면 해당 태스크로 돌아간다.

- [ ] **Step 2: README 갱신**

`poc/workflow_3/README.md` 의 env 표에 4개를 추가하고, 점유 동작 설명을
"검출 후 포기" 에서 "화면 공유 요청 후 녹화" 로 고친다. 오피스 확인 항목 3가지
(거절 시 화면, 점유자 컬럼 기하, 실제 OCR 문구)를 함께 적는다.

- [ ] **Step 3: CLAUDE.md 갱신**

`monitor/` 설명에 `share_request.py`, `rcs/` 설명에 `row_occupant.py` 를 한 줄씩 추가한다.
`ALIGN_FAIL_SHARE_*` env 네임스페이스를 언급한다.

- [ ] **Step 4: 커밋**

```bash
git add poc/workflow_3/README.md CLAUDE.md
git commit -m "docs: 점유 tool 화면 공유 요청 기능 문서화"
```

---

## Self-Review

**Spec 커버리지**

| Spec 절 | 태스크 |
|---|---|
| ① actuator + 확인 게이트 | 2, 7 |
| ② 승낙/거절 판별 + close_window | 4, 8 |
| ③ 점유 3-상태 + 전용 crop | 3, 8 |
| ④ cycle 흐름 + status 치환 | 8 |
| ⑤ 알림 문구 (생략 없음) | 5 |
| ⑥ 재시도 집합 + 시도 상한 | 6 |
| ⑦ env 4개 | 1 |
| 오류 처리 표 | 2, 4, 7, 8 |
| 진단 산출물 | 7 (`ShareRequestResult` 판정 + `read_text_near_point` 의 crop 저장) |
| 테스트 목록 | 2, 3, 4, 5, 6, 7, 8 |

**타입 일관성**

- `classify_label` / `accepts_label` 는 Task 2 정의, Task 7 사용 — 이름·인자 일치.
- `wait_share_response` 는 Task 4 정의, Task 8 사용 — 키워드 인자 `find_window_fn` 일치.
- `OCCUPIED_BY_OTHER`/`FREE`/`UNKNOWN` 은 Task 3 정의, Task 8 사용.
- `VIEW_ONLY_OBSERVATION`/`CORRECTED_UNVERIFIED` 는 Task 5 정의, Task 6·8 사용.
- `ShareRequestResult.status` 값은 Task 7 의 `STATUS_*` 상수만 쓴다.
