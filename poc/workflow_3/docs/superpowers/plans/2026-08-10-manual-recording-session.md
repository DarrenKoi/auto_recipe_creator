# 엔지니어 수동 조작 녹화 세션 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 알람 없이 이미 열린 Remote Monitoring 창에 붙어 엔지니어의 수동 조작을 녹화하고, 그 프레임에서 "무엇을 클릭했는가"까지 뽑아내는 파이프라인을 만든다.

**Architecture:** 두 부분이다. (1) `monitor/manual_record.py` 런처 — 창을 찾아 `RecordingSession`(기존, **무수정**)을 `capture_fn` 주입점으로 감싸 프레임 + 사이드카 메타를 남기고, 시간/프레임/디스크 예산을 감시한다. (2) `recording_filter/` 확장 — 영역 게이트(Stage 1.5, VLM 0콜)로 라이브 SEM 영상의 자율 변화를 걸러낸 뒤, 살아남은 소수 프레임만 기존 클릭 판정(Stage 2a)과 신규 요소 라벨링(Stage 2c)에 태운다.

**Tech Stack:** Python >= 3.10, uv, PIL, numpy, cv2, pytest. VLM 은 Flask 프록시 경유(`mai-ui` grounding, `paddleocr-vl-1.5` OCR). Windows 전용 부분은 `ctypes` + pywinauto wrapper.

설계 근거: `poc/workflow_3/docs/superpowers/specs/2026-08-10-manual-recording-session-design.md`

## Global Constraints

이 저장소의 규약이며 **모든 태스크에 암묵적으로 적용된다.**

- **Korean docstrings** — 모든 모듈/함수 docstring 은 한국어로 쓴다.
- **Print-based logging** — `[INFO]` / `[ERROR]` / `[WARNING]` 접두. `logging` 모듈 금지.
- **`print()` 문자열에 em-dash(U+2014) 금지** — 오피스 콘솔이 cp949 라 인코딩 실패한다. docstring 에는 써도 된다.
- **`from __future__ import ...` 금지** — 명시 요청이 없는 한 추가하지 않는다.
- **절대 임포트** — `from poc.workflow_3.xxx import ...`.
- **CLI 인자 금지** — `argparse`/플래그 없음. 설정은 모듈 상수 + env 오버라이드. 스크립트는 `uv run python <script>.py` 만으로 실행돼야 한다.
- **이미지 포맷** — 로컬 저장은 JPEG, VLM 전송은 WebP(quality=90).
- **VLM service 는 route slug** — `Workflow1VLMClient("mai-ui")` 가 맞고 `"mai-ui-8b"` 는 틀리다.
- **`RecordingSession` (`monitor/recording.py`) 은 수정하지 않는다** — 감싸기만 한다. 알람 사이클 동작에 영향이 가면 안 된다.
- **테스트 실행 방식이 패키지마다 다르다** — `recording_filter/` 는 pytest (`uv run pytest poc/workflow_3/recording_filter`), `monitor/` 는 직접 실행 스크립트 (`uv run python poc/workflow_3/monitor/test_xxx.py`, 파일 끝에 `if __name__ == "__main__":` 로 전체 호출). 각 태스크가 지시하는 쪽을 따른다.
- **Windows 전용 코드는 Mac 에서 import 가능해야 한다** — `os.name != "nt"` 조기 반환 또는 try/except ImportError + `AVAILABLE` 플래그 패턴. Mac 에서 테스트가 돌아야 하기 때문이다.
- **커밋은 pathspec 으로** — 병렬 세션이 같은 저장소를 편집하므로 `git add -A` / `commit -a` 금지. 건드린 파일만 명시한다.

## File Structure

| 파일 | 책임 |
|------|------|
| `poc/workflow_3/monitor/manual_record.py` (신규) | 런처. 창 탐색 → EQP 추출 → 세션 시작 → 예산 감시 → 종료 안내 |
| `poc/workflow_3/monitor/frame_meta.py` (신규) | 사이드카 메타 기록기 + 가림 판정. 런처가 `capture_fn` 을 감쌀 때 쓰는 부품 |
| `poc/workflow_3/monitor/test_manual_record.py` (신규) | 위 두 모듈의 직접 실행 테스트 |
| `poc/workflow_3/debug_artifacts.py` (수정) | `save_debug_jpeg` 에 `quality` 인자 추가 (기본값 95 유지) |
| `poc/workflow_3/recording_filter/region_gate.py` (신규) | 레이아웃 세대 관리 + 영역 지도 + ambient/candidate 게이팅 |
| `poc/workflow_3/recording_filter/element_label.py` (신규) | 클릭 지점 crop → OCR → (실패 시) VLM 라벨링 |
| `poc/workflow_3/recording_filter/test_region_gate.py` (신규) | pytest |
| `poc/workflow_3/recording_filter/test_element_label.py` (신규) | pytest |
| `poc/workflow_3/recording_filter/timeline.py` (수정) | 이벤트 스키마에 element/target_kind/region/generation/occlusion 추가 |
| `poc/workflow_3/recording_filter/settings.py` (수정) | Stage 1.5 / 2c 튜닝 필드 추가 |
| `poc/workflow_3/recording_filter/filter_recording.py` (수정) | 신규 스테이지 배선 + `_manual` glob 추가 |
| `poc/workflow_3/README.md` (수정) | 수동 녹화 세션 사용법 |

---

### Task 1: EQP 제목 파싱 + 세션 경로

런처의 순수 함수부. Windows 없이 Mac 에서 전부 검증된다.

**Files:**
- Create: `poc/workflow_3/monitor/manual_record.py`
- Create: `poc/workflow_3/monitor/test_manual_record.py`

**Interfaces:**
- Consumes: `poc.workflow_3.ALIGN_IMAGES_DIR` (Path), `poc.workflow_3.rcs.login_rcs_common.REMOTE_MONITORING_WINDOW_TITLE_PREFIX` (str = `"Remote Monitoring System -"`)
- Produces:
  - `parse_eqp_from_title(title: str) -> str` — 제목에서 EQP 추출, 실패 시 `""`
  - `sanitize_eqp_for_path(eqp: str) -> str` — 폴더명 안전 문자열, 빈 입력이면 `"unknown_eqp"`
  - `manual_recording_dir(eqp_id: str, tag: str) -> Path` — `ALIGN_IMAGES_DIR/<eqp>/_manual/<tag>/recording`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/monitor/test_manual_record.py` 를 새로 만든다.

```python
"""수동 녹화 런처 단위 테스트 - RCS/Windows 없이 Mac 에서 돈다.

`uv run python poc/workflow_3/monitor/test_manual_record.py` 로 직접 실행.
"""

from poc.workflow_3.monitor.manual_record import (
    manual_recording_dir,
    parse_eqp_from_title,
    sanitize_eqp_for_path,
)


def test_parse_eqp_from_plain_title():
    """표준 제목에서 EQP 를 뽑는다."""
    assert parse_eqp_from_title("Remote Monitoring System - MCD916") == "MCD916"
    print("[OK] test_parse_eqp_from_plain_title")


def test_parse_eqp_strips_surrounding_whitespace():
    """접두어 뒤 공백은 제거된다."""
    assert parse_eqp_from_title("Remote Monitoring System -   MCD916  ") == "MCD916"
    print("[OK] test_parse_eqp_strips_surrounding_whitespace")


def test_parse_eqp_keeps_trailing_tokens():
    """EQP 뒤에 부가 정보가 붙어도 통째로 보존한다(정규화는 sanitize 담당)."""
    assert parse_eqp_from_title("Remote Monitoring System - MCD916 (Online)") == "MCD916 (Online)"
    print("[OK] test_parse_eqp_keeps_trailing_tokens")


def test_parse_eqp_returns_empty_for_prefix_only():
    """접두어만 있고 EQP 가 없으면 빈 문자열."""
    assert parse_eqp_from_title("Remote Monitoring System -") == ""
    assert parse_eqp_from_title("Remote Monitoring System - ") == ""
    print("[OK] test_parse_eqp_returns_empty_for_prefix_only")


def test_parse_eqp_returns_empty_for_other_window():
    """다른 창 제목이면 빈 문자열."""
    assert parse_eqp_from_title("RCS - Main") == ""
    assert parse_eqp_from_title("") == ""
    print("[OK] test_parse_eqp_returns_empty_for_other_window")


def test_parse_eqp_is_case_insensitive_on_prefix():
    """접두어 대소문자는 무시한다(창 제목 표기 흔들림 대비)."""
    assert parse_eqp_from_title("REMOTE MONITORING SYSTEM - MCD916") == "MCD916"
    print("[OK] test_parse_eqp_is_case_insensitive_on_prefix")


def test_sanitize_replaces_path_hostile_chars():
    """폴더명에 못 쓰는 문자는 밑줄로 바꾼다."""
    assert sanitize_eqp_for_path("MCD916 (Online)") == "MCD916_Online"
    assert sanitize_eqp_for_path("A/B:C*D") == "A_B_C_D"
    print("[OK] test_sanitize_replaces_path_hostile_chars")


def test_sanitize_falls_back_for_empty():
    """빈 입력은 unknown_eqp 로 떨어진다 - 프레임을 잃지 않기 위해서."""
    assert sanitize_eqp_for_path("") == "unknown_eqp"
    assert sanitize_eqp_for_path("   ") == "unknown_eqp"
    assert sanitize_eqp_for_path("///") == "unknown_eqp"
    print("[OK] test_sanitize_falls_back_for_empty")


def test_manual_recording_dir_shape():
    """경로는 <root>/<eqp>/_manual/<tag>/recording 형태다."""
    from poc.workflow_3 import ALIGN_IMAGES_DIR

    path = manual_recording_dir("MCD916", "20260810_140000")
    assert path == ALIGN_IMAGES_DIR / "MCD916" / "_manual" / "20260810_140000" / "recording", path
    print("[OK] test_manual_recording_dir_shape")


if __name__ == "__main__":
    test_parse_eqp_from_plain_title()
    test_parse_eqp_strips_surrounding_whitespace()
    test_parse_eqp_keeps_trailing_tokens()
    test_parse_eqp_returns_empty_for_prefix_only()
    test_parse_eqp_returns_empty_for_other_window()
    test_parse_eqp_is_case_insensitive_on_prefix()
    test_sanitize_replaces_path_hostile_chars()
    test_sanitize_falls_back_for_empty()
    test_manual_recording_dir_shape()
    print("\n[OK] manual_record 파싱/경로 테스트 통과")
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.monitor.manual_record'`

- [ ] **Step 3: 최소 구현을 쓴다**

`poc/workflow_3/monitor/manual_record.py` 를 새로 만든다.

```python
"""엔지니어 수동 조작 녹화 런처 - 알람 없이 이미 열린 tool 창을 녹화한다.

알람 사이클(`monitor/cycle.py`)의 녹화는 align fail 이 떠야만 시작된다. 이 모듈은
엔지니어와 "지금부터 녹화하겠다"고 약속한 뒤, 이미 열려 있는 Remote Monitoring 창을
그 자리에서 녹화하기 위한 독립 진입점이다. 접속(tool 더블클릭)은 하지 않는다.

수집한 프레임은 모방 학습/절차 분석의 원천 데이터가 되며, 분석은 별도 실행이다
(`recording_filter/filter_recording.py`).

실행:
    uv run python poc/workflow_3/monitor/manual_record.py
"""

import re

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.rcs.login_rcs_common import REMOTE_MONITORING_WINDOW_TITLE_PREFIX

# 폴더명으로 쓸 수 없는 문자(Windows 예약 문자 + 공백/괄호)를 밑줄로 바꾼다.
_PATH_HOSTILE_RE = re.compile(r"[^A-Za-z0-9._-]+")
# EQP 를 못 읽었을 때의 대체 폴더명 - 프레임을 잃는 것보다 낫다.
UNKNOWN_EQP = "unknown_eqp"
# 수동 세션 전용 하위 폴더명 (알람 캡처의 captured_img_from_rcs 와 구분).
MANUAL_DIRNAME = "_manual"


def parse_eqp_from_title(title) -> str:
    """창 제목에서 EQP 문자열을 추출한다(접두어 제거). 실패하면 빈 문자열.

    제목은 "Remote Monitoring System - <EQP>" 형태다. 접두어 매칭은 대소문자를
    무시하고, EQP 뒤에 부가 정보가 붙어 있으면 통째로 보존한다(폴더명 정규화는
    sanitize_eqp_for_path 의 몫이라 여기서는 자르지 않는다).
    """
    normalized = (title or "").strip()
    prefix = REMOTE_MONITORING_WINDOW_TITLE_PREFIX
    if len(normalized) < len(prefix):
        return ""
    if normalized[: len(prefix)].lower() != prefix.lower():
        return ""
    return normalized[len(prefix):].strip()


def sanitize_eqp_for_path(eqp) -> str:
    """EQP 문자열을 폴더명으로 안전한 형태로 바꾼다. 비면 UNKNOWN_EQP."""
    cleaned = _PATH_HOSTILE_RE.sub("_", (eqp or "").strip()).strip("_")
    return cleaned or UNKNOWN_EQP


def manual_recording_dir(eqp_id, tag):
    """수동 세션 프레임 저장 폴더 - <root>/<eqp>/_manual/<tag>/recording."""
    return ALIGN_IMAGES_DIR / sanitize_eqp_for_path(eqp_id) / MANUAL_DIRNAME / str(tag) / "recording"
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: PASS — 9개 `[OK]` 출력 후 `[OK] manual_record 파싱/경로 테스트 통과`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/manual_record.py poc/workflow_3/monitor/test_manual_record.py
git commit -m "feat(workflow_3): 수동 녹화 런처의 EQP 제목 파싱 + 세션 경로"
```

---

### Task 2: 프레임 메타 사이드카 (창 rect / 가림 / 커서)

프레임마다 1줄 JSONL 을 남기는 부품. 가림 판정의 **분류 로직만 순수 함수로 분리**해 Mac 에서 검증하고, `WindowFromPoint` 호출부는 Windows 전용 얇은 층으로 둔다.

**Files:**
- Create: `poc/workflow_3/monitor/frame_meta.py`
- Modify: `poc/workflow_3/monitor/test_manual_record.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1 의 모듈은 쓰지 않는다(독립)
- Produces:
  - `classify_occlusion(hit_handles: list, our_handles: set) -> str` — `"none"` | `"partial"` | `"full"`
  - `probe_points(rect: dict) -> list[tuple[int, int]]` — 중앙 + 사분면 5점 (화면 좌표)
  - `FrameMetaWriter(out_dir: Path)` — `.append(record: dict) -> None`, `.close() -> None`
  - `build_meta_record(frame_name, t_sec, rect, foreground_title, occlusion, cursor_xy) -> dict`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/monitor/test_manual_record.py` 상단 import 아래에 추가한다.

```python
import json

from poc.workflow_3.monitor.frame_meta import (
    FrameMetaWriter,
    build_meta_record,
    classify_occlusion,
    probe_points,
)


def test_classify_occlusion_none_when_all_hits_are_ours():
    """5점 모두 우리 창이면 가림 없음."""
    assert classify_occlusion([10, 10, 10, 10, 10], {10}) == "none"
    print("[OK] test_classify_occlusion_none_when_all_hits_are_ours")


def test_classify_occlusion_full_when_no_hit_is_ours():
    """한 점도 우리 창이 아니면 완전히 가려진 것."""
    assert classify_occlusion([99, 99, 99, 99, 99], {10}) == "full"
    print("[OK] test_classify_occlusion_full_when_no_hit_is_ours")


def test_classify_occlusion_partial_when_mixed():
    """일부만 우리 창이면 부분 가림 - 포커스를 안 뺏은 겹침이 여기 잡힌다."""
    assert classify_occlusion([10, 99, 10, 10, 99], {10}) == "partial"
    print("[OK] test_classify_occlusion_partial_when_mixed")


def test_classify_occlusion_accepts_child_handles():
    """자식 컨트롤 핸들도 우리 창으로 친다(our_handles 에 여러 개 허용)."""
    assert classify_occlusion([10, 11, 12, 10, 11], {10, 11, 12}) == "none"
    print("[OK] test_classify_occlusion_accepts_child_handles")


def test_classify_occlusion_unknown_when_no_hits():
    """조회 자체가 실패해 표본이 없으면 판정하지 않는다."""
    assert classify_occlusion([], {10}) == "unknown"
    print("[OK] test_classify_occlusion_unknown_when_no_hits")


def test_probe_points_are_inside_rect():
    """5개 표본점은 모두 rect 내부다(경계 포함 안 함)."""
    rect = {"left": 100, "top": 50, "right": 500, "bottom": 250}
    points = probe_points(rect)
    assert len(points) == 5, points
    for x, y in points:
        assert rect["left"] < x < rect["right"], (x, rect)
        assert rect["top"] < y < rect["bottom"], (y, rect)
    # 중앙점이 포함된다.
    assert (300, 150) in points, points
    print("[OK] test_probe_points_are_inside_rect")


def test_probe_points_handles_tiny_rect():
    """1픽셀 창에서도 터지지 않는다(클램프)."""
    points = probe_points({"left": 0, "top": 0, "right": 1, "bottom": 1})
    assert len(points) == 5, points
    print("[OK] test_probe_points_handles_tiny_rect")


def test_meta_writer_appends_one_json_per_line(tmp_path=None):
    """프레임당 정확히 1줄 JSON 이 append 된다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        writer = FrameMetaWriter(out_dir)
        writer.append(build_meta_record(
            frame_name="rec_rcs_0000_00000000ms.jpg", t_sec=0.0,
            rect={"left": 0, "top": 0, "right": 100, "bottom": 100},
            foreground_title="Remote Monitoring System - MCD916",
            occlusion="none", cursor_xy=(10, 20),
        ))
        writer.append(build_meta_record(
            frame_name="rec_rcs_0001_00000200ms.jpg", t_sec=0.2,
            rect={"left": 0, "top": 0, "right": 100, "bottom": 100},
            foreground_title="Notepad", occlusion="partial", cursor_xy=None,
        ))
        writer.close()

        lines = (out_dir / "frame_meta.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2, lines
        first = json.loads(lines[0])
        assert first["frame"] == "rec_rcs_0000_00000000ms.jpg"
        assert first["occlusion"] == "none"
        assert first["cursor_screen_xy"] == [10, 20]
        assert first["cursor_in_window"] is True
        second = json.loads(lines[1])
        assert second["cursor_screen_xy"] is None
        assert second["cursor_in_window"] is False
    print("[OK] test_meta_writer_appends_one_json_per_line")


def test_meta_writer_survives_write_failure():
    """기록 실패는 경고만 하고 삼킨다 - 프레임 손실보다 나쁜 건 없다."""
    from pathlib import Path

    writer = FrameMetaWriter(Path("/nonexistent-root/definitely/not/writable"))
    writer.append({"frame": "x.jpg"})   # 예외가 밖으로 나오면 안 된다.
    writer.close()
    print("[OK] test_meta_writer_survives_write_failure")
```

`__main__` 블록에도 위 함수 9개를 호출하도록 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.monitor.frame_meta'`

- [ ] **Step 3: 최소 구현을 쓴다**

`poc/workflow_3/monitor/frame_meta.py` 를 새로 만든다.

```python
"""수동 녹화 프레임의 사이드카 메타 기록 - 창 rect / 전면 창 / 가림 / 커서 좌표.

capture_window 는 창 핸들이 아니라 **창 rect 의 스크린 그랩**이라(util/image_utils.py),
다른 앱이 위에 뜨면 그 앱이 찍힌다. 수십 분 도는 수동 세션에서는 실제로 발생하므로,
프레임마다 "그때 이 창이 실제로 보이고 있었는가"를 같이 남겨 분석 단계에서 걸러낸다.

가림 판정은 픽셀이 아니라 기하다 - 창 영역 5개 지점에서 WindowFromPoint 를 찍어
그 지점의 최상위 창이 우리 창인지 본다. 포커스를 뺏지 않은 부분 겹침까지 잡힌다.

커서 좌표는 GetCursorPos 폴링이며 **입력 후킹이 아니다.** 키 입력은 기록하지 않는다.
"""

import ctypes
import json
import os
from pathlib import Path

# 사이드카 파일명 - 분석 단계가 frame 키로 프레임과 조인한다.
FRAME_META_FILENAME = "frame_meta.jsonl"
# 가림 표본점 수 (중앙 + 사분면).
_PROBE_COUNT = 5


def classify_occlusion(hit_handles, our_handles) -> str:
    """표본점의 최상위 창 핸들 목록으로 가림 정도를 판정한다.

    반환: "none"(전부 우리 창) | "partial"(일부) | "full"(하나도 아님) |
          "unknown"(표본 없음 - 조회 실패라 판정하지 않는다).
    """
    hits = [h for h in (hit_handles or []) if h is not None]
    if not hits:
        return "unknown"
    ours = sum(1 for h in hits if h in (our_handles or set()))
    if ours == len(hits):
        return "none"
    if ours == 0:
        return "full"
    return "partial"


def probe_points(rect) -> list:
    """창 rect 내부의 표본점 5개(중앙 + 사분면 중심)를 화면 좌표로 만든다.

    경계에 붙으면 이웃 창이 잡힐 수 있어 안쪽으로 넣는다. 아주 작은 창에서도
    좌표가 rect 밖으로 나가지 않도록 클램프한다.
    """
    left, top = int(rect["left"]), int(rect["top"])
    right, bottom = int(rect["right"]), int(rect["bottom"])
    width = max(1, right - left)
    height = max(1, bottom - top)

    def _clamp(value, low, high):
        return max(low, min(high, value))

    cx = left + width // 2
    cy = top + height // 2
    qx, qy = width // 4, height // 4
    raw = [
        (cx, cy),
        (cx - qx, cy - qy),
        (cx + qx, cy - qy),
        (cx - qx, cy + qy),
        (cx + qx, cy + qy),
    ]
    points = [
        (_clamp(x, left + 1, right - 1), _clamp(y, top + 1, bottom - 1))
        for x, y in raw
    ]
    return points[:_PROBE_COUNT]


def _win_point_type():
    """ctypes.wintypes.POINT 를 지연 import 로 얻는다(Mac 에서 import 실패 회피).

    `import ctypes` 만으로는 ctypes.wintypes 가 로드되지 않으며, 비 Windows 에서는
    wintypes import 자체가 실패한다. 그래서 Windows 분기 안에서만 가져온다.
    """
    from ctypes import wintypes

    return wintypes.POINT


def read_cursor_screen_xy():
    """현재 커서의 화면 좌표를 읽는다(입력 후킹 아님). 실패/비 Windows 는 None."""
    if os.name != "nt":
        return None
    try:
        point = _win_point_type()()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return (int(point.x), int(point.y))
    except Exception:
        return None
    return None


def probe_occlusion(rect, our_handles) -> str:
    """rect 표본점에서 WindowFromPoint 를 찍어 가림 정도를 판정한다."""
    if os.name != "nt" or not rect:
        return "unknown"
    try:
        user32 = ctypes.windll.user32
        point_type = _win_point_type()
        hits = []
        for x, y in probe_points(rect):
            hits.append(int(user32.WindowFromPoint(point_type(x, y))))
    except Exception as exc:
        print(f"[WARNING] 가림 판정 실패(unknown 으로 기록): {exc}")
        return "unknown"
    return classify_occlusion(hits, our_handles)


def build_meta_record(
    *, frame_name, t_sec, rect, foreground_title, occlusion, cursor_xy
) -> dict:
    """사이드카 1줄 레코드를 만든다(커서가 창 안인지도 함께 계산)."""
    cursor_list = [int(cursor_xy[0]), int(cursor_xy[1])] if cursor_xy else None
    in_window = False
    if cursor_list and rect:
        in_window = (
            int(rect["left"]) <= cursor_list[0] <= int(rect["right"])
            and int(rect["top"]) <= cursor_list[1] <= int(rect["bottom"])
        )
    return {
        "frame": frame_name,
        "t_sec": round(float(t_sec), 3),
        "window_rect": rect,
        "foreground_title": foreground_title or "",
        "occlusion": occlusion,
        "cursor_screen_xy": cursor_list,
        "cursor_in_window": bool(in_window),
    }


class FrameMetaWriter:
    """frame_meta.jsonl 에 프레임당 1줄을 append 한다(실패는 삼킨다)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / FRAME_META_FILENAME
        self._handle = None
        self._failed = False

    def _ensure_open(self):
        if self._handle is not None or self._failed:
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        except Exception as exc:
            self._failed = True
            print(f"[WARNING] frame_meta 기록 비활성화(열기 실패): {exc}")

    def append(self, record) -> None:
        """레코드 1건을 기록한다. 어떤 실패도 밖으로 던지지 않는다."""
        self._ensure_open()
        if self._handle is None:
            return
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        except Exception as exc:
            self._failed = True
            print(f"[WARNING] frame_meta 기록 실패(이후 생략): {exc}")

    def close(self) -> None:
        """파일 핸들을 닫는다(실패 무시)."""
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None
```

주의: `ctypes.wintypes` 는 Windows 밖에서 import 가 실패할 수 있다. 모듈 상단에서
`import ctypes` 만 하고 `wintypes` 접근은 `os.name == "nt"` 분기 안에서만 일어나도록
위 코드처럼 유지한다. Mac 에서 이 모듈 import 자체는 성공해야 한다.

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: PASS — Task 1 의 9개 + 신규 9개 모두 `[OK]`

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/frame_meta.py poc/workflow_3/monitor/test_manual_record.py
git commit -m "feat(workflow_3): 수동 녹화 프레임 사이드카 메타(가림 판정/커서 좌표)"
```

---

### Task 3: 런처 본체 - 창 탐색, 예산 감시, 세션 수명주기

**Files:**
- Modify: `poc/workflow_3/monitor/manual_record.py` (Task 1 파일에 추가)
- Modify: `poc/workflow_3/monitor/test_manual_record.py` (테스트 추가)
- Modify: `poc/workflow_3/debug_artifacts.py:31-35` (`save_debug_jpeg` 에 quality 인자)
- Modify: `poc/workflow_3/README.md` (사용법 절 추가)

**Interfaces:**
- Consumes: Task 1 의 `parse_eqp_from_title` / `manual_recording_dir`, Task 2 의 `FrameMetaWriter` / `build_meta_record` / `probe_occlusion` / `read_cursor_screen_xy`, `poc.workflow_3.monitor.recording.RecordingSession`
- Produces:
  - `ManualRecordSettings` (dataclass) + `load_manual_record_settings() -> ManualRecordSettings`
  - `budget_stop_reason(frame_count: int, disk_mb: float, settings) -> str` — 초과 사유 또는 `""`
  - `dir_size_mb(path: Path) -> float`
  - `pick_window_row(rows: list, wanted_eqp: str) -> tuple | None` — `rows` 는 `(title, handle)` 튜플 목록
  - `main() -> int`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/monitor/test_manual_record.py` 에 추가한다.

```python
from poc.workflow_3.monitor.manual_record import (
    ManualRecordSettings,
    budget_stop_reason,
    dir_size_mb,
    pick_window_row,
)


def test_budget_ok_when_under_all_limits():
    """상한 아래면 빈 문자열(계속 진행)."""
    s = ManualRecordSettings(max_frames=4000, max_disk_mb=2000)
    assert budget_stop_reason(100, 10.0, s) == ""
    print("[OK] test_budget_ok_when_under_all_limits")


def test_budget_stops_on_frame_limit():
    """프레임 상한 도달 시 frame_budget."""
    s = ManualRecordSettings(max_frames=100, max_disk_mb=2000)
    assert budget_stop_reason(100, 10.0, s) == "frame_budget"
    assert budget_stop_reason(101, 10.0, s) == "frame_budget"
    print("[OK] test_budget_stops_on_frame_limit")


def test_budget_stops_on_disk_limit():
    """디스크 상한 도달 시 disk_budget."""
    s = ManualRecordSettings(max_frames=4000, max_disk_mb=50)
    assert budget_stop_reason(10, 50.0, s) == "disk_budget"
    print("[OK] test_budget_stops_on_disk_limit")


def test_budget_frame_limit_wins_when_both_exceeded():
    """둘 다 넘으면 프레임을 먼저 보고한다(사유가 하나여야 manifest 가 명확)."""
    s = ManualRecordSettings(max_frames=10, max_disk_mb=10)
    assert budget_stop_reason(999, 999.0, s) == "frame_budget"
    print("[OK] test_budget_frame_limit_wins_when_both_exceeded")


def test_budget_zero_means_unlimited():
    """0 은 무제한 - max_sec 과 같은 규약."""
    s = ManualRecordSettings(max_frames=0, max_disk_mb=0)
    assert budget_stop_reason(10 ** 9, 10.0 ** 9, s) == ""
    print("[OK] test_budget_zero_means_unlimited")


def test_dir_size_mb_counts_files():
    """폴더 용량을 MB 로 센다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.jpg").write_bytes(b"x" * (1024 * 1024))
        (root / "b.jpg").write_bytes(b"x" * (1024 * 512))
        assert 1.4 < dir_size_mb(root) < 1.6, dir_size_mb(root)
    print("[OK] test_dir_size_mb_counts_files")


def test_pick_window_row_single_match():
    """모니터링 창이 하나면 그대로 채택한다."""
    rows = [("Remote Monitoring System - MCD916", 10)]
    assert pick_window_row(rows, "") == rows[0]
    print("[OK] test_pick_window_row_single_match")


def test_pick_window_row_none_when_empty():
    """하나도 없으면 None."""
    assert pick_window_row([], "") is None
    print("[OK] test_pick_window_row_none_when_empty")


def test_pick_window_row_requires_eqp_when_ambiguous():
    """여러 개인데 EQP 지정이 없으면 None - 임의 선택하지 않는다."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "") is None
    print("[OK] test_pick_window_row_requires_eqp_when_ambiguous")


def test_pick_window_row_disambiguates_by_eqp():
    """EQP 를 주면 그 창을 고른다(대소문자 무시)."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "mcd917") == rows[1]
    assert pick_window_row(rows, "MCD918") is None
    print("[OK] test_pick_window_row_disambiguates_by_eqp")
```

`__main__` 블록에 위 10개 호출을 추가한다.

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: FAIL — `ImportError: cannot import name 'ManualRecordSettings'`

- [ ] **Step 3: 설정 + 순수 함수를 구현한다**

`poc/workflow_3/monitor/manual_record.py` 에 추가한다 (Task 1 코드 아래).

```python
import time
from dataclasses import dataclass
from pathlib import Path

from poc.workflow_3.util import env_flag, env_float, env_int, make_timestamp_tag


@dataclass
class ManualRecordSettings:
    """수동 녹화 세션 파라미터 (CLI 인자 없음 - 모듈 상수 + env 오버라이드)."""

    max_sec: float = 600.0        # 실질 상한. 0 이면 무제한.
    max_frames: int = 4000        # 백스톱. 정상이면 안 걸린다. 0 이면 무제한.
    max_disk_mb: float = 2000.0   # 백스톱. 0 이면 무제한.
    poll_sec: float = 0.2         # 샘플링 요청 간격(실제 주기는 처리시간 + 이 값).
    heartbeat_sec: float = 5.0    # 변화가 없어도 이 간격마다 1장.
    change_min_px: int = 4        # 변화 판정 임계(알람 녹화와 동일).
    jpeg_quality: int = 85        # q95 대비 용량 약 절반.
    eqp_id: str = ""              # 모니터링 창이 여럿일 때만 필요.
    meta_enabled: bool = True     # 사이드카 기록 on/off.
    watch_interval_sec: float = 5.0   # 예산 감시 주기.


def load_manual_record_settings() -> ManualRecordSettings:
    """env 오버라이드를 적용한 설정을 만든다."""
    import os

    return ManualRecordSettings(
        max_sec=env_float("MANUAL_RECORD_MAX_SEC", 600.0),
        max_frames=env_int("MANUAL_RECORD_MAX_FRAMES", 4000),
        max_disk_mb=env_float("MANUAL_RECORD_MAX_DISK_MB", 2000.0),
        poll_sec=env_float("MANUAL_RECORD_POLL_SEC", 0.2),
        heartbeat_sec=env_float("MANUAL_RECORD_HEARTBEAT_SEC", 5.0),
        change_min_px=env_int("MANUAL_RECORD_CHANGE_MIN_PX", 4),
        jpeg_quality=env_int("MANUAL_RECORD_JPEG_QUALITY", 85),
        eqp_id=os.getenv("MANUAL_RECORD_EQP_ID", "").strip(),
        meta_enabled=env_flag("MANUAL_RECORD_META", True),
        watch_interval_sec=env_float("MANUAL_RECORD_WATCH_INTERVAL_SEC", 5.0),
    )


def budget_stop_reason(frame_count, disk_mb, settings) -> str:
    """프레임/디스크 예산 초과 사유를 돌려준다. 여유가 있으면 빈 문자열.

    사유는 하나만 돌려준다 - manifest 만 보고 원인을 구분할 수 있어야 하기 때문이다.
    0 은 무제한을 뜻하며 max_sec 규약과 같다.
    """
    if settings.max_frames > 0 and frame_count >= settings.max_frames:
        return "frame_budget"
    if settings.max_disk_mb > 0 and disk_mb >= settings.max_disk_mb:
        return "disk_budget"
    return ""


def dir_size_mb(path) -> float:
    """폴더 안 파일 용량 합계를 MB 로 돌려준다(실패는 0.0)."""
    total = 0
    try:
        for item in Path(path).rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        return 0.0
    return total / (1024.0 * 1024.0)


def pick_window_row(rows, wanted_eqp):
    """모니터링 창 후보 중 하나를 고른다.

    rows 는 (title, handle) 튜플 목록이다. 후보가 여럿인데 EQP 지정이 없으면
    None 을 돌려준다 - 엉뚱한 장비를 녹화하느니 다시 실행하는 편이 낫다.
    """
    if not rows:
        return None
    wanted = (wanted_eqp or "").strip().lower()
    if wanted:
        for row in rows:
            if wanted in (row[0] or "").lower():
                return row
        return None
    if len(rows) == 1:
        return rows[0]
    return None
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: PASS — 누적 28개 `[OK]`

- [ ] **Step 5: `save_debug_jpeg` 에 quality 인자를 추가한다**

`poc/workflow_3/debug_artifacts.py:31-35` 를 아래로 바꾼다. 기본값이 95 라 기존
호출부는 전부 그대로 동작한다.

```python
def save_debug_jpeg(image: "Image.Image", out_path: Path, *, quality: int = 95) -> None:
    """원본 스크린샷을 JPEG 로 저장한다.

    quality 기본값 95 는 단발 캡처 기준이다. 연속 녹화처럼 장수가 많은 경우
    호출부가 낮춰 쓴다(수동 녹화 세션은 85).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="JPEG", quality=int(quality))
```

- [ ] **Step 6: 런처 본체(Windows 경로)를 구현한다**

`poc/workflow_3/monitor/manual_record.py` 끝에 추가한다. 이 부분은 Mac 에서 실행할 수
없으므로(창이 없음) 단위 테스트 대상이 아니다. import 는 성공해야 한다.

```python
from poc.workflow_3.monitor.frame_meta import (
    FrameMetaWriter,
    build_meta_record,
    probe_occlusion,
    read_cursor_screen_xy,
)
from poc.workflow_3.monitor.recording import RecordingSession


def _collect_monitoring_rows():
    """열려 있는 Remote Monitoring 창 목록을 (title, handle) 로 모은다."""
    from poc.workflow_3.util import collect_window_rows

    if collect_window_rows is None:
        print("[ERROR] window_utils 를 쓸 수 없습니다(Windows 에서 실행하세요).")
        return []
    prefix = REMOTE_MONITORING_WINDOW_TITLE_PREFIX.lower()
    rows = []
    for row in collect_window_rows(visible_only=True):
        if (row.title or "").strip().lower().startswith(prefix):
            rows.append((row.title, row.handle))
    return rows


def _make_capture_fn(tool_window, meta_writer, settings, started_at, our_handles):
    """RecordingSession 에 주입할 capture_fn 을 만든다(캡처 + 사이드카 기록).

    RecordingSession 은 수정하지 않는다. 캡처 함수를 감싸는 것만으로 프레임과
    같은 시각의 창 rect/가림/커서를 남길 수 있다. 사이드카 기록 실패는 삼켜
    캡처 자체를 방해하지 않는다.
    """
    from poc.workflow_3.util import capture_window

    state = {"seq": 0}

    def _capture():
        image = capture_window(tool_window)
        if meta_writer is None:
            return image
        try:
            rect_obj = tool_window.rectangle()
            rect = {
                "left": int(rect_obj.left), "top": int(rect_obj.top),
                "right": int(rect_obj.right), "bottom": int(rect_obj.bottom),
            }
            from poc.workflow_3.util import read_foreground_window_info

            _fg_handle, fg_title = (
                read_foreground_window_info() if read_foreground_window_info else (None, "")
            )
            meta_writer.append(build_meta_record(
                frame_name=f"seq_{state['seq']:04d}",
                t_sec=time.time() - started_at,
                rect=rect,
                foreground_title=fg_title,
                occlusion=probe_occlusion(rect, our_handles),
                cursor_xy=read_cursor_screen_xy(),
            ))
            state["seq"] += 1
        except Exception as exc:
            print(f"[WARNING] frame_meta 수집 실패(계속 진행): {exc}")
        return image

    return _capture


def main() -> int:
    """수동 녹화 세션을 실행한다. 종료 코드 0=정상, 1=시작 실패."""
    settings = load_manual_record_settings()
    rows = _collect_monitoring_rows()
    if not rows:
        print("[ERROR] 열려 있는 Remote Monitoring 창이 없습니다. RCS 에서 tool 을 먼저 열어주세요.")
        return 1

    chosen = pick_window_row(rows, settings.eqp_id)
    if chosen is None:
        print(f"[ERROR] 모니터링 창이 {len(rows)}개 있습니다. MANUAL_RECORD_EQP_ID 로 지정하세요:")
        for title, handle in rows:
            print(f"        - {title}  (handle={handle})")
        return 1

    title, handle = chosen
    eqp_id = parse_eqp_from_title(title) or UNKNOWN_EQP
    tag = make_timestamp_tag()
    out_dir = manual_recording_dir(eqp_id, tag)
    print(f"[INFO] 수동 녹화 대상: eqp={eqp_id!r}, title={title!r}")
    print(f"[INFO] 저장 경로: {out_dir}")
    print(
        f"[INFO] 상한: max_sec={settings.max_sec}s, max_frames={settings.max_frames}, "
        f"max_disk_mb={settings.max_disk_mb}, poll={settings.poll_sec}s"
    )

    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window

    tool_window, resolved_title, _backend = find_remote_monitoring_window(eqp_id)
    if tool_window is None:
        print(f"[ERROR] 창 핸들을 얻지 못했습니다: {title!r}")
        return 1

    meta_writer = FrameMetaWriter(out_dir) if settings.meta_enabled else None
    started_at = time.time()
    session = RecordingSession(
        tool_window, out_dir, tag=tag,
        poll_sec=settings.poll_sec,
        heartbeat_sec=settings.heartbeat_sec,
        change_min_px=settings.change_min_px,
        max_sec=settings.max_sec,
        capture_fn=_make_capture_fn(
            tool_window, meta_writer, settings, started_at, {handle},
        ),
    )
    session.start()
    print("[INFO] 녹화 중입니다. 중지하려면 Ctrl+C 를 누르세요.")

    stop_reason = "stopped"
    try:
        while session.is_alive():
            time.sleep(settings.watch_interval_sec)
            reason = budget_stop_reason(len(session.frames), dir_size_mb(out_dir), settings)
            if reason:
                print(f"[WARNING] ===== 예산 상한 도달({reason}) - 녹화를 종료합니다 =====")
                stop_reason = reason
                break
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 감지 - 녹화를 종료합니다.")
        stop_reason = "user_interrupt"

    frames = session.stop(stop_reason)
    if meta_writer is not None:
        meta_writer.close()
    print(f"[INFO] ===== 녹화 종료: {len(frames)} 프레임, 사유={session.stop_reason} =====")
    print(f"[INFO] 프레임 경로: {out_dir}")
    print("[INFO] 분석하려면: RECORDING_FILTER_INPUT_DIR=<위 경로> "
          "uv run python poc/workflow_3/recording_filter/filter_recording.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

주의 1: `capture_fn` 안에서 만드는 `frame_name` 은 `seq_NNNN` 형태의 **임시 키**다.
`RecordingSession` 이 실제 파일명을 정하고, 변화가 없는 샘플은 저장하지 않으므로
capture 순번과 파일 seq 가 어긋난다. Task 6 에서 분석기가 사이드카를 **`t_sec` 최근접
매칭**으로 조인하는 이유가 이것이다.

주의 2: `settings.jpeg_quality` 는 이 시점에 아직 배선되지 않았다. `RecordingSession`
이 `save_debug_jpeg(image, out_path)` 를 직접 부르기 때문이다(`recording.py:169`).
`RecordingSession` 무수정 원칙을 지키기 위해, 런처는 `capture_fn` 이 돌려주는 PIL
이미지를 그대로 두고 **품질 조정은 하지 않는다.** `MANUAL_RECORD_JPEG_QUALITY` 는
Task 6 에서 후처리(분석 전 재압축)로 쓰거나, 필요해지면 별도 스펙에서 다룬다.
계획 단계의 이 타협을 README 에 명시한다.

- [ ] **Step 7: 테스트를 다시 돌려 회귀가 없는지 확인한다**

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: PASS — 28개 `[OK]` (신규 Windows 코드는 import 만 되고 실행되지 않는다)

Run: `uv run python -c "import poc.workflow_3.monitor.manual_record"`
Expected: 오류 없이 종료 (Mac 에서 import 가능해야 한다)

- [ ] **Step 8: README 에 사용법을 추가한다**

`poc/workflow_3/README.md` 에 절을 추가한다.

```markdown
### 엔지니어 수동 조작 녹화 (알람 불필요)

엔지니어와 약속한 뒤, 이미 열려 있는 Remote Monitoring 창을 그 자리에서 녹화한다.
접속(tool 더블클릭)은 하지 않는다.

```bash
uv run python poc/workflow_3/monitor/manual_record.py
```

- 모니터링 창이 여러 개면 목록을 출력하고 종료한다. `MANUAL_RECORD_EQP_ID` 로 지정한다.
- 기본 상한은 **600초(10분)**. 프레임 4000장 / 2000MB 는 백스톱이라 정상이면 걸리지 않는다.
- Ctrl+C 로 종료. 창을 닫아도 자동 종료된다.
- 저장 경로: `align_images/<EQP>/_manual/<tag>/recording/`
- 분석은 별도 실행: `RECORDING_FILTER_INPUT_DIR=<경로> uv run python poc/workflow_3/recording_filter/filter_recording.py`
- 알려진 제약: 프레임 JPEG 품질은 `RecordingSession` 기본값(95)을 따른다. `MANUAL_RECORD_JPEG_QUALITY` 는 아직 배선되지 않았다.

| env | 기본값 | 역할 |
|-----|--------|------|
| `MANUAL_RECORD_MAX_SEC` | 600 | 시간 상한(0=무제한) |
| `MANUAL_RECORD_MAX_FRAMES` | 4000 | 프레임 백스톱 |
| `MANUAL_RECORD_MAX_DISK_MB` | 2000 | 디스크 백스톱 |
| `MANUAL_RECORD_POLL_SEC` | 0.2 | 샘플링 요청 간격 |
| `MANUAL_RECORD_EQP_ID` | (빈값) | 창이 여럿일 때 지정 |
| `MANUAL_RECORD_META` | 1 | 사이드카 메타 기록 |
```

- [ ] **Step 9: 커밋**

```bash
git add poc/workflow_3/monitor/manual_record.py poc/workflow_3/monitor/test_manual_record.py \
        poc/workflow_3/debug_artifacts.py poc/workflow_3/README.md
git commit -m "feat(workflow_3): 수동 녹화 런처 본체 - 창 탐색/예산 감시/세션 수명주기"
```

---

### Task 4: 영역 게이트 + 레이아웃 세대 (Stage 1.5)

**Files:**
- Create: `poc/workflow_3/recording_filter/region_gate.py`
- Create: `poc/workflow_3/recording_filter/test_region_gate.py`
- Modify: `poc/workflow_3/recording_filter/settings.py`

**Interfaces:**
- Consumes: `poc.workflow_3.recording_filter.frame_reduce.ChangeEvent` (필드: `rank`, `frame_path`, `prev_frame_path`, `timestamp_sec`, `frame_index`, `change_bbox`, `largest_blob_area_px`, `changed_pixels`), `poc.workflow_3.sem_monitor.sem_box_detect.detect_sem_box(image, client) -> SemBoxDetection` (필드 중 `detected: bool`, `bbox_px: dict | None`)
- Produces:
  - `FrameMeta` (dataclass): `t_sec: float`, `rect: dict | None`, `occlusion: str`, `cursor_xy: list | None`, `cursor_in_window: bool`
  - `load_frame_meta(capture_dir: Path) -> list[FrameMeta]` — 없으면 `[]`
  - `nearest_meta(metas: list, t_sec: float) -> FrameMeta | None`
  - `assign_generations(metas: list) -> list[int]`
  - `RegionMap` (dataclass): `generation: int`, `live_box: dict | None`
  - `gate_verdict(change_bbox, live_box, cursor_in_live, has_meta) -> str` — `"ambient"` | `"candidate"`
  - `build_region_maps(events, metas, client, out_dir) -> dict[int, RegionMap]`
  - `apply_region_gate(events, metas, region_maps) -> list[tuple]` — `(event, generation, verdict, occlusion)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/recording_filter/test_region_gate.py` 를 새로 만든다.

```python
"""Stage 1.5 영역 게이트 - 세대 분할 / ambient 판정 / 사이드카 조인 테스트."""

import json

from poc.workflow_3.recording_filter.region_gate import (
    FrameMeta,
    assign_generations,
    gate_verdict,
    load_frame_meta,
    nearest_meta,
)


def _meta(t_sec, rect, occlusion="none", cursor_xy=None, cursor_in_window=False):
    return FrameMeta(
        t_sec=t_sec, rect=rect, occlusion=occlusion,
        cursor_xy=cursor_xy, cursor_in_window=cursor_in_window,
    )


_RECT_A = {"left": 0, "top": 0, "right": 1600, "bottom": 1000}
_RECT_B = {"left": 100, "top": 50, "right": 1700, "bottom": 1050}


def test_single_generation_when_rect_stable():
    """창이 안 움직이면 세대는 하나다."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_A), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 0, 0]


def test_new_generation_when_rect_changes():
    """창을 옮기면 그 시점부터 새 세대."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_B), _meta(0.4, _RECT_B)]
    assert assign_generations(metas) == [0, 1, 1]


def test_generation_increments_again_on_return():
    """원래 위치로 되돌아와도 새 세대다(지도 재검출이 필요하므로)."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_B), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 1, 2]


def test_generation_ignores_missing_rect():
    """rect 가 없는 프레임은 직전 세대를 물려받는다(판정 불가로 쪼개지 않는다)."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, None), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 0, 0]


def test_generation_empty_input():
    assert assign_generations([]) == []


_LIVE_BOX = {"left": 400, "top": 200, "right": 1200, "bottom": 800}


def test_gate_ambient_when_change_inside_live_box_only():
    """라이브 박스 안에서만 변했고 커서가 밖이면 장비 자율 갱신."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, False, True) == "ambient"


def test_gate_candidate_when_change_touches_ui():
    """UI 영역에 걸치면 승격."""
    overlapping = {"left": 100, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(overlapping, _LIVE_BOX, False, True) == "candidate"


def test_gate_candidate_when_cursor_in_live_box():
    """커서가 라이브 박스 안이면 직접 조작 가능성이 있어 승격."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, True, True) == "candidate"


def test_gate_candidate_when_no_live_box():
    """박스를 못 찾은 세대는 게이트 없이 통과 - 오탐이 늘 뿐 데이터는 안 잃는다."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, None, False, True) == "candidate"


def test_gate_candidate_when_meta_missing():
    """사이드카가 없으면 커서 예외를 못 쓰므로 안전하게 전부 승격."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, False, False) == "candidate"


def test_load_frame_meta_missing_file(tmp_path):
    """사이드카가 없으면 빈 목록(실패 아님)."""
    assert load_frame_meta(tmp_path) == []


def test_load_frame_meta_parses_lines(tmp_path):
    """JSONL 을 FrameMeta 로 읽는다. 깨진 줄은 건너뛴다."""
    lines = [
        json.dumps({"frame": "a", "t_sec": 0.0, "window_rect": _RECT_A,
                    "occlusion": "none", "cursor_screen_xy": [10, 20],
                    "cursor_in_window": True}),
        "{ broken json",
        json.dumps({"frame": "b", "t_sec": 0.4, "window_rect": _RECT_A,
                    "occlusion": "partial", "cursor_screen_xy": None,
                    "cursor_in_window": False}),
    ]
    (tmp_path / "frame_meta.jsonl").write_text("\n".join(lines), encoding="utf-8")
    metas = load_frame_meta(tmp_path)
    assert len(metas) == 2, metas
    assert metas[0].cursor_xy == [10, 20]
    assert metas[1].occlusion == "partial"


def test_nearest_meta_picks_closest_timestamp():
    """capture 순번과 파일 seq 가 어긋나므로 t_sec 최근접으로 조인한다."""
    metas = [_meta(0.0, _RECT_A), _meta(0.4, _RECT_A), _meta(1.0, _RECT_A)]
    assert nearest_meta(metas, 0.35).t_sec == 0.4
    assert nearest_meta(metas, 0.9).t_sec == 1.0
    assert nearest_meta([], 0.5) is None
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_region_gate.py -v`
Expected: FAIL — collection error, `No module named 'poc.workflow_3.recording_filter.region_gate'`

- [ ] **Step 3: 구현을 쓴다**

`poc/workflow_3/recording_filter/region_gate.py` 를 새로 만든다.

```python
"""STAGE 1.5 - 영역 게이트: 라이브 SEM 영상의 자율 변화를 이벤트에서 걷어낸다.

알람 사이클은 장비가 멈춰 화면이 정적이라 "변화 = 사람의 조작"이 성립했다. 엔지니어가
수동 조작하는 동안은 라이브 SEM 영상이 계속 갱신되므로 그 전제가 깨진다. 이 스테이지는
프레임을 live_image(라이브 박스) 와 ui(나머지) 로만 나눠, 라이브 박스 안에서만 일어난
변화를 ambient 로 강등한다.

비용 설계가 핵심이다 - 영역 지도는 **세대당 한 번만** VLM(detect_sem_box)을 쓰고,
프레임 단위 게이팅은 순수 기하 비교라 VLM 0콜이다. 세션 길이가 아니라 세대 수에만
비용이 비례한다.

레이아웃 세대(generation): 창을 옮기거나 리사이즈하면 좌표계가 통째로 바뀐다. 창 rect
가 달라지는 시점마다 새 세대를 열어 지도를 다시 뽑는다. 장비 A/B/C 의 레이아웃 차이는
detect_sem_box 가 그 장비의 실제 프레임을 보고 찾으므로 자동으로 흡수된다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

FRAME_META_FILENAME = "frame_meta.jsonl"


@dataclass
class FrameMeta:
    """사이드카 1줄 - 프레임과 같은 시각의 창 상태."""

    t_sec: float
    rect: dict | None
    occlusion: str
    cursor_xy: list | None
    cursor_in_window: bool


@dataclass
class RegionMap:
    """한 레이아웃 세대의 영역 지도."""

    generation: int
    live_box: dict | None   # 프레임 픽셀 기준 {left,top,right,bottom}. None = 검출 실패.


def load_frame_meta(capture_dir) -> list:
    """frame_meta.jsonl 을 읽어 FrameMeta 목록으로 만든다. 없으면 빈 목록."""
    path = Path(capture_dir) / FRAME_META_FILENAME
    if not path.is_file():
        print(f"[INFO] 사이드카 없음 - 커서/가림 신호 없이 진행합니다: {path}")
        return []
    metas = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except Exception:
            continue
        metas.append(FrameMeta(
            t_sec=float(raw.get("t_sec") or 0.0),
            rect=raw.get("window_rect"),
            occlusion=str(raw.get("occlusion") or "unknown"),
            cursor_xy=raw.get("cursor_screen_xy"),
            cursor_in_window=bool(raw.get("cursor_in_window")),
        ))
    metas.sort(key=lambda m: m.t_sec)
    print(f"[INFO] 사이드카 {len(metas)} 건 로드: {path}")
    return metas


def nearest_meta(metas, t_sec):
    """t_sec 에 가장 가까운 FrameMeta 를 돌려준다(없으면 None).

    capture 호출 순번과 저장된 프레임 seq 는 어긋난다(변화 없는 샘플은 저장되지
    않는다). 그래서 순번이 아니라 시각으로 조인한다.
    """
    if not metas:
        return None
    return min(metas, key=lambda m: abs(m.t_sec - float(t_sec)))


def _rect_key(rect):
    if not rect:
        return None
    return (int(rect["left"]), int(rect["top"]), int(rect["right"]), int(rect["bottom"]))


def assign_generations(metas) -> list:
    """FrameMeta 목록에 레이아웃 세대 번호를 매긴다.

    rect 가 바뀌는 지점마다 세대가 하나 늘어난다. rect 가 없는 프레임은 직전 세대를
    물려받는다(판정 불가를 이유로 세대를 쪼개면 지도만 늘고 이득이 없다).
    """
    generations = []
    current = 0
    last_key = None
    for meta in metas:
        key = _rect_key(meta.rect)
        if key is not None:
            if last_key is not None and key != last_key:
                current += 1
            last_key = key
        generations.append(current)
    return generations


def _boxes_overlap(a, b) -> bool:
    return not (
        a["right"] <= b["left"] or a["left"] >= b["right"]
        or a["bottom"] <= b["top"] or a["top"] >= b["bottom"]
    )


def _box_contains(outer, inner) -> bool:
    return (
        inner["left"] >= outer["left"] and inner["top"] >= outer["top"]
        and inner["right"] <= outer["right"] and inner["bottom"] <= outer["bottom"]
    )


def gate_verdict(change_bbox, live_box, cursor_in_live, has_meta) -> str:
    """변화 bbox 를 ambient / candidate 로 판정한다.

    ambient 는 "라이브 박스 안에서만 변했고 커서는 밖" 인 경우뿐이다. 나머지는 전부
    candidate 로 승격한다 - 조용히 이벤트를 잃는 것보다 오탐이 낫다.
    """
    if live_box is None:
        return "candidate"          # 지도 없음 - 게이트 무효화.
    if not has_meta:
        return "candidate"          # 커서 예외를 쓸 수 없음 - 안전 쪽으로.
    if cursor_in_live:
        return "candidate"          # 라이브 영상 직접 조작 가능성.
    if change_bbox and _box_contains(live_box, change_bbox):
        return "ambient"
    return "candidate"


def build_region_maps(events, metas, client, out_dir) -> dict:
    """세대별 영역 지도를 만든다(세대당 detect_sem_box 1회 + 확인용 오버레이 저장)."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")
    from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box

    generations = assign_generations(metas)
    gen_by_time = list(zip([m.t_sec for m in metas], generations))
    maps = {}
    out_dir = Path(out_dir)

    for event in events:
        generation = _generation_for(gen_by_time, event.timestamp_sec)
        if generation in maps:
            continue
        try:
            image = Image.open(event.frame_path).convert("RGB")
            detection = detect_sem_box(image, client)
            live_box = detection.bbox_px if detection.detected else None
        except Exception as exc:
            print(f"[WARNING] 세대 {generation} 영역 지도 검출 실패(게이트 없이 통과): {exc}")
            live_box = None
            image = None
        maps[generation] = RegionMap(generation=generation, live_box=live_box)
        if image is not None and live_box is not None:
            try:
                save_marked_bboxes(
                    image, {"live_image": {"bbox": live_box}},
                    {"live_image": "cyan"},
                    out_dir / f"region_map_gen{generation}.jpg",
                )
            except Exception as exc:
                print(f"[WARNING] 영역 지도 오버레이 저장 실패: {exc}")

    # 스펙 8.2 의 region_map.json - 오피스에서 세대별 박스를 텍스트로 대조할 수 있어야 한다.
    from poc.workflow_3.debug_artifacts import save_debug_json

    save_debug_json(
        out_dir / "region_map.json",
        {
            "generations": [
                {"generation": gen, "live_box": region_map.live_box}
                for gen, region_map in sorted(maps.items())
            ]
        },
    )
    print(f"[INFO] 영역 지도 {len(maps)} 세대 확보(VLM 호출 = 세대 수).")
    return maps


def _generation_for(gen_by_time, t_sec) -> int:
    """시각으로 세대 번호를 찾는다(사이드카 없으면 항상 0)."""
    if not gen_by_time:
        return 0
    return min(gen_by_time, key=lambda pair: abs(pair[0] - float(t_sec)))[1]


def apply_region_gate(events, metas, region_maps) -> list:
    """변화 이벤트마다 (event, generation, verdict, occlusion) 을 계산한다."""
    generations = assign_generations(metas)
    gen_by_time = list(zip([m.t_sec for m in metas], generations))
    has_meta = bool(metas)
    results = []
    for event in events:
        generation = _generation_for(gen_by_time, event.timestamp_sec)
        region_map = region_maps.get(generation)
        live_box = region_map.live_box if region_map else None
        meta = nearest_meta(metas, event.timestamp_sec)
        cursor_in_live = False
        occlusion = "unknown"
        if meta is not None:
            occlusion = meta.occlusion
            if meta.cursor_xy and meta.rect and live_box:
                # 화면 좌표 -> 프레임 좌표로 옮긴 뒤 라이브 박스 포함 여부를 본다.
                fx = int(meta.cursor_xy[0]) - int(meta.rect["left"])
                fy = int(meta.cursor_xy[1]) - int(meta.rect["top"])
                cursor_in_live = (
                    live_box["left"] <= fx <= live_box["right"]
                    and live_box["top"] <= fy <= live_box["bottom"]
                )
        verdict = gate_verdict(event.change_bbox, live_box, cursor_in_live, has_meta)
        results.append((event, generation, verdict, occlusion))

    n_ambient = sum(1 for _e, _g, v, _o in results if v == "ambient")
    print(f"[INFO] Stage 1.5 완료: ambient={n_ambient} / 전체={len(results)}")
    return results
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_region_gate.py -v`
Expected: PASS — 14 passed

- [ ] **Step 5: settings 에 Stage 1.5 필드를 추가한다**

`poc/workflow_3/recording_filter/settings.py` 의 `RecordingFilterSettings` 에 추가하고,
`load_recording_filter_settings()` 에도 env 오버라이드를 배선한다.

```python
    # ---- Stage 1.5: 영역 게이트 ----
    region_gate_enabled: bool = True     # 0 이면 게이트 없이 전부 candidate.
    # ---- Stage 2c: 요소 라벨링 ----
    element_crop_px: int = 260           # 클릭 지점 주변 crop 한 변.
    element_ocr_service: str = "paddleocr-vl-1.5"
    element_vlm_service: str = "mai-ui"
    element_label_enabled: bool = True
```

```python
        region_gate_enabled=env_flag("RECORDING_FILTER_REGION_GATE", True),
        element_crop_px=env_int("RECORDING_FILTER_ELEMENT_CROP_PX", 260),
        element_label_enabled=env_flag("RECORDING_FILTER_ELEMENT_LABEL", True),
```

`env_flag` 를 `from poc.workflow_3.util import env_flag, env_float, env_int` 로 import 에 추가한다.

- [ ] **Step 6: 전체 recording_filter 테스트로 회귀를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter -v`
Expected: PASS — 기존 18 + 신규 14 = 32 passed

- [ ] **Step 7: 커밋**

```bash
git add poc/workflow_3/recording_filter/region_gate.py \
        poc/workflow_3/recording_filter/test_region_gate.py \
        poc/workflow_3/recording_filter/settings.py
git commit -m "feat(workflow_3): recording_filter Stage 1.5 영역 게이트 + 레이아웃 세대"
```

---

### Task 5: 요소 라벨링 (Stage 2c)

클릭 지점 주변을 crop 해 **OCR 로 먼저** 라벨을 읽고, 못 읽으면 VLM 에 서술을 맡긴다.

**Files:**
- Create: `poc/workflow_3/recording_filter/element_label.py`
- Create: `poc/workflow_3/recording_filter/test_element_label.py`

**Interfaces:**
- Consumes: `poc.workflow_3.vlm.ocr_spotting.parse_spotting_items(raw_text) -> list[dict]` (각 항목: `{"text": str, "box": {"left","top","right","bottom"}}`), `poc.workflow_3.vlm.prompts.prompt_ocr_assist.build_spotting_prompt() -> (system, user)`, `poc.workflow_3.util.encode_image_webp(image, quality) -> (b64, w, h)`, `poc.workflow_3.util.json_utils.extract_json(text) -> dict`
- Produces:
  - `ElementLabel` (dataclass): `text: str`, `source: str` (`"ocr"`|`"vlm"`|`"none"`), `confidence: float`
  - `crop_box_around(x, y, side, width, height) -> dict`
  - `pick_nearest_item(items, click_xy, crop_origin) -> dict | None`
  - `label_element(image, click_xy, settings, *, ocr_client, vlm_client) -> ElementLabel`
  - `element_label_prompt() -> tuple[str, str]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/recording_filter/test_element_label.py` 를 새로 만든다.

```python
"""Stage 2c 요소 라벨링 - crop 기하 / OCR 우선 / VLM 폴백 테스트(클라이언트 주입)."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.element_label import (
    ElementLabel,
    crop_box_around,
    label_element,
    pick_nearest_item,
)
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """호출 횟수를 세는 가짜 VLM/OCR 클라이언트."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat_with_image_b64(self, **_kwargs):
        self.calls += 1
        return _Resp(self.text)


class _BoomClient:
    def __init__(self):
        self.calls = 0

    def chat_with_image_b64(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("service down")


def _image(w=1200, h=800):
    return Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8), mode="RGB")


def test_crop_box_is_centered_and_clamped():
    box = crop_box_around(600, 400, 200, 1200, 800)
    assert box == {"left": 500, "top": 300, "right": 700, "bottom": 500}, box
    # 좌상단 모서리에서도 이미지 밖으로 안 나간다.
    edge = crop_box_around(10, 10, 200, 1200, 800)
    assert edge["left"] == 0 and edge["top"] == 0, edge


def test_pick_nearest_item_uses_click_point():
    items = [
        {"text": "Cancel", "box": {"left": 0, "top": 0, "right": 40, "bottom": 20}},
        {"text": "Start", "box": {"left": 100, "top": 100, "right": 160, "bottom": 130}},
    ]
    # crop 원점이 (500, 300) 이고 클릭이 (630, 415) 면 crop 좌표로 (130, 115) -> Start.
    picked = pick_nearest_item(items, (630, 415), (500, 300))
    assert picked["text"] == "Start", picked


def test_pick_nearest_item_empty():
    assert pick_nearest_item([], (10, 10), (0, 0)) is None


def test_label_uses_ocr_and_skips_vlm():
    """OCR 이 텍스트를 주면 VLM 은 호출되지 않는다 - 비용 설계의 핵심."""
    ocr_text = json.dumps([
        {"text": "Start Measurement",
         "box": {"left": 100, "top": 100, "right": 220, "bottom": 130}},
    ])
    ocr = _FakeClient(ocr_text)
    vlm = _FakeClient('{"element": "should not be used"}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert isinstance(label, ElementLabel)
    assert label.text == "Start Measurement", label
    assert label.source == "ocr", label
    assert ocr.calls == 1 and vlm.calls == 0, (ocr.calls, vlm.calls)


def test_label_falls_back_to_vlm_when_ocr_empty():
    """OCR 이 빈 결과면 VLM 이 서술한다(아이콘 버튼/라이브 영상)."""
    ocr = _FakeClient("[]")
    vlm = _FakeClient('{"element": "zoom icon button", "confidence": 0.6}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert label.text == "zoom icon button", label
    assert label.source == "vlm", label
    assert ocr.calls == 1 and vlm.calls == 1, (ocr.calls, vlm.calls)


def test_label_falls_back_to_vlm_when_ocr_raises():
    """OCR 이 던져도 VLM 폴백으로 이어진다."""
    ocr = _BoomClient()
    vlm = _FakeClient('{"element": "OK button"}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert label.text == "OK button" and label.source == "vlm", label


def test_label_none_when_both_fail():
    """둘 다 실패하면 source=none - 이벤트 자체는 남아야 한다."""
    settings = RecordingFilterSettings(element_crop_px=260)
    label = label_element(
        _image(), (630, 415), settings, ocr_client=_BoomClient(), vlm_client=_BoomClient(),
    )
    assert label.text == "" and label.source == "none", label


def test_label_none_when_clients_missing():
    """클라이언트가 없으면 호출 없이 none."""
    settings = RecordingFilterSettings(element_crop_px=260)
    label = label_element(_image(), (10, 10), settings, ocr_client=None, vlm_client=None)
    assert label.source == "none", label
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_element_label.py -v`
Expected: FAIL — `No module named 'poc.workflow_3.recording_filter.element_label'`

- [ ] **Step 3: 구현을 쓴다**

`poc/workflow_3/recording_filter/element_label.py` 를 새로 만든다.

```python
"""STAGE 2c - 클릭 지점의 UI 요소 라벨을 읽는다(OCR 우선, VLM 폴백).

timeline 의 element 필드는 지금까지 예약만 되어 있었다. workflow 로 변환하려면
"언제 어디를" 다음에 "무엇을" 이 필요하다. 라벨이 있어야 다른 장비에서도 그 요소를
다시 찾을 수 있기 때문이다(좌표는 창 위치가 달라 이식되지 않는다).

순서가 곧 비용 설계다 - 텍스트 버튼은 PaddleOCR 한 번으로 끝나고, 아이콘/라이브 영상
처럼 읽을 텍스트가 없을 때만 VLM 이 나선다. 전체 스크린샷 OCR 은 환각이 심하므로
반드시 작은 crop 에만 적용한다.
"""

from dataclasses import dataclass

from poc.workflow_3.util import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_spotting_prompt


@dataclass
class ElementLabel:
    """클릭 지점의 요소 라벨 1건."""

    text: str
    source: str        # "ocr" | "vlm" | "none"
    confidence: float


def crop_box_around(x, y, side, width, height) -> dict:
    """클릭 지점을 중심으로 한 정사각 crop 박스를 이미지 안으로 클램프해 만든다."""
    half = max(1, int(side) // 2)
    left = max(0, int(x) - half)
    top = max(0, int(y) - half)
    right = min(int(width), int(x) + half)
    bottom = min(int(height), int(y) + half)
    if right <= left:
        right = min(int(width), left + 1)
    if bottom <= top:
        bottom = min(int(height), top + 1)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def pick_nearest_item(items, click_xy, crop_origin):
    """OCR 항목 중 클릭 지점에 가장 가까운 것을 고른다(crop 좌표계로 환산해 비교)."""
    if not items:
        return None
    cx = int(click_xy[0]) - int(crop_origin[0])
    cy = int(click_xy[1]) - int(crop_origin[1])

    def _distance(item):
        box = item.get("box") or {}
        try:
            mx = (int(box["left"]) + int(box["right"])) / 2.0
            my = (int(box["top"]) + int(box["bottom"])) / 2.0
        except Exception:
            return float("inf")
        return ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5

    best = min(items, key=_distance)
    return best if _distance(best) != float("inf") else None


def element_label_prompt():
    """아이콘/비텍스트 요소용 VLM 프롬프트 (system, user) 를 만든다."""
    system_message = (
        "You are a GUI analyst. Look at the cropped screenshot region and identify "
        "the single UI element at its center. Answer with JSON only."
    )
    user_text = (
        "The center of this crop is where the user clicked. "
        "Identify that UI element concisely (e.g. \"OK button\", \"zoom in icon\", "
        "\"recipe list row\", \"live SEM image\"). "
        "Respond with JSON: {\"element\": \"<short name>\", \"confidence\": <0-1>}"
    )
    return system_message, user_text


def _read_with_ocr(crop_image, click_xy, crop_box, ocr_client):
    """crop 을 PaddleOCR Spotting 으로 읽어 클릭 지점 최근접 텍스트를 돌려준다."""
    crop_b64, _w, _h = encode_image_webp(crop_image, quality=90)
    system_msg, user_text = build_spotting_prompt()
    response = ocr_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    items = parse_spotting_items((response.text or "").strip())
    picked = pick_nearest_item(items, click_xy, (crop_box["left"], crop_box["top"]))
    if picked is None:
        return ""
    return str(picked.get("text") or "").strip()


def _describe_with_vlm(crop_image, vlm_client):
    """crop 중앙의 요소를 VLM 에 서술시킨다. (text, confidence) 반환."""
    crop_b64, _w, _h = encode_image_webp(crop_image, quality=90)
    system_msg, user_text = element_label_prompt()
    response = vlm_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    parsed = extract_json(response.text)
    text = str(parsed.get("element") or "").strip()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return text, confidence


def label_element(image, click_xy, settings, *, ocr_client, vlm_client) -> ElementLabel:
    """클릭 지점의 요소 라벨을 읽는다. 실패해도 던지지 않고 source="none" 을 준다."""
    width, height = image.size
    crop_box = crop_box_around(
        click_xy[0], click_xy[1], settings.element_crop_px, width, height
    )
    crop_image = image.crop(
        (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
    )

    if ocr_client is not None:
        try:
            text = _read_with_ocr(crop_image, click_xy, crop_box, ocr_client)
            if text:
                return ElementLabel(text=text, source="ocr", confidence=1.0)
        except Exception as exc:
            print(f"[WARNING] 요소 OCR 실패(VLM 폴백): {exc}")

    if vlm_client is not None:
        try:
            text, confidence = _describe_with_vlm(crop_image, vlm_client)
            if text:
                return ElementLabel(text=text, source="vlm", confidence=confidence)
        except Exception as exc:
            print(f"[WARNING] 요소 VLM 서술 실패: {exc}")

    return ElementLabel(text="", source="none", confidence=0.0)
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_element_label.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/recording_filter/element_label.py \
        poc/workflow_3/recording_filter/test_element_label.py
git commit -m "feat(workflow_3): recording_filter Stage 2c 요소 라벨링(OCR 우선, VLM 폴백)"
```

---

### Task 6: 타임라인 스키마 확장 + 파이프라인 배선

**Files:**
- Modify: `poc/workflow_3/recording_filter/timeline.py:18-47` (`build_timeline`)
- Modify: `poc/workflow_3/recording_filter/filter_recording.py:50-57` (glob), `:110-181` (`run_filter`)
- Modify: `poc/workflow_3/recording_filter/test_timeline.py` (테스트 추가)
- Modify: `poc/workflow_3/recording_filter/__init__.py` (신규 모듈 export)

**Interfaces:**
- Consumes: Task 4 의 `load_frame_meta` / `build_region_maps` / `apply_region_gate`, Task 5 의 `label_element` / `ElementLabel`
- Produces:
  - `build_timeline(click_events, typing_events=None, *, gate_info=None, labels=None) -> list[dict]` — 이벤트에 `element` / `element_source` / `target_kind` / `region` / `generation` / `occlusion` 추가
  - `derive_target_kind(region: str, element_source: str) -> str`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`poc/workflow_3/recording_filter/test_timeline.py` 에 추가한다.

```python
from poc.workflow_3.recording_filter.timeline import build_timeline, derive_target_kind


def test_derive_target_kind_ui_control():
    """ui 영역 + 라벨 있음 -> 이식 가능한 ui_control."""
    assert derive_target_kind("ui", "ocr") == "ui_control"
    assert derive_target_kind("ui", "vlm") == "ui_control"


def test_derive_target_kind_live_image():
    """라이브 영상 위 조작은 라벨 유무와 무관하게 live_image."""
    assert derive_target_kind("live_image", "ocr") == "live_image"
    assert derive_target_kind("live_image", "none") == "live_image"


def test_derive_target_kind_unknown():
    """라벨이 없으면 사람이 봐야 한다."""
    assert derive_target_kind("ui", "none") == "unknown"
    assert derive_target_kind("unknown", "none") == "unknown"


class _Change:
    def __init__(self, rank, t):
        self.rank = rank
        self.frame_path = f"/tmp/f{rank}.jpg"
        self.prev_frame_path = f"/tmp/f{rank - 1}.jpg"
        self.timestamp_sec = t


class _Click:
    def __init__(self, rank, t):
        self.change = _Change(rank, t)
        self.status = "click"
        self.is_click = True
        self.cursor_xy = [100, 200]
        self.confidence = 0.8

    @property
    def frame_path(self):
        return self.change.frame_path

    @property
    def prev_frame_path(self):
        return self.change.prev_frame_path

    @property
    def timestamp_sec(self):
        return self.change.timestamp_sec

    @property
    def rank(self):
        return self.change.rank


def test_timeline_carries_new_fields():
    """게이트/라벨 정보가 이벤트에 실린다."""
    from poc.workflow_3.recording_filter.element_label import ElementLabel

    clicks = [_Click(rank=0, t=1.0)]
    gate_info = {0: {"generation": 2, "region": "ui", "occlusion": "none"}}
    labels = {0: ElementLabel(text="Start", source="ocr", confidence=1.0)}

    events = build_timeline(clicks, gate_info=gate_info, labels=labels)

    assert len(events) == 1
    ev = events[0]
    assert ev["element"] == "Start"
    assert ev["element_source"] == "ocr"
    assert ev["target_kind"] == "ui_control"
    assert ev["region"] == "ui"
    assert ev["generation"] == 2
    assert ev["occlusion"] == "none"


def test_timeline_defaults_without_gate_or_labels():
    """게이트/라벨 정보가 없어도 기존 스키마로 동작한다(하위 호환)."""
    events = build_timeline([_Click(rank=0, t=1.0)])
    ev = events[0]
    assert ev["element"] is None
    assert ev["element_source"] == "none"
    assert ev["target_kind"] == "unknown"
    assert ev["region"] == "unknown"
    assert ev["generation"] == 0
    assert ev["occlusion"] == "unknown"
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_timeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_target_kind'`

- [ ] **Step 3: timeline 을 확장한다**

`poc/workflow_3/recording_filter/timeline.py` 의 `build_timeline` 을 아래로 교체하고
`derive_target_kind` 를 추가한다.

```python
def derive_target_kind(region, element_source) -> str:
    """region + 라벨 출처로 이식 가능성 종류를 정한다.

    ui_control 은 다른 장비에서 라벨로 다시 찾을 수 있고, live_image 는 좌표가 아니라
    영상 내용에 의존해 CV 재해석이 필요하다. 파생 규칙이 바뀌어도 원본 region 이
    남아 있어 다시 계산할 수 있다.
    """
    if region == "live_image":
        return "live_image"
    if region == "ui" and element_source in {"ocr", "vlm"}:
        return "ui_control"
    return "unknown"


def build_timeline(click_events, typing_events=None, *, gate_info=None, labels=None) -> list:
    """클릭(+미래 타이핑) 이벤트를 시간순 정렬된 dict 목록으로 만든다.

    gate_info / labels 는 rank 키의 dict 다(없으면 기본값으로 채운다). 알람 사이클
    녹화처럼 Stage 1.5/2c 를 돌리지 않은 입력도 그대로 처리된다.
    """
    gate_info = gate_info or {}
    labels = labels or {}
    events: list = []
    for ce in click_events:
        if ce.status != "click" or not ce.is_click:
            continue
        coords = {"x": ce.cursor_xy[0], "y": ce.cursor_xy[1]} if ce.cursor_xy else None
        gate = gate_info.get(ce.rank) or {}
        label = labels.get(ce.rank)
        element_source = label.source if label is not None else "none"
        region = str(gate.get("region") or "unknown")
        events.append(
            {
                "t_sec": ce.timestamp_sec,
                "seq": 0,
                "action": "click",
                "coords": coords,
                "element": (label.text if label is not None and label.text else None),
                "element_source": element_source,
                "target_kind": derive_target_kind(region, element_source),
                "region": region,
                "generation": int(gate.get("generation") or 0),
                "occlusion": str(gate.get("occlusion") or "unknown"),
                "text": None,              # 예약: 타이핑 텍스트 (Stage 2b)
                "confidence": ce.confidence,
                "frame": Path(ce.frame_path).name,
                "source_frames": {
                    "prev": Path(ce.prev_frame_path).name,
                    "curr": Path(ce.frame_path).name,
                },
            }
        )
    for te in (typing_events or []):
        events.append(te)  # 이미 동일 스키마 dict 라고 가정(Stage 2b).

    events.sort(key=lambda e: e["t_sec"])
    for i, event in enumerate(events):
        event["seq"] = i
    return events
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter/test_timeline.py -v`
Expected: PASS — 기존 테스트 + 신규 5개

- [ ] **Step 5: `filter_recording.py` 에 신규 스테이지를 배선한다**

두 곳을 고친다.

(a) 자동 탐색 glob 에 `_manual` 을 추가한다 (`filter_recording.py:50-57`).

```python
    candidates = sorted(
        [
            *ALIGN_IMAGES_DIR.glob("*/*/*/captured_img_from_rcs/*/recording"),
            *ALIGN_IMAGES_DIR.glob("*/_unregistered/*/recording"),
            *ALIGN_IMAGES_DIR.glob("*/_manual/*/recording"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
```

(b) `run_filter` 의 Stage 1 과 Stage 2a 사이에 게이트를, Stage 2a 뒤에 라벨링을 넣는다.
`# ---- Stage 1 ----` 블록 다음, `# ---- Stage 2a ----` 앞에 삽입한다.

```python
    # ---- Stage 1.5: 영역 게이트 ----
    from poc.workflow_3.recording_filter.region_gate import (
        apply_region_gate,
        build_region_maps,
        load_frame_meta,
    )

    metas = load_frame_meta(frames_dir)
    gate_info = {}
    if settings.region_gate_enabled:
        if client is None:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            client = Workflow1VLMClient(settings.vlm_service, model_name=settings.vlm_model)
        region_maps = build_region_maps(change_events, metas, client, out_dir)
        gated = apply_region_gate(change_events, metas, region_maps)
        for event, generation, verdict, occlusion in gated:
            gate_info[event.rank] = {
                "generation": generation,
                "region": "live_image" if verdict == "ambient" else "ui",
                "occlusion": occlusion,
                "verdict": verdict,
            }
        # ambient 와 가려진 프레임은 비싼 Stage 2a 에 태우지 않는다.
        change_events = [
            event for event, _g, verdict, occlusion in gated
            if verdict == "candidate" and occlusion != "full"
        ]
        print(f"[INFO] Stage 1.5 통과: {len(change_events)} 건이 Stage 2a 로 갑니다.")
```

`# ---- Stage 2a ----` 블록의 `click_events = detect_clicks(...)` 다음에 라벨링을 넣는다.

```python
    # ---- Stage 2c: 요소 라벨링 ----
    from poc.workflow_3.recording_filter.element_label import label_element

    labels = {}
    if settings.element_label_enabled:
        from PIL import Image

        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        ocr_client = Workflow1VLMClient(settings.element_ocr_service)
        label_vlm = Workflow1VLMClient(settings.element_vlm_service)
        crops_dir = out_dir / "element_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for ce in click_events:
            if not ce.is_click or not ce.cursor_xy:
                continue
            frame_image = Image.open(ce.frame_path).convert("RGB")
            label = label_element(
                frame_image, (ce.cursor_xy[0], ce.cursor_xy[1]), settings,
                ocr_client=ocr_client, vlm_client=label_vlm,
            )
            labels[ce.rank] = label
            box = crop_box_around(
                ce.cursor_xy[0], ce.cursor_xy[1], settings.element_crop_px,
                frame_image.size[0], frame_image.size[1],
            )
            save_debug_jpeg(
                frame_image.crop((box["left"], box["top"], box["right"], box["bottom"])),
                crops_dir / f"{ce.rank:03d}_{label.source}.jpg",
            )
        n_labeled = sum(1 for lb in labels.values() if lb.source != "none")
        print(f"[INFO] Stage 2c 완료: 라벨 {n_labeled} / {len(labels)}")
```

`build_timeline` 호출을 아래로 바꾼다.

```python
    timeline = build_timeline(click_events, gate_info=gate_info, labels=labels)
```

`summary.json` payload 에 세 줄을 추가한다.

```python
            "generations": len({info["generation"] for info in gate_info.values()}) if gate_info else 0,
            "gate_passed": len(change_events),
            "labeled": sum(1 for lb in labels.values() if lb.source != "none"),
```

파일 상단 import 에 `crop_box_around` 와 `save_debug_jpeg` 를 추가한다.

```python
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.recording_filter.element_label import crop_box_around
```

- [ ] **Step 6: `__init__.py` 에 신규 모듈을 노출한다**

`poc/workflow_3/recording_filter/__init__.py` 의 기존 import 목록 스타일에 맞춰
`region_gate` 의 `apply_region_gate` / `build_region_maps` / `load_frame_meta` 와
`element_label` 의 `ElementLabel` / `label_element` 를 추가한다.

- [ ] **Step 7: 전체 테스트로 회귀를 확인한다**

Run: `uv run pytest poc/workflow_3/recording_filter -v`
Expected: PASS — 기존 18 + region_gate 14 + element_label 9 + timeline 신규 5 = 46 passed

Run: `uv run python poc/workflow_3/monitor/test_manual_record.py`
Expected: PASS — 28개 `[OK]`

Run: `uv run python -c "from poc.workflow_3.recording_filter.filter_recording import run_filter"`
Expected: 오류 없이 종료

- [ ] **Step 8: 커밋**

```bash
git add poc/workflow_3/recording_filter/timeline.py \
        poc/workflow_3/recording_filter/filter_recording.py \
        poc/workflow_3/recording_filter/test_timeline.py \
        poc/workflow_3/recording_filter/__init__.py
git commit -m "feat(workflow_3): 타임라인에 element/target_kind/generation 추가 + 신규 스테이지 배선"
```

---

## 오피스(Windows) 확인 절차

Mac 에서 검증할 수 없는 항목이다. 구현 완료 후 오피스에서 순서대로 확인한다.

1. RCS 로그인 후 tool 을 하나 열어둔다.
2. `uv run python poc/workflow_3/monitor/manual_record.py` — 대상 창 제목과 추출된 EQP 가 콘솔에 맞게 찍히는지 확인한다. (**EQP 추출이 실제 제목 형식과 다르면 여기서 드러난다.**)
3. 30초쯤 마우스를 움직이고 버튼 몇 개를 눌러본 뒤 Ctrl+C.
4. `recording_manifest.json` 의 `sampled_count / 경과시간` 으로 실측 샘플링 주기를 확인한다. 목표는 약 5/s 다.
5. `frame_meta.jsonl` 에서 `occlusion` 이 대부분 `none` 인지, 다른 앱을 띄운 구간이 `partial`/`full` 로 찍히는지 확인한다.
6. `RECORDING_FILTER_INPUT_DIR=<경로> uv run python poc/workflow_3/recording_filter/filter_recording.py`
7. `region_map_gen0.jpg` 의 시안색 박스가 실제 라이브 SEM 영상 영역과 맞는지 눈으로 확인한다. **틀리면 이후 게이팅이 전부 어긋나므로 여기서 멈추고 보고한다.**
8. `interaction_timeline.json` 의 `element` 가 실제 누른 버튼 이름과 맞는지, `element_crops/` 이미지로 대조한다.
9. `summary.json` 의 `gate_passed / total_change_events` 비율을 본다. 게이트가 90% 이상을 걷어냈다면 정상이고, 0% 라면 사이드카 조인이나 박스 검출을 의심한다.

## 범위 밖 (스펙 §11 재확인)

재생기(replay), 키보드 입력 검출, 여러 장비 창 동시 녹화, 툴 상태 요약,
`RecordingSession` 수정. `MANUAL_RECORD_JPEG_QUALITY` 배선도 Task 3 Step 6 주의 2 에
따라 이번 범위에서 제외한다(README 에 명시).
