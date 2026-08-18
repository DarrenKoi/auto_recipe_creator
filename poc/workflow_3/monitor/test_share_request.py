"""share_request 단위 테스트 - VLM/실장비 없이 Mac 에서 실행.

  uv run pytest poc/workflow_3/monitor/test_share_request.py
"""

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.share_request import (
    ACCEPTED,
    DENIED_OR_TIMEOUT,
    REQUEST_BTN_FORBIDDEN,
    REQUEST_BTN_REQUIRED,
    SHARE_SCREEN_FORBIDDEN,
    SHARE_SCREEN_REQUIRED,
    STATUS_BLOCKED_SAFE_MODE,
    STATUS_CONFIRM_FAILED,
    STATUS_NOT_FOUND,
    STATUS_REQUESTED,
    accepts_label,
    classify_label,
    request_screen_share,
    wait_share_response,
)

_SHARE_ENV = (
    "ALIGN_FAIL_SHARE_REQUEST",
    "ALIGN_FAIL_SHARE_CONFIRM",
    "ALIGN_FAIL_SHARE_WAIT_SEC",
    "ALIGN_FAIL_SHARE_MAX_ATTEMPTS",
)


# ------------------------------------------------------------------
# 설정.
# ------------------------------------------------------------------


def test_share_settings_defaults(monkeypatch):
    """기본값: 켜짐, strict, 45초, 2회."""
    for name in _SHARE_ENV:
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


def test_blank_confirm_policy_falls_back_to_strict(monkeypatch):
    """빈 문자열이 게이트를 조용히 열지 않는다."""
    monkeypatch.setenv("ALIGN_FAIL_SHARE_CONFIRM", "   ")
    assert load_workflow3_settings().share_confirm_policy == "strict"


# ------------------------------------------------------------------
# 확인 게이트 판정.
# ------------------------------------------------------------------


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
    """crop 이 옆 라디오를 삼킨 경우 - 둘 다 읽혀도 거부여야 한다."""
    assert _radio(["share", "screen", "terminate"]) == "forbidden"


def test_radio_partial_is_unreadable():
    """'share' 만으로는 제어 공유와 구분되지 않는다."""
    assert _radio(["Request", "to", "share"]) == "unreadable"


def test_radio_empty_is_unreadable():
    assert _radio([]) == "unreadable"


def test_radio_strips_punctuation():
    """OCR 이 구두점을 붙여 와도 읽힌다."""
    assert _radio(["(share)", "screen."]) == "confirmed"


def test_button_confirmed():
    assert _button(["Request"]) == "confirmed"


def test_button_rejects_cancel():
    assert _button(["Cancel"]) == "forbidden"


def test_button_korean():
    assert _button(["요청"]) == "confirmed"


def test_strict_requires_confirmation():
    assert accepts_label("confirmed", "strict") is True
    assert accepts_label("unreadable", "strict") is False
    assert accepts_label("forbidden", "strict") is False


def test_lenient_passes_unreadable_but_never_forbidden():
    assert accepts_label("confirmed", "lenient") is True
    assert accepts_label("unreadable", "lenient") is True
    assert accepts_label("forbidden", "lenient") is False


def test_off_still_blocks_forbidden():
    """off 는 좌표 진단용이지 강제 종료 오클릭 허용이 아니다."""
    assert accepts_label("confirmed", "off") is True
    assert accepts_label("unreadable", "off") is True
    assert accepts_label("forbidden", "off") is False


def test_unknown_policy_falls_back_to_strict():
    """정책 오타가 게이트를 조용히 열면 안 된다."""
    assert accepts_label("unreadable", "typo-policy") is False


# ------------------------------------------------------------------
# 승낙 대기.
# ------------------------------------------------------------------


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
    status, found = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=lambda eqp_id: ("win", "title", "uia"),
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert status == ACCEPTED
    assert found == ("win", "title", "uia")


def test_accepted_when_window_appears_late():
    """상대가 뒤늦게 수락하는 경우."""
    clock = _FakeClock()
    calls = {"n": 0}

    def find(eqp_id):
        calls["n"] += 1
        return ("win", "title", "uia") if calls["n"] >= 4 else None

    status, found = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=find, sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert status == ACCEPTED
    assert found is not None


def test_timeout_when_window_never_appears():
    clock = _FakeClock()
    status, found = wait_share_response(
        "MCD427", 5.0,
        find_window_fn=lambda eqp_id: None,
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert status == DENIED_OR_TIMEOUT
    assert found is None
    assert clock.t >= 5.0


def test_window_lookup_exception_does_not_abort_wait():
    """탐색 1회 실패가 대기 전체를 죽이면 안 된다 - 창은 뒤에 뜰 수 있다."""
    clock = _FakeClock()
    calls = {"n": 0}

    def find(eqp_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("window enum failed")
        return ("win", "title", "uia")

    status, _found = wait_share_response(
        "MCD427", 45.0,
        find_window_fn=find, sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert status == ACCEPTED


def test_zero_wait_returns_timeout_without_sleeping():
    clock = _FakeClock()
    status, _found = wait_share_response(
        "MCD427", 0.0,
        find_window_fn=lambda eqp_id: None,
        sleep_fn=clock.sleep, now_fn=clock.now,
    )
    assert status == DENIED_OR_TIMEOUT
    assert clock.t == 0.0


# ------------------------------------------------------------------
# 클릭 실행 경로.
# ------------------------------------------------------------------


class _Settings:
    """request_screen_share 가 읽는 필드만 갖는 대역."""

    def __init__(self, policy="strict", action_enabled=True):
        self.share_confirm_policy = policy
        self.action_enabled = action_enabled


def _run(*, radio_tokens, button_tokens, policy="strict", action_enabled=True,
         popup=True, locate=True):
    """클릭 기록과 결과를 함께 돌려주는 공통 실행기."""
    clicks = []
    result = request_screen_share(
        _Settings(policy, action_enabled),
        locate_fn=lambda image, target: {"x": 10, "y": 20} if locate else None,
        read_tokens_fn=lambda image, point, key: (
            radio_tokens if key == "share_screen_radio" else button_tokens
        ),
        click_fn=lambda window, image, point, key: clicks.append((key, point)),
        capture_fn=lambda window: object(),
        find_popup_fn=lambda: object() if popup else None,
    )
    return result, clicks


def test_requests_when_both_labels_confirmed():
    result, clicks = _run(
        radio_tokens=["Request", "to", "share", "the", "screen"],
        button_tokens=["Request"],
    )
    assert result.status == STATUS_REQUESTED
    assert [key for key, _point in clicks] == ["share_screen_radio", "request_button"]


def test_no_click_at_all_when_radio_reads_terminate():
    """최악의 경우. 라디오조차 누르지 않아야 한다."""
    result, clicks = _run(
        radio_tokens=["Request", "termination", "of", "existant", "user"],
        button_tokens=["Request"],
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_no_click_when_button_reads_cancel():
    """버튼 확인은 라디오 클릭 **전에** 끝나야 한다 - 남의 팝업 상태를 바꾸지 않는다."""
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
    assert [key for key, _point in clicks] == ["share_screen_radio", "request_button"]


def test_lenient_still_blocks_terminate():
    result, clicks = _run(
        radio_tokens=["terminate"], button_tokens=["Request"], policy="lenient",
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_popup_missing_is_not_found():
    result, clicks = _run(
        radio_tokens=["share", "screen"], button_tokens=["Request"], popup=False,
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
        radio_tokens=["share", "screen"], button_tokens=["Request"], locate=False,
    )
    assert result.status == STATUS_CONFIRM_FAILED
    assert clicks == []


def test_verdicts_are_reported_for_diagnosis():
    """오피스 진단을 위해 판정이 결과에 실려야 한다."""
    result, _clicks = _run(radio_tokens=["control"], button_tokens=["Request"])
    assert result.radio_verdict == "forbidden"


def test_exception_returns_error_not_success():
    """actuator 는 예외를 삼켜 성공으로 만들지 않는다."""
    def _boom(window):
        raise RuntimeError("capture failed")

    result = request_screen_share(
        _Settings(),
        locate_fn=lambda image, target: {"x": 1, "y": 1},
        read_tokens_fn=lambda image, point, key: ["share", "screen"],
        click_fn=lambda window, image, point, key: None,
        capture_fn=_boom,
        find_popup_fn=lambda: object(),
    )
    assert result.status == "error"
    assert "capture failed" in result.error


def test_unknown_confirm_policy_env_warns_and_falls_back(monkeypatch, capsys):
    """오타가 게이트를 다른 정책으로 조용히 바꾸면 안 된다."""
    monkeypatch.setenv("ALIGN_FAIL_SHARE_CONFIRM", "strcit")
    settings = load_workflow3_settings()
    assert settings.share_confirm_policy == "strict"
    assert "알 수 없는 값" in capsys.readouterr().out


def test_ocr_exception_returns_error_not_success():
    """OCR 이 던져도 성공으로 새면 안 된다 (spec 오류처리표)."""
    clicks = []

    def _boom(image, point, key):
        raise RuntimeError("ocr down")

    result = request_screen_share(
        _Settings(),
        locate_fn=lambda image, target: {"x": 1, "y": 1},
        read_tokens_fn=_boom,
        click_fn=lambda window, image, point, key: clicks.append(key),
        capture_fn=lambda window: object(),
        find_popup_fn=lambda: object(),
    )
    assert result.status == "error"
    assert clicks == []


def test_click_fn_receives_window_and_image_for_coord_transform():
    """click_fn 계약: 창과 이미지를 함께 받아야 image->screen 변환이 가능하다.

    이 인자들이 없으면 orchestrator 가 배율 보정을 할 수 없고, 확인 게이트가 읽은 점과
    실제 클릭 지점이 어긋난다(= 게이트 무력화). 계약이 바뀌면 여기서 깨져야 한다.
    """
    seen = []
    popup, image = object(), object()
    request_screen_share(
        _Settings(),
        locate_fn=lambda img, target: {"x": 7, "y": 9},
        read_tokens_fn=lambda img, point, key: (
            ["share", "screen"] if key == "share_screen_radio" else ["Request"]
        ),
        click_fn=lambda w, i, point, key: seen.append((w, i, point, key)),
        capture_fn=lambda w: image,
        find_popup_fn=lambda: popup,
    )
    assert len(seen) == 2
    for window_arg, image_arg, point_arg, _key in seen:
        assert window_arg is popup
        assert image_arg is image
        assert point_arg == {"x": 7, "y": 9}
