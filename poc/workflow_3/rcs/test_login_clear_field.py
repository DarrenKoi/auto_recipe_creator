"""로그인 입력창 클리어 회귀 테스트 — 남은 글자가 있으면 로그인은 반드시 실패한다.

오피스 증상(2026-08-19): 비밀번호 칸에 이전 값이 남아 있는데 backspace 를 **한 번만**
눌러서, 지워지지 않은 글자 뒤에 새 비밀번호가 이어 붙었다. 그 결과 실제로 전송된
값이 달라져 로그인이 실패했다.

가짜 키보드가 **실제 텍스트 필드처럼** 동작한다(캐럿 위치, backspace/delete, 선택
영역). 그래야 "몇 번 눌렀나" 가 아니라 "최종 필드 값이 무엇인가" 를 검증할 수 있다 -
호출 횟수만 세면 구현을 그대로 베낀 동어반복 테스트가 된다.

    uv run pytest poc/workflow_3/rcs/test_login_clear_field.py
"""

import dataclasses

import pytest

from poc.workflow_3.runner.workflow_config import WorkflowSettings


def _settings(**overrides):
    base = dict(
        action_enabled=True,
        char_type_delay_sec=0.0,
        post_type_backspace_settle_sec=0.0,
    )
    base.update(overrides)
    return dataclasses.replace(WorkflowSettings(), **base)


class FakeKey:
    """pynput Key enum 자리표시자."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Key.{self.name}"


class FakeTextField:
    """캐럿과 선택 영역을 가진 최소 텍스트 필드 모델."""

    def __init__(self, initial=""):
        self.text = initial
        self.caret = len(initial)
        self.sel_start = None   # 선택 영역이 있으면 (start, end)

    def _drop_selection(self):
        if self.sel_start is None:
            return False
        start, end = sorted((self.sel_start, self.caret))
        self.text = self.text[:start] + self.text[end:]
        self.caret = start
        self.sel_start = None
        return True

    def select_all(self):
        self.sel_start = 0
        self.caret = len(self.text)

    def backspace(self):
        if self._drop_selection():
            return
        if self.caret > 0:
            self.text = self.text[: self.caret - 1] + self.text[self.caret:]
            self.caret -= 1

    def delete(self):
        if self._drop_selection():
            return
        if self.caret < len(self.text):
            self.text = self.text[: self.caret] + self.text[self.caret + 1:]

    def end(self):
        self.sel_start = None
        self.caret = len(self.text)

    def insert(self, ch):
        self._drop_selection()
        self.text = self.text[: self.caret] + ch + self.text[self.caret:]
        self.caret += 1


class FakeKeyboard:
    """pynput KeyboardController 대역 - FakeTextField 에 키를 흘려보낸다."""

    def __init__(self, field, *, supports_ctrl_a=True):
        self.field = field
        self.supports_ctrl_a = supports_ctrl_a
        self._held = set()

    # --- pynput API ---
    def press(self, key):
        name = getattr(key, "name", None)
        if name in {"ctrl", "ctrl_l", "ctrl_r"}:
            self._held.add("ctrl")
            return
        self._dispatch(key)

    def release(self, key):
        name = getattr(key, "name", None)
        if name in {"ctrl", "ctrl_l", "ctrl_r"}:
            self._held.discard("ctrl")

    def type(self, text):
        for ch in text:
            self.field.insert(ch)

    def pressed(self, key):
        """`with keyboard.pressed(Key.ctrl):` 컨텍스트 매니저."""
        outer = self

        class _Ctx:
            def __enter__(self):
                outer._held.add("ctrl")
                return outer

            def __exit__(self, *exc):
                outer._held.discard("ctrl")
                return False

        return _Ctx()

    # --- 내부 ---
    def _dispatch(self, key):
        name = getattr(key, "name", None)
        if "ctrl" in self._held:
            if key == "a" or name == "a":
                if self.supports_ctrl_a:
                    self.field.select_all()
                return
            return
        if name == "backspace":
            self.field.backspace()
        elif name == "delete":
            self.field.delete()
        elif name == "end":
            self.field.end()
        elif isinstance(key, str):
            self.field.insert(key)


def _clear_and_type(text, target_key, settings, keyboard):
    from poc.workflow_3.rcs.workflow_login import _clear_and_type as impl

    return impl(text, target_key, settings, keyboard=keyboard)


# ------------------------------------------------------------------


@pytest.mark.parametrize("existing", ["", "a", "old", "P@ssw0rd!", "x" * 40])
def test_field_holds_exactly_the_new_text(existing):
    """기존 내용 길이와 무관하게 최종 값은 새로 넣은 문자열 하나뿐이어야 한다.

    이게 이 버그의 본질이다 - backspace 1회는 '한 글자'만 지우므로, 남은 글자
    뒤에 새 값이 이어 붙어 전혀 다른 문자열이 전송된다.
    """
    field = FakeTextField(existing)
    _clear_and_type("NEWPASS", "password_input", _settings(), FakeKeyboard(field))
    assert field.text == "NEWPASS", f"기존={existing!r} -> 결과={field.text!r}"


def test_single_char_password_after_longer_leftover():
    """한 글자짜리 값을 넣는데 앞에 긴 잔여 문자열이 있어도 그 한 글자만 남아야 한다."""
    field = FakeTextField("remembered")
    _clear_and_type("Q", "password_input", _settings(), FakeKeyboard(field))
    assert field.text == "Q", field.text


def test_works_without_ctrl_a_support():
    """Ctrl+A 를 안 먹는 레거시 컨트롤에서도 지워져야 한다.

    RCS 는 MFC 계열이라 표준 단축키가 통하지 않는 컨트롤이 있다(pywinauto UIA/win32
    백엔드가 ComboBox/Button 을 노출 못 한 것과 같은 계열). Ctrl+A 하나에만 기대면
    그런 컨트롤에서 조용히 예전 동작으로 되돌아간다.
    """
    field = FakeTextField("leftover-value")
    _clear_and_type(
        "NEW", "password_input", _settings(), FakeKeyboard(field, supports_ctrl_a=False)
    )
    assert field.text == "NEW", field.text


def test_dry_run_does_not_touch_field():
    """SAFE_MODE(dry-run)에서는 키를 하나도 보내지 않는다."""
    field = FakeTextField("untouched")
    _clear_and_type(
        "NEW", "password_input", _settings(action_enabled=False), FakeKeyboard(field)
    )
    assert field.text == "untouched", field.text


def test_empty_text_clears_field():
    """빈 문자열을 넣으면 필드는 비어야 한다(잔여물이 남으면 안 된다)."""
    field = FakeTextField("stale")
    _clear_and_type("", "userid_input", _settings(), FakeKeyboard(field))
    assert field.text == "", field.text
