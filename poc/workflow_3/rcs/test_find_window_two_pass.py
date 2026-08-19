"""최소화/숨김 창까지 훑는 2-pass 창 탐색 회귀 테스트.

오피스 증상: RCS 메인 창이 다른 창 뒤에 있거나 작업 표시줄로 내려가 있으면
`메인 RCS 창 없음` 으로 끝나, 엔지니어가 작업 표시줄 아이콘을 눌러 띄워야만
자동화가 이어졌다. 원인은 복원(activate_window 는 이미 SW_RESTORE 를 한다)이 아니라
**탐색** 이었다 - `visible_only=True` 로 한 번만 훑었다.

Windows/pywinauto 없이 Mac 에서 그대로 돈다(find_window_by_title_prefix 를 stub).

    uv run pytest poc/workflow_3/rcs/test_find_window_two_pass.py
"""

from poc.workflow_3.rcs import login_rcs_common as lrc


class _FakeWindow:
    """창 객체 자리표시자."""


def _stub_scanner(monkeypatch, *, visible_hit, hidden_hit, calls):
    """visible/hidden 스캔 결과를 따로 지정하는 stub 을 심는다."""

    def _scan(title_prefix, backends, *, visible_only=True, window_filter=None):
        calls.append(visible_only)
        hit = visible_hit if visible_only else hidden_hit
        if hit is None:
            return None, "", ""
        if window_filter is not None and not window_filter(hit, "RCS - MCD630"):
            return None, "", ""
        return hit, "RCS - MCD630", "uia"

    monkeypatch.setattr(lrc, "find_window_by_title_prefix", _scan)


def test_visible_hit_skips_hidden_scan(monkeypatch):
    """보이는 창에서 찾으면 2차 스캔은 돌지 않는다(후보가 적은 쪽 우선)."""
    win = _FakeWindow()
    calls = []
    _stub_scanner(monkeypatch, visible_hit=win, hidden_hit=None, calls=calls)

    found, title, backend = lrc.find_window_two_pass("RCS - ", label="메인 RCS 창")

    assert found is win
    assert title == "RCS - MCD630"
    assert backend == "uia"
    assert calls == [True], f"2차 스캔이 불필요하게 돌았다: {calls}"


def test_minimized_window_found_by_hidden_scan(monkeypatch):
    """보이는 창에 없어도 숨은/최소화 창 스캔에서 찾아야 한다 - 이 버그의 핵심."""
    win = _FakeWindow()
    calls = []
    _stub_scanner(monkeypatch, visible_hit=None, hidden_hit=win, calls=calls)

    found, title, _backend = lrc.find_window_two_pass("RCS - ", label="메인 RCS 창")

    assert found is win, "최소화된 창을 놓쳤다 - 엔지니어가 수동으로 띄워야 한다"
    assert title == "RCS - MCD630"
    assert calls == [True, False], f"2-pass 순서가 어긋남: {calls}"


def test_absent_window_returns_none_triple(monkeypatch):
    """양쪽 스캔 모두 미검출이면 (None, '', '') - 호출부 가드 계약 유지."""
    calls = []
    _stub_scanner(monkeypatch, visible_hit=None, hidden_hit=None, calls=calls)

    assert lrc.find_window_two_pass("RCS - ", label="메인 RCS 창") == (None, "", "")
    assert calls == [True, False]


def test_window_filter_applies_to_both_passes(monkeypatch):
    """window_filter 는 2차 스캔에도 걸려야 한다 - 엉뚱한 tool 창을 집으면 안 된다."""
    win = _FakeWindow()
    calls = []
    _stub_scanner(monkeypatch, visible_hit=None, hidden_hit=win, calls=calls)

    found, _title, _backend = lrc.find_window_two_pass(
        "RCS - ", label="tool 창", window_filter=lambda w, t: False,
    )

    assert found is None, "필터가 2차 스캔에서 무시됐다"
    assert calls == [True, False]
