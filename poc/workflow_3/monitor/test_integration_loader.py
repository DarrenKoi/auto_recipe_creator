"""office adapter 로더의 실패 보고 단위 테스트 - Mac 에서 오피스 없이 돈다.

핵심 계약은 하나다: **office_* 모듈이 import 중에 죽으면 그 파일의 몇 번째 줄에서
죽었는지를 보고해야 한다.** office_* 는 gitignore 라 오피스 PC 에만 존재하고 개발
PC 에서 열어볼 수 없으므로, 콘솔 보고가 유일한 진단 채널이다. 타입과 메시지만
있으면(예: "TypeError: 'NoneType' object is not subscriptable") 어느 줄인지 좁힐
방법이 없다 - 2026-08-24 오피스에서 office_success_downloader 가 정확히 이 상태로
막혔다.

동시에 지켜야 하는 반대쪽 계약: **어떤 실패든 모니터 import 자체를 죽이면 안 된다.**
office adapter 부재/고장은 그 기능만 끄고 루프는 계속 돌아야 한다.

`uv run pytest poc/workflow_3/monitor/test_integration_loader.py` 로 실행.
"""

import sys
import textwrap

from poc.workflow_3.monitor.integration_loader import (
    _exc_location,
    load_office_integration,
)


def _write_module(tmp_path, name: str, body: str) -> None:
    """임시 office 모듈을 만들고 import 경로에 얹는다."""
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    sys.modules.pop(name, None)


def test_import_error_reports_failing_file_and_line(tmp_path, capsys):
    """모듈 레벨 예외의 file:line 이 콘솔에 나와야 한다 (오피스 증상 재현)."""
    _write_module(tmp_path, "boom_office_mod", '''
        import os

        _CFG = os.environ.get("DEFINITELY_NOT_SET_OFFICE_CFG")   # -> None
        ROOT = _CFG["root"]        # 5번째 줄: TypeError

        def make_thing():
            return None
    ''')

    result = load_office_integration(
        "boom", "boom_office_mod", required_attrs=("make_thing",)
    )
    out = capsys.readouterr().out

    assert result.available is False
    assert "status=import_error" in out
    # 진단의 본체: 파일명과 줄번호가 있어야 한다.
    assert "boom_office_mod.py:5" in out, out
    # 전체 traceback 에 실제 소스 라인까지 실려야 한다.
    assert '_CFG["root"]' in out, out


def test_import_error_does_not_kill_the_caller(tmp_path, capsys):
    """adapter 가 깨져도 예외를 밖으로 던지지 않는다 (루프가 계속 돌아야 한다)."""
    _write_module(tmp_path, "boom_office_mod2", '''
        raise RuntimeError("office MES 연결 실패")
    ''')

    result = load_office_integration("boom2", "boom_office_mod2")

    assert result.available is False
    assert result.module is None
    assert "status=import_error" in capsys.readouterr().out


def test_missing_module_stays_quiet_without_traceback(capsys):
    """모듈 자체가 없는 개발 PC 정상 케이스는 traceback 을 찍지 않는다.

    없는 것은 고장이 아니다. 여기서 traceback 을 찍으면 Mac 개발 중 매 실행마다
    노이즈가 쌓여 정작 진짜 import_error 를 못 보게 된다.
    """
    result = load_office_integration("nope", "poc.workflow_3.monitor.office_not_here_xyz")
    out = capsys.readouterr().out

    assert result.available is False
    assert "status=missing" in out
    assert "traceback" not in out.lower()


def test_missing_required_attr_is_distinct_from_import_error(tmp_path, capsys):
    """import 은 됐지만 팩토리가 없는 경우는 import_error 와 구분되어야 한다."""
    _write_module(tmp_path, "attrless_office_mod", '''
        VALUE = 1
    ''')

    result = load_office_integration(
        "attrless", "attrless_office_mod", required_attrs=("make_thing",)
    )
    out = capsys.readouterr().out

    assert result.available is False
    assert "status=missing_attr" in out
    assert "missing_attrs=make_thing" in out
    assert "status=import_error" not in out


def test_exc_location_returns_deepest_frame():
    """_exc_location 은 (중간이 아니라) 가장 깊은 프레임을 돌려준다."""
    def inner():
        raise ValueError("boom")

    def outer():
        inner()

    try:
        outer()
    except ValueError as exc:
        location = _exc_location(exc)

    # 가장 깊은 프레임 = inner 의 raise 줄. 파일은 이 테스트 파일이다.
    assert location.startswith(__file__), location
    assert location.rsplit(":", 1)[1].isdigit()


def test_exc_location_empty_when_no_traceback():
    """traceback 이 없는 예외에서도 죽지 않고 빈 문자열을 준다."""
    assert _exc_location(ValueError("no tb")) == ""
