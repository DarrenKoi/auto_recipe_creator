"""seed_env_from_constants 규약 + 진입점 상수 블록의 env 이름 대조.

가장 중요한 것은 마지막 테스트다. 상수 블록의 env 이름에 오타가 있으면 그 knob 은
**조용히 아무것도 하지 않는다** - 켰다고 생각한 기능이 안 켜지는데 에러도 안 난다.
그래서 진입점이 쓰는 이름이 실제 reader 가 읽는 이름 집합에 있는지 확인한다.

    uv run pytest poc/workflow_3/util/test_env_utils.py
"""

import os
import re
from pathlib import Path

import pytest

from poc.workflow_3.util.env_utils import as_env_value, seed_env_from_constants

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("T_ONE", "T_ZERO", "T_NONE", "T_EMPTY", "T_BOOL", "T_LIST"):
        monkeypatch.delenv(name, raising=False)


def test_상수가_env_로_흐른다():
    ns = {"A": 5}
    seed_env_from_constants(ns, (("A", "T_ONE"),))
    assert os.environ["T_ONE"] == "5"


def test_셸_env_가_항상_이긴다(monkeypatch):
    monkeypatch.setenv("T_ONE", "99")
    seed_env_from_constants({"A": 5}, (("A", "T_ONE"),))
    assert os.environ["T_ONE"] == "99"


def test_숫자_0_은_유효한_값이라_주입된다():
    """명시적 off 를 미설정과 구분해야 한다 - 0 을 skip 하면 기능을 끌 수가 없다."""
    seed_env_from_constants({"A": 0}, (("A", "T_ZERO"),))
    assert os.environ["T_ZERO"] == "0"


def test_None_과_빈문자열은_미설정이라_skip():
    seed_env_from_constants({"A": None, "B": "  "}, (("A", "T_NONE"), ("B", "T_EMPTY")))
    assert "T_NONE" not in os.environ
    assert "T_EMPTY" not in os.environ


def test_bool_은_1_0_으로_내려간다():
    seed_env_from_constants({"A": True, "B": False}, (("A", "T_BOOL"), ("B", "T_ZERO")))
    assert os.environ["T_BOOL"] == "1"
    assert os.environ["T_ZERO"] == "0"


def test_리스트는_콤마로_합쳐진다():
    seed_env_from_constants({"A": ["x", "y"]}, (("A", "T_LIST"),))
    assert os.environ["T_LIST"] == "x,y"


def test_무시된_상수는_콘솔에_보고된다(monkeypatch, capsys):
    monkeypatch.setenv("T_ONE", "99")
    seed_env_from_constants({"A": 5}, (("A", "T_ONE"),), label="테스트")
    out = capsys.readouterr().out
    assert "무시" in out and "T_ONE" in out


def test_as_env_value_미설정():
    assert as_env_value(None) == ""
    assert as_env_value("") == ""
    assert as_env_value(0) == "0"      # 숫자 0 은 값이다


# ---------------------------------------------------------------------------
# producer/consumer 대조 - 오타 잡는 진짜 게이트
# ---------------------------------------------------------------------------

def _env_names_read_by(*rel_paths) -> set:
    """대상 파일들이 os.environ/env_* 로 읽는 env 이름을 전부 모은다."""
    pattern = re.compile(
        r'(?:os\.environ\.get|os\.getenv|env_flag|env_int|env_float|_env_str|_env_int|_env_float)'
        r'\(\s*"([A-Z][A-Z0-9_]*)"'
    )
    names = set()
    for rel in rel_paths:
        names |= set(pattern.findall((_REPO_ROOT / rel).read_text(encoding="utf-8")))
    return names


def test_align_fail_monitor_상수의_env_이름이_실제_reader_와_일치한다():
    from poc.workflow_3.monitor import align_fail_monitor as mod

    declared = {env for _, env in mod._CONST_TO_ENV}
    # 루프의 env 를 실제로 읽는 모듈 전부. config.py 하나가 아니다 -
    # SAFE_MODE 는 runner 가, REPLAY_CSV 는 alarm_source 가 읽는다.
    readable = _env_names_read_by(
        "poc/workflow_3/config.py",
        "poc/workflow_3/runner/workflow_config.py",
        "poc/workflow_3/monitor/alarm_source.py",
        "poc/workflow_3/vlm/ui_venus_mai_locator.py",
    )
    # LOCATOR_COMBO 는 호출 시점에 모듈 상수를 통해 읽혀 정규식에 안 잡힌다.
    readable.add("VLM_LOCATOR_COMBO")

    orphans = sorted(declared - readable)
    assert not orphans, (
        f"이 env 이름을 읽는 코드가 없다 - 상수를 바꿔도 아무 일도 안 일어난다: {orphans}"
    )


def test_상수명은_블록에_실제로_정의돼_있다():
    """표에만 있고 상수가 없으면 영원히 skip 되는 죽은 줄이 된다."""
    from poc.workflow_3.monitor import align_fail_monitor as mod

    missing = [c for c, _ in mod._CONST_TO_ENV if c not in vars(mod)]
    assert not missing, f"표에 있으나 상수 블록에 없음: {missing}"
