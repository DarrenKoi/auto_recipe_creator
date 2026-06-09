"""success_gather 합성 self-test — gating + office 2단 loader 검증(스레드 join).

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/monitor/test_success_gather.py
"""

import dataclasses
import sys
import types
from pathlib import Path

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import success_gather
from poc.workflow_3.vision.consensus_gather import GatherResult


def _settings(**overrides):
    """기본 settings 에 일부 필드만 바꿔 만든다(frozen dataclass → replace)."""
    return dataclasses.replace(load_workflow3_settings(), **overrides)


def _install_recording_gather():
    """gather_success_images 를 호출 인자 기록용 fake 로 교체하고 calls 리스트를 돌려준다."""
    calls = []

    def _fake(eqp_id, recipe_id, *, downloader, max_events):
        calls.append({"eqp_id": eqp_id, "recipe_id": recipe_id,
                      "downloader": downloader, "max_events": max_events})
        return GatherResult(eqp_id, recipe_id, Path("x"), 1, 2, "ok")

    success_gather.gather_success_images = _fake
    return calls


def test_fires_when_enabled():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    sentinel = object()
    success_gather._DOWNLOADER = sentinel
    st = _settings(gather_enabled=True, gather_max_events=7)

    t = success_gather.gather_success_async("EQP1", "CLS/RCP", st)
    if t is not None:
        t.join(timeout=5)
    ok = (
        t is not None
        and len(calls) == 1
        and calls[0]["eqp_id"] == "EQP1"
        and calls[0]["recipe_id"] == "CLS/RCP"
        and calls[0]["downloader"] is sentinel
        and calls[0]["max_events"] == 7
    )
    print(f"[{'PASS' if ok else 'FAIL'}] fires_when_enabled: thread={t is not None} calls={len(calls)}")
    return ok


def test_skips_when_disabled():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    success_gather._DOWNLOADER = object()
    t = success_gather.gather_success_async("EQP1", "CLS/RCP", _settings(gather_enabled=False))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_when_disabled: thread={t} calls={len(calls)}")
    return ok


def test_skips_without_recipe():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    success_gather._DOWNLOADER = object()
    t = success_gather.gather_success_async("EQP1", "", _settings(gather_enabled=True))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_recipe: thread={t} calls={len(calls)}")
    return ok


def test_skips_without_downloader():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = False
    success_gather._DOWNLOADER = None
    t = success_gather.gather_success_async("EQP1", "CLS/RCP", _settings(gather_enabled=True))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_downloader: thread={t} calls={len(calls)}")
    return ok


def test_loader_canonical():
    """정위치 모듈의 make_success_downloader 팩토리 결과를 돌려준다."""
    mod = types.ModuleType("poc.workflow_3.monitor.office_success_downloader")
    sentinel = object()
    mod.make_success_downloader = lambda: sentinel
    sys.modules["poc.workflow_3.monitor.office_success_downloader"] = mod
    try:
        got = success_gather._load_office_downloader()
        ok = got is sentinel
    finally:
        del sys.modules["poc.workflow_3.monitor.office_success_downloader"]
    print(f"[{'PASS' if ok else 'FAIL'}] loader_canonical: got_is_sentinel={ok}")
    return ok


def main():
    print("[INFO] success_gather self-test 시작")
    results = [
        test_fires_when_enabled(),
        test_skips_when_disabled(),
        test_skips_without_recipe(),
        test_skips_without_downloader(),
        test_loader_canonical(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
