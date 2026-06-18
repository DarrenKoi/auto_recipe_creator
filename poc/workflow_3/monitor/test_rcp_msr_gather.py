"""rcp_msr_gather 합성 self-test — gating + office 2단 loader + dest 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/monitor/test_rcp_msr_gather.py
"""

import dataclasses
import sys
import types

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import rcp_msr_gather


def _settings(**overrides):
    """기본 settings 에 일부 필드만 바꿔 만든다(frozen dataclass → replace)."""
    return dataclasses.replace(load_workflow3_settings(), **overrides)


class _RecordingDownloader:
    """download_rcp_msr 호출 인자를 기록하는 fake downloader."""

    def __init__(self):
        self.calls = []

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True):
        self.calls.append({
            "eqp_id": eqp_id, "recipe_id": recipe_id,
            "dest_dir": dest_dir, "include_msr": include_msr,
        })
        return 4  # 받은 이미지 수(rcp 2 + msr 2 가정).


def test_fires_when_enabled():
    """게이트 통과 시 동기로 호출되고 dest_dir 가 ALIGN_IMAGES_DIR/<eqp>/<recipe> 다."""
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    st = _settings(rcp_msr_gather_enabled=True)

    ok_ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", st)
    expected_dest = ALIGN_IMAGES_DIR / "EQP1" / "CLS/RCP"
    ok = (
        ok_ret is True
        and len(fake.calls) == 1
        and fake.calls[0]["eqp_id"] == "EQP1"
        and fake.calls[0]["recipe_id"] == "CLS/RCP"
        and fake.calls[0]["dest_dir"] == expected_dest
    )
    print(f"[{'PASS' if ok else 'FAIL'}] fires_when_enabled: ret={ok_ret} "
          f"calls={len(fake.calls)} dest_ok={fake.calls and fake.calls[0]['dest_dir'] == expected_dest}")
    return ok


def test_skips_when_disabled():
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=False))
    ok = ret is False and len(fake.calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_when_disabled: ret={ret} calls={len(fake.calls)}")
    return ok


def test_skips_without_recipe():
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "", _settings(rcp_msr_gather_enabled=True))
    ok = ret is False and len(fake.calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_recipe: ret={ret} calls={len(fake.calls)}")
    return ok


def test_skips_without_downloader():
    rcp_msr_gather._DOWNLOADER = None
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = False
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = ret is False
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_downloader: ret={ret}")
    return ok


def test_swallows_downloader_exception():
    """downloader 예외는 삼키고 False 를 반환해 루프가 죽지 않는다(best-effort)."""
    class _Boom:
        def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True):
            raise RuntimeError("FTP timeout")

    rcp_msr_gather._DOWNLOADER = _Boom()
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = ret is False
    print(f"[{'PASS' if ok else 'FAIL'}] swallows_downloader_exception: ret={ret}")
    return ok


def test_loader_canonical():
    """정위치 모듈의 make_rcp_msr_downloader 팩토리 결과를 돌려준다."""
    mod = types.ModuleType("poc.workflow_3.monitor.office_rcp_msr_downloader")
    sentinel = object()
    mod.make_rcp_msr_downloader = lambda: sentinel
    sys.modules["poc.workflow_3.monitor.office_rcp_msr_downloader"] = mod
    try:
        got = rcp_msr_gather._load_office_downloader()
        ok = got is sentinel
    finally:
        del sys.modules["poc.workflow_3.monitor.office_rcp_msr_downloader"]
    print(f"[{'PASS' if ok else 'FAIL'}] loader_canonical: got_is_sentinel={ok}")
    return ok


def test_production_requests_rcp_only():
    """기본(프로덕션) 호출은 include_msr=False 로 rcp 만 받는다."""
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = len(fake.calls) == 1 and fake.calls[0]["include_msr"] is False
    print(f"[{'PASS' if ok else 'FAIL'}] production_requests_rcp_only: "
          f"include_msr={fake.calls[0]['include_msr'] if fake.calls else '-'}")
    return ok


def test_include_msr_propagates():
    """include_msr=True 를 주면 downloader 까지 그대로 전달된다(오프라인 벤치용)."""
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    rcp_msr_gather.gather_rcp_msr(
        "EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True), include_msr=True
    )
    ok = len(fake.calls) == 1 and fake.calls[0]["include_msr"] is True
    print(f"[{'PASS' if ok else 'FAIL'}] include_msr_propagates: "
          f"include_msr={fake.calls[0]['include_msr'] if fake.calls else '-'}")
    return ok


def main():
    print("[INFO] rcp_msr_gather self-test 시작")
    results = [
        test_fires_when_enabled(),
        test_skips_when_disabled(),
        test_skips_without_recipe(),
        test_skips_without_downloader(),
        test_swallows_downloader_exception(),
        test_loader_canonical(),
        test_production_requests_rcp_only(),
        test_include_msr_propagates(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
