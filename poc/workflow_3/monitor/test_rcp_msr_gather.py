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


class _LegacyDownloader:
    """오피스 PC 의 구버전 사본 — include_msr 키워드를 아직 모른다."""

    def __init__(self):
        self.calls = []

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir):
        self.calls.append({"eqp_id": eqp_id, "recipe_id": recipe_id, "dest_dir": dest_dir})
        return 2


class _OpaqueLegacyDownloader(_LegacyDownloader):
    """서명을 읽을 수 없는 구버전(C 구현/래퍼 흉내) — TypeError 폴백 경로."""

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name != "download_rcp_msr":
            return attr

        class _Opaque:
            """inspect.signature 가 실패하도록 __call__ 만 노출한다."""

            def __call__(self, *args, **kwargs):
                return attr(*args, **kwargs)

        return _Opaque()


def test_legacy_downloader_without_include_msr():
    """include_msr 를 모르는 구버전 사본에도 rcp 다운로드가 성공해야 한다.

    회귀 방지: office_rcp_msr_downloader.py 는 gitignore 라 저장소와 버전이 어긋난다.
    호출부가 include_msr 를 무조건 붙이면 구버전에서 TypeError 로 통째로 실패해
    rcp 가 한 장도 안 받아지고 feasibility 가 '자산 없음' 을 오판한다.
    """
    fake = _LegacyDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = ret is True and len(fake.calls) == 1
    print(f"[{'PASS' if ok else 'FAIL'}] legacy_downloader_without_include_msr: "
          f"ret={ret} calls={len(fake.calls)}")
    return ok


def test_legacy_downloader_unreadable_signature():
    """서명을 못 읽으면 include_msr 로 시도했다가 TypeError 를 보고 인자 없이 재시도한다."""
    fake = _OpaqueLegacyDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = ret is True and len(fake.calls) == 1
    print(f"[{'PASS' if ok else 'FAIL'}] legacy_downloader_unreadable_signature: "
          f"ret={ret} calls={len(fake.calls)}")
    return ok


class _InnerTypeErrorDownloader:
    """downloader **내부** 에서 난 TypeError 는 재시도 대상이 아니다."""

    def __init__(self):
        self.calls = 0

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True):
        self.calls += 1
        raise TypeError("내부 버그: unsupported operand type(s)")


def test_inner_type_error_not_retried():
    """downloader 내부 TypeError 를 '구버전' 으로 오인해 재호출하면 안 된다(부작용 2회)."""
    fake = _InnerTypeErrorDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    ret = rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = ret is False and fake.calls == 1
    print(f"[{'PASS' if ok else 'FAIL'}] inner_type_error_not_retried: "
          f"ret={ret} calls={fake.calls}")
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
        test_legacy_downloader_without_include_msr(),
        test_legacy_downloader_unreadable_signature(),
        test_inner_type_error_not_retried(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
