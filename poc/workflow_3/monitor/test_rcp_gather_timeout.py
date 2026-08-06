"""rcp 동기 다운로드의 bounded 대기 + in-flight 가드 테스트 (F3).

office downloader 없이 도는 단위 테스트다 - 모듈의 _DOWNLOADER 를 stub 으로
바꿔 느린/정상 다운로드를 흉내낸다.

`uv run python poc/workflow_3/monitor/test_rcp_gather_timeout.py` 로 직접 실행.
"""

import threading
import time
from types import SimpleNamespace

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import rcp_msr_gather as rmg


def _install_downloader(fn):
    """모듈의 downloader 를 교체하고 원복 함수를 돌려준다."""
    orig_dl = rmg._DOWNLOADER
    orig_flag = rmg.RCP_MSR_DOWNLOADER_AVAILABLE
    rmg._DOWNLOADER = SimpleNamespace(download_rcp_msr=fn)
    rmg.RCP_MSR_DOWNLOADER_AVAILABLE = True

    def _restore():
        rmg._DOWNLOADER = orig_dl
        rmg.RCP_MSR_DOWNLOADER_AVAILABLE = orig_flag
        with rmg._IN_FLIGHT_LOCK:
            rmg._IN_FLIGHT.clear()
    return _restore


def test_slow_download_returns_within_bound():
    """timeout 을 넘긴 다운로드는 bound 안에서 False 로 돌아온다(무한 정지 금지)."""
    def _slow(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        time.sleep(5.0)
        return 1

    restore = _install_downloader(_slow)
    try:
        settings = load_workflow3_settings()
        started = time.time()
        ok = rmg.gather_rcp_msr("EQP1", "C/R1", settings, timeout_sec=0.5)
        elapsed = time.time() - started
        assert ok is False, ok
        assert elapsed < 3.0, elapsed      # 5초 다운로드에 묶이지 않았다.
    finally:
        restore()
    print("[OK] test_slow_download_returns_within_bound")


def test_fast_download_returns_true():
    """timeout 안에 끝나면 True."""
    def _fast(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        return 3

    restore = _install_downloader(_fast)
    try:
        settings = load_workflow3_settings()
        assert rmg.gather_rcp_msr("EQP2", "C/R2", settings, timeout_sec=5.0) is True
    finally:
        restore()
    print("[OK] test_fast_download_returns_true")


def test_in_flight_guard_skips_concurrent_same_recipe():
    """같은 recipe 의 gather 가 아직 도는 중이면 새 gather 를 fire 하지 않는다.

    timeout 만 넣고 이 가드가 없으면, 시간 초과된 스레드가 계속 쓰는 동안 새
    스레드가 같은 dest_dir 에 겹쳐 써서 '보이는 정지'가 '조용한 부분읽기 경쟁'
    으로 바뀐다.
    """
    calls = []
    release = threading.Event()

    def _blocking(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        calls.append(eqp_id)
        release.wait(5.0)
        return 1

    restore = _install_downloader(_blocking)
    try:
        settings = load_workflow3_settings()
        # 1차: timeout 으로 포기하지만 스레드는 계속 돈다.
        rmg.gather_rcp_msr("EQP3", "C/R3", settings, timeout_sec=0.3)
        assert calls == ["EQP3"], calls
        # 2차: 같은 recipe - 아직 진행 중이므로 새 fire 없음.
        rmg.gather_rcp_msr("EQP3", "C/R3", settings, timeout_sec=0.3)
        assert calls == ["EQP3"], calls
    finally:
        release.set()
        time.sleep(0.1)
        restore()
    print("[OK] test_in_flight_guard_skips_concurrent_same_recipe")


def test_in_flight_guard_is_keyed_by_eqp_and_recipe():
    """가드 키는 (eqp_id, recipe_id) 다 - recipe_id 단독이면 안 된다.

    dest_dir 이 ALIGN_IMAGES_DIR/<eqp>/<recipe_id> 로 eqp-keyed 라서, 같은 recipe 라도
    장비가 다르면 서로 다른 디렉토리에 쓴다. EQP_A 의 다운로드가 timeout 으로 아직
    도는 중이어도 EQP_B 의 같은-recipe 다운로드는 다른 dest_dir 이라 반드시 fire 돼야
    한다. 반면 같은 EQP_A 가 같은 recipe 로 재진입하면 여전히 skip 돼야 한다(원래
    가드가 막으려던 진짜 경쟁).
    """
    calls = []
    release = threading.Event()

    def _blocking(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        calls.append(eqp_id)
        release.wait(5.0)
        return 1

    restore = _install_downloader(_blocking)
    try:
        settings = load_workflow3_settings()
        # EQP_A 다운로드 시작 - timeout 으로 포기하지만 스레드는 계속 돈다.
        rmg.gather_rcp_msr("EQP_A", "C/R_SHARED", settings, timeout_sec=0.3)
        assert calls == ["EQP_A"], calls

        # 같은 recipe 지만 다른 장비(EQP_B) - dest_dir 이 달라 반드시 fire 돼야 한다.
        rmg.gather_rcp_msr("EQP_B", "C/R_SHARED", settings, timeout_sec=0.3)
        assert calls == ["EQP_A", "EQP_B"], calls

        # 같은 (EQP_A, C/R_SHARED) 재진입은 여전히 skip 돼야 한다.
        rmg.gather_rcp_msr("EQP_A", "C/R_SHARED", settings, timeout_sec=0.3)
        assert calls == ["EQP_A", "EQP_B"], calls
    finally:
        release.set()
        time.sleep(0.1)
        restore()
    print("[OK] test_in_flight_guard_is_keyed_by_eqp_and_recipe")


def test_exception_in_download_returns_false():
    """다운로드 예외는 삼키고 False - 모니터 루프를 죽이지 않는다(기존 계약 유지)."""
    def _boom(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        raise RuntimeError("network down")

    restore = _install_downloader(_boom)
    try:
        settings = load_workflow3_settings()
        assert rmg.gather_rcp_msr("EQP4", "C/R4", settings, timeout_sec=5.0) is False
    finally:
        restore()
    print("[OK] test_exception_in_download_returns_false")


if __name__ == "__main__":
    test_slow_download_returns_within_bound()
    test_fast_download_returns_true()
    test_in_flight_guard_skips_concurrent_same_recipe()
    test_in_flight_guard_is_keyed_by_eqp_and_recipe()
    test_exception_in_download_returns_false()
    print("\n[OK] rcp gather timeout / in-flight 가드 테스트 통과")
