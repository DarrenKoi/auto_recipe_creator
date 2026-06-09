"""consensus gather 의 office 접점 + 비차단 fire (monitor glue).

vision.consensus_gather 의 순수 orchestration 을 office 다운로더 해석(2단 fallback)과
daemon thread 로 감싼다. office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를
죽이지 않는다(alarm_source/notify 와 동일 철학).

(설계: poc/workflow_2/docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md §3-A)
"""

import importlib
import threading

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.vision.consensus_gather import gather_success_images

LOG_COMPONENT = "consensus_gather"


def _load_office_downloader():
    """SuccessDownloader 구현을 정위치 -> legacy 순서로 찾는다. 없으면 None.

    office_* 모듈은 gitignore 라 오피스 PC 에만 존재한다. 모듈은 인자 없는
    `make_success_downloader()` 팩토리를 노출해야 한다.
    """
    for module_path, is_legacy in (
        ("poc.workflow_3.monitor.office_success_downloader", False),
        ("poc.workflow_1.office_success_downloader", True),
    ):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        factory = getattr(module, "make_success_downloader", None)
        if factory is None:
            continue
        if is_legacy:
            print("[WARNING] office_success_downloader 가 legacy 위치(workflow_1)에서 "
                  "로드됨, poc/workflow_3/monitor/ 로 복사하세요.")
        try:
            return factory()
        except Exception as exc:
            print(f"[WARNING] success downloader 생성 실패: {exc}")
            return None
    return None


_DOWNLOADER = _load_office_downloader()
DOWNLOADER_AVAILABLE = _DOWNLOADER is not None
if not DOWNLOADER_AVAILABLE:
    print("[INFO] success downloader 없음, consensus gather 비활성(개발 PC/미구현).")


def gather_success_async(eqp_id, recipe_id, settings: Workflow3Settings):
    """recipe 의 최근 성공 S 이미지 stage 를 daemon thread 로 비차단 실행한다.

    gather_enabled off / recipe_id 없음 / downloader 부재면 아무것도 안 하고 None.
    실제 fire 하면 시작된 Thread 를 반환한다(테스트 join 용). 예외는 thread 안에서 삼킨다.
    """
    if not settings.gather_enabled or not recipe_id or not DOWNLOADER_AVAILABLE:
        return None

    def _run():
        try:
            result = gather_success_images(
                eqp_id, recipe_id,
                downloader=_DOWNLOADER,
                max_events=settings.gather_max_events,
            )
            print(f"[INFO] consensus gather: EQP_ID={eqp_id} recipe={recipe_id} "
                  f"reason={result.reason} events={result.n_events} images={result.n_images}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_done",
                eqp_id=eqp_id, recipe_id=recipe_id, reason=result.reason,
                n_events=result.n_events, n_images=result.n_images,
            )
        except Exception as exc:
            print(f"[WARNING] consensus gather 예외: EQP_ID={eqp_id}, error={exc}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_error", level="warning",
                eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


__all__ = ["DOWNLOADER_AVAILABLE", "gather_success_async"]
