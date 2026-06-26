"""consensus gather 의 office 접점 + 비차단 fire (monitor glue).

align.consensus_gather 의 순수 orchestration 을 office 다운로더 해석(정위치)과
daemon thread 로 감싼다. office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를
죽이지 않는다(alarm_source/notify 와 동일 철학).

동일 recipe_id(=cache_key, class/recipe)에 대해 gather 가 이미 진행 중이면 skip 한다.
consensus pool 은 eqp 무관이라 같은 recipe 의 고정 .events_staging 경로를 모든 장비가
공유한다 — 두 스레드가 동시에 쓰면 partial promote 경쟁이 발생하므로 _IN_FLIGHT
레지스트리(recipe 단독 키)로 in-flight dedupe 를 보장한다.

(설계: poc/workflow_2/docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md §3-A)
"""

import threading

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.align.consensus_gather import gather_success_images
from poc.workflow_3.monitor.integration_loader import (
    load_office_integration,
    log_office_factory_error,
    log_office_factory_loaded,
)

LOG_COMPONENT = "consensus_gather"

# 동일 recipe 동시 gather 의 staging 경쟁 방지.
# (eqp_id, recipe_id) -> Thread. 살아있는 Thread 가 있으면 새 gather 는 skip.
_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict = {}  # recipe_id(class/recipe) -> Thread. 같은 recipe 동시 gather 의 staging 경쟁 방지(eqp 무관).


def _load_office_downloader():
    """SuccessDownloader 구현을 정위치에서 찾는다. 없으면 None.

    office_* 모듈은 gitignore 라 오피스 PC 에만 존재한다. 모듈은 인자 없는
    `make_success_downloader()` 팩토리를 노출해야 한다.
    """
    integration = load_office_integration(
        "office_success_downloader",
        "poc.workflow_3.monitor.office_success_downloader",
        required_attrs=("make_success_downloader",),
    )
    if not integration.available:
        return None

    factory = integration.attrs["make_success_downloader"]
    try:
        downloader = factory()
    except Exception as exc:
        log_office_factory_error("office_success_downloader", integration.module_path, exc)
        return None
    log_office_factory_loaded("office_success_downloader", integration.module_path)
    return downloader


_DOWNLOADER = _load_office_downloader()
DOWNLOADER_AVAILABLE = _DOWNLOADER is not None
if not DOWNLOADER_AVAILABLE:
    print("[INFO] success downloader 없음, consensus gather 비활성(개발 PC/미구현).")


def gather_success_async(eqp_id, recipe_id, settings: Workflow3Settings):
    """recipe 의 최근 성공 S 이미지 stage 를 daemon thread 로 비차단 실행한다.

    gather_enabled off / recipe_id 없음 / downloader 부재면 아무것도 안 하고 None.
    실제 fire 하면 시작된 Thread 를 반환한다(테스트 join 용). 예외는 thread 안에서 삼킨다.

    같은 recipe_id(eqp 무관)에 대해 gather thread 가 이미 살아있으면 skip(None 반환).
    고정 .events_staging 경로 공유로 인한 동시 쓰기/부분 promote 경쟁을 _IN_FLIGHT 레지스트리로 차단.
    """
    # 게이트 단락은 조용히 None 이면 "왜 캐시가 비나"를 콘솔에서 분간할 수 없다.
    # 각 사유를 명시 로깅해 오피스 1회 실행으로 어느 경계가 막혔는지 바로 드러낸다(진단).
    if not settings.gather_enabled:
        print(f"[INFO] consensus gather skip(gather_enabled=0): "
              f"EQP_ID={eqp_id} recipe={recipe_id}")
        return None
    if not recipe_id:
        print(f"[INFO] consensus gather skip(recipe_id 비어있음): EQP_ID={eqp_id}")
        return None
    if not DOWNLOADER_AVAILABLE:
        print(f"[INFO] consensus gather skip(downloader 부재/임포트시 적재 실패): "
              f"EQP_ID={eqp_id} recipe={recipe_id}")
        return None

    # dedupe 키 = recipe_id(=cache_key, class/recipe) 단독. consensus pool 은 eqp 무관이라
    # 같은 recipe 의 events/staging 경로를 모든 장비가 공유한다 — 키에 eqp 를 넣으면 두 장비가
    # 같은 recipe 를 동시에 gather 할 때 한 .events_staging 을 서로 다른 키로 동시에 써서
    # partial promote 경쟁이 난다. recipe 단독 키로 묶어 한 번만 fire 한다.
    key = recipe_id
    with _IN_FLIGHT_LOCK:
        # 죽은 entry 정리 (dict 를 소형으로 유지).
        dead = [k for k, t in _IN_FLIGHT.items() if not t.is_alive()]
        for k in dead:
            del _IN_FLIGHT[k]

        if key in _IN_FLIGHT and _IN_FLIGHT[key].is_alive():
            print(f"[INFO] consensus gather 이미 진행 중(skip): EQP_ID={eqp_id} recipe={recipe_id}")
            return None

        def _run():
            try:
                result = gather_success_images(
                    eqp_id, recipe_id,
                    downloader=_DOWNLOADER,
                    max_events=settings.gather_max_events,
                    refresh_ttl_sec=settings.consensus_refresh_ttl_sec,
                )
                print(f"[INFO] consensus gather: EQP_ID={eqp_id} recipe={recipe_id} "
                      f"reason={result.reason} events={result.n_events} images={result.n_images}")
            except Exception as exc:
                print(f"[WARNING] consensus gather 예외: EQP_ID={eqp_id}, error={exc}")
                log_work2_event(
                    component=LOG_COMPONENT, message="gather_error", level="warning",
                    eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
                )

        thread = threading.Thread(target=_run, daemon=True)
        _IN_FLIGHT[key] = thread
        # start 도 lock 안에서: 등록-시작 사이 틈에 다른 호출자의 prune 이
        # 미시작 thread(is_alive()=False)를 지우고 중복 fire 하는 창을 닫는다.
        thread.start()

    return thread


def _cache_has_min_events(eqp_id, recipe_id) -> bool:
    """events/ 에 S 이미지가 1장 이상 있나(채워졌는지 거친 판정)."""
    from poc.workflow_3.align.consensus_gather import count_staged_events
    _, n_images = count_staged_events(eqp_id, recipe_id)
    return n_images > 0   # docstring/Task5 규약대로 S 이미지 수 기준(이벤트 수 아님).


def wait_for_gather(eqp_id, recipe_id, timeout) -> bool:
    """진행 중인 gather thread 를 bounded join 한 뒤 캐시가 채워졌는지 반환한다.

    lock 안에서는 thread 스냅샷만 -- join 은 lock 밖에서(데드락 방지). 알람 시점에 이미
    async fire 됐으면 그 thread 를 join(중복 fetch 회피). 없고 캐시도 비면 1회 fire 후 join.
    반환 bool = join 후 events/ 에 S 가 있나(resolver 는 True 일 때만 crop 재로드).
    """
    if not recipe_id or not DOWNLOADER_AVAILABLE:
        return _cache_has_min_events(eqp_id, recipe_id)

    key = recipe_id   # eqp 무관 pool — recipe 단독 키(gather_success_async 와 동일).
    with _IN_FLIGHT_LOCK:
        thread = _IN_FLIGHT.get(key)
        if thread is not None and not thread.is_alive():
            thread = None
    # lock 해제 후 join/fire -- lock 안에서 join 이나 gather_success_async 호출 금지(데드락).
    if thread is None and not _cache_has_min_events(eqp_id, recipe_id):
        # 알람 fire 가 없었거나 이미 끝났는데 캐시가 비어 있음 -> 1회 fire(내부에서 thread 등록).
        from poc.workflow_3.config import load_workflow3_settings
        thread = gather_success_async(eqp_id, recipe_id, load_workflow3_settings())

    if thread is not None:
        thread.join(timeout)
    return _cache_has_min_events(eqp_id, recipe_id)


__all__ = ["DOWNLOADER_AVAILABLE", "gather_success_async", "wait_for_gather"]
