"""rcp(+선택적 msr) 입력 이미지 office 다운로드 접점 + 동기 fetch (monitor glue).

align_img_from_rcp(등록 align key)는 보정/점검의 런타임 입력이다. 보정/feasibility 는
라이브 캡처 프레임에 consensus(우선)/rcp(폴백) 템플릿을 매칭하며, align_img_from_msr
(측정 궤적)은 런타임에서 소비하지 않는다. 따라서 프로덕션 gather 는 rcp 만 받는다
(include_msr=False 기본). msr 은 오프라인 벤치(golden set 확장)에서만 fetch_msr_offline.py
로 include_msr=True 로 받는다.

기본 계약은 office MES 가 align_images 트리에 직접 적재하는 것이지만, MES 출력을 그
트리로 받지 못하는 환경에서는 office_rcp_msr_downloader 가 알람 시점에 그 트리로 내려받는다.

success_gather(consensus S 이미지)와 달리 **동기(blocking)** 다. rcp 는 cycle 이
assets(feasibility/보정)를 읽기 *전에* 반드시 디스크에 있어야 하므로, async 로 fire 하면
feasibility 가 빈 트리를 읽어 '보정 불가' 오판을 낼 수 있다. 따라서 cycle 직전에 받아
완료를 보장한다.

office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를 죽이지 않는다.
office_rcp_msr_downloader 는 정위치(poc.workflow_3.monitor)에서 로드한다.
"""

import threading
from typing import Protocol

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.integration_loader import (
    load_office_integration,
    log_office_factory_error,
    log_office_factory_loaded,
)

LOG_COMPONENT = "rcp_msr_gather"


class RcpMsrDownloader(Protocol):
    """recipe 의 등록 align key + 측정 궤적을 align_images 트리로 쓰는 office 구현 계약."""

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr: bool = True) -> int:
        """eqp_id + recipe_id('<class>/<recipe>') 의 align_img_from_rcp 를 dest_dir 아래에
        office MES 와 동일한 레이아웃으로 쓰고, 쓴 이미지 총개수를 반환한다.

        include_msr=True 일 때만 align_img_from_msr(측정 궤적, S*/E* + 숨김폴더 cond)도 함께
        받는다. 프로덕션 루프는 msr 을 소비하지 않으므로 include_msr=False(rcp 만)로 부른다.
        오프라인 벤치(golden set 확장)에서만 include_msr=True 로 부른다.

        dest_dir 는 호출부가 넘기는 recipe leaf 경로(`ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe>`).
        받을 게 없으면 0 을 반환한다."""
        ...


def _load_office_downloader():
    """RcpMsrDownloader 구현을 정위치에서 찾는다. 없으면 None.

    office_* 모듈은 gitignore 라 오피스 PC 에만 존재한다. 모듈은 인자 없는
    `make_rcp_msr_downloader()` 팩토리를 노출해야 한다.
    """
    integration = load_office_integration(
        "office_rcp_msr_downloader",
        "poc.workflow_3.monitor.office_rcp_msr_downloader",
        required_attrs=("make_rcp_msr_downloader",),
    )
    if not integration.available:
        return None

    factory = integration.attrs["make_rcp_msr_downloader"]
    try:
        downloader = factory()
    except Exception as exc:
        log_office_factory_error("office_rcp_msr_downloader", integration.module_path, exc)
        return None
    log_office_factory_loaded("office_rcp_msr_downloader", integration.module_path)
    return downloader


_DOWNLOADER = _load_office_downloader()
RCP_MSR_DOWNLOADER_AVAILABLE = _DOWNLOADER is not None
if not RCP_MSR_DOWNLOADER_AVAILABLE:
    print("[INFO] rcp/msr downloader 없음 (개발 PC/미구현). "
          "rcp/msr 은 office MES 가 align_images 트리에 직접 적재해야 합니다.")

# (eqp_id, recipe_id) -> Thread. 진행 중인 gather 가 있으면 새로 fire 하지 않는다 - timeout 으로
# 포기한 스레드가 같은 dest_dir 에 계속 쓰는 동안 같은 (eqp, recipe) 로 새 스레드가 겹쳐
# 쓰면 부분읽기 경쟁이 난다. dest_dir 이 eqp-keyed 라 success_gather(recipe 단독 키, consensus
# 캐시가 eqp 무관)와는 키 구성이 다르다 - 여기서 recipe_id 단독 키를 쓰면 서로 다른
# dest_dir 에 쓰는 다른 장비의 다운로드까지 잘못 skip 된다.
_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict = {}


def gather_rcp_msr(
    eqp_id, recipe_id, settings: Workflow3Settings, *,
    include_msr: bool = False, timeout_sec=None,
) -> bool:
    """recipe 의 rcp 입력 이미지를 align_images 트리로 **동기** 다운로드한다.

    동기 계약은 유지된다 - cycle 이 assets(feasibility/보정)를 읽기 전에 디스크
    적재를 보장해야 하기 때문이다. 다만 대기는 bounded 다: daemon thread 로 돌리고
    timeout_sec 만큼만 join 한다(success_gather.wait_for_gather 와 같은 관용구).

    timeout_sec=None 이면 무한 대기(기존 동작) - 오프라인 벤치 fetch_msr_offline.py
    는 수 분짜리 msr 다운로드가 정상이라 상한을 두지 않는다. 모니터는
    settings.rcp_gather_timeout_sec 를 넘긴다.

    반환 True = 시간 안에 예외 없이 끝남. False = 게이트 미충족/예외/시간 초과.
    시간 초과 시 스레드는 계속 돌지만 루프는 진행한다 - assets 가 없거나 부분일 수
    있어 feasibility 가 '보정 불가' 오판을 낼 수 있다(알람 1건의 bounded 오답이
    전체 루프 무한 정지보다 낫다는 판단).
    """
    if not settings.rcp_msr_gather_enabled or not recipe_id or not RCP_MSR_DOWNLOADER_AVAILABLE:
        return False

    # recipe_id = '<class>/<recipe>' 라 ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe> 로 3단 중첩.
    dest_dir = ALIGN_IMAGES_DIR / eqp_id / recipe_id
    outcome = {"ok": False}

    def _run():
        try:
            n_images = _DOWNLOADER.download_rcp_msr(
                eqp_id, recipe_id, dest_dir=dest_dir, include_msr=include_msr
            )
            kind = "rcp+msr" if include_msr else "rcp"
            print(f"[INFO] {kind} 다운로드 완료: EQP_ID={eqp_id} recipe={recipe_id} "
                  f"images={n_images} dest={dest_dir}")
            outcome["ok"] = True
        except Exception as exc:
            print(f"[WARNING] rcp/msr 다운로드 예외: EQP_ID={eqp_id} recipe={recipe_id} error={exc}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_error", level="warning",
                eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
            )

    # (eqp_id, recipe_id) 쌍으로 키를 잡는다. success_gather 의 consensus 캐시와 달리
    # 이 모듈의 dest_dir 은 ALIGN_IMAGES_DIR/<eqp>/<recipe_id> 로 eqp-keyed 다 - 같은
    # recipe 라도 장비가 다르면 서로 다른 디렉토리에 쓰므로 실제로는 겹쳐 쓰지 않는다.
    # recipe_id 단독 키를 쓰면 tool A 의 다운로드가 timeout 으로 아직 살아있는 동안 같은
    # recipe 로 알람이 난 tool B 의 다운로드가 통째로 skip 되어(다른 디렉토리인데도) B 가
    # 참조 이미지 없이 진행하는 오탐을 낳는다. 이 가드가 실제로 막아야 하는 경쟁은 같은
    # (eqp, recipe) 가 재진입하며 이전 다운로드가 아직 같은 dest_dir 에 쓰는 중인 경우뿐이다.
    key = (eqp_id, recipe_id)
    with _IN_FLIGHT_LOCK:
        dead = [k for k, t in _IN_FLIGHT.items() if not t.is_alive()]
        for k in dead:
            del _IN_FLIGHT[k]
        if key in _IN_FLIGHT and _IN_FLIGHT[key].is_alive():
            print(f"[INFO] rcp gather 이미 진행 중(skip): EQP_ID={eqp_id} recipe={recipe_id}")
            return False
        thread = threading.Thread(target=_run, daemon=True)
        _IN_FLIGHT[key] = thread
        # start 도 lock 안에서 - 등록과 시작 사이 틈에 다른 호출자의 prune 이
        # 미시작 thread 를 지우고 중복 fire 하는 창을 닫는다.
        thread.start()

    thread.join(timeout_sec)
    if thread.is_alive():
        print(f"[WARNING] rcp 다운로드 시간 초과({timeout_sec}s) - 받은 만큼으로 진행: "
              f"EQP_ID={eqp_id} recipe={recipe_id}")
        return False
    return outcome["ok"]


__all__ = ["RCP_MSR_DOWNLOADER_AVAILABLE", "RcpMsrDownloader", "gather_rcp_msr"]
