"""rcp/msr 입력 이미지 office 다운로드 접점 + 동기 fetch (monitor glue).

align_img_from_rcp(등록 align key) / align_img_from_msr(측정 궤적)는 보정/점검의
1차 입력이다. 기본 계약은 office MES 가 align_images 트리에 직접 적재하는 것이지만,
MES 출력을 그 트리로 받지 못하는 환경에서는 office_rcp_msr_downloader 가 알람 시점에
그 트리로 내려받는다.

success_gather(consensus S 이미지)와 달리 **동기(blocking)** 다. rcp/msr 은 cycle 이
assets(feasibility/보정)를 읽기 *전에* 반드시 디스크에 있어야 하므로, async 로 fire 하면
feasibility 가 빈 트리를 읽어 '보정 불가' 오판을 낼 수 있다(이 모듈이 막으려는 바로 그
버그). 따라서 cycle 직전에 받아 완료를 보장한다.

office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를 죽이지 않는다
(alarm_source/notify/success_gather 와 동일 철학). office_rcp_msr_downloader 는
정위치(poc.workflow_3.monitor)에서 로드한다.
"""

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

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir) -> int:
        """eqp_id + recipe_id('<class>/<recipe>') 의 align_img_from_rcp / align_img_from_msr
        를 dest_dir 아래에 office MES 와 동일한 레이아웃으로 쓰고, 쓴 이미지 총개수를 반환한다.

        dest_dir 는 호출부가 넘기는 recipe leaf 경로 (`ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe>`).
        downloader 는 그 아래 align_img_from_rcp/(IMAP0001=OM, IMAP0002=SEM)와
        align_img_from_msr/(S*/E* + 숨김폴더 cond)를 채운다. 받을 게 없으면 0 을 반환한다."""
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


def gather_rcp_msr(eqp_id, recipe_id, settings: Workflow3Settings) -> bool:
    """recipe 의 rcp/msr 입력 이미지를 align_images 트리로 **동기** 다운로드한다.

    rcp_msr_gather_enabled off / recipe_id 없음 / downloader 부재면 아무것도 안 하고 False.
    다운로드가 (예외 없이) 끝나면 True. 예외는 삼키고(best-effort) False 를 반환해
    모니터 루프가 죽지 않게 한다 — rcp/msr 이 안 받아져도 cycle 은 캡처/점검을 진행한다.

    cycle 직전에 호출해 assets 읽기 전 디스크 적재를 보장하는 게 핵심이다(모듈 docstring 참고).
    """
    if not settings.rcp_msr_gather_enabled or not recipe_id or not RCP_MSR_DOWNLOADER_AVAILABLE:
        return False

    # recipe_id = '<class>/<recipe>' 라 ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe> 로 3단 중첩.
    dest_dir = ALIGN_IMAGES_DIR / eqp_id / recipe_id
    try:
        n_images = _DOWNLOADER.download_rcp_msr(eqp_id, recipe_id, dest_dir=dest_dir)
        print(f"[INFO] rcp/msr 다운로드 완료: EQP_ID={eqp_id} recipe={recipe_id} "
              f"images={n_images} dest={dest_dir}")
        return True
    except Exception as exc:
        print(f"[WARNING] rcp/msr 다운로드 예외: EQP_ID={eqp_id} recipe={recipe_id} error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="gather_error", level="warning",
            eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
        )
        return False


__all__ = ["RCP_MSR_DOWNLOADER_AVAILABLE", "RcpMsrDownloader", "gather_rcp_msr"]
