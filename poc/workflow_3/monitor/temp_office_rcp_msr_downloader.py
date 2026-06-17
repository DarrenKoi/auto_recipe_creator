r"""[TEMPLATE] office_rcp_msr_downloader.py 채움용 견본 — RcpMsrDownloader 구현.

이 파일은 *템플릿* 이다. 실제 `office_rcp_msr_downloader.py` 는 gitignore 라 git 으로
전달이 안 되므로, 추적 가능한 `temp_` 접두 이름으로 견본을 싣는다.

오피스 PC 사용법:
  1. git pull
  2. copy poc\workflow_3\monitor\temp_office_rcp_msr_downloader.py ^
          poc\workflow_3\monitor\office_rcp_msr_downloader.py
  3. (Case 1 — download 함수가 내부에서 경로를 계산해 align_images 트리에 직접 적재)
     아래 두 호출은 그대로 두면 된다.

로드 경로: monitor/rcp_msr_gather.py 의 make_rcp_msr_downloader() 팩토리가
정위치(poc.workflow_3.monitor.office_rcp_msr_downloader) -> legacy 순으로 찾아
알람마다 사이클 직전 **동기** 호출한다.

[중요] download_align_images_from_rcp/msr 는 **idempotent** 해야 한다 — 같은
(eqp_id, recipe_id)로 여러 번 불려도 안전하고, 파일이 이미 트리에 있으면 FTP 를
생략하고 기존 경로를 반환해야 한다. 한 알람당 이 함수들은 최소 두 번 호출되기
때문이다:
  1) 여기(office_rcp_msr_downloader) — 사이클 직전, 보정/feasibility 가 읽기 전에 적재.
  2) office_rich_notify.send_cube_align_fail_info — 실패 알림에 이미지를 embed 할 때.
idempotent 가드(예: dest 에 IMAP000* 가 이미 있으면 그 경로 반환)가 없으면 알람마다
같은 이미지를 두 번 FTP 로 받는다(정상 동작이지만 낭비). 가드가 있으면 첫 호출만
실제 fetch, 이후 호출은 디스크 read 로 즉시 끝난다.

주의: 파일명이 temp_ 라 pytest 수집 대상이 아니다(test_ 접두였다면 수집됨). 그래도
office_rich_notify 가 없는 개발 PC 에서도 import 가 깨지지 않게 office import 는
가드한다(아래). 오피스에서 office_rcp_msr_downloader.py 로 복사하면 가드는 통과한다.
"""

import os
import time
from pathlib import Path

# msr 측정 이미지는 align fail 직후 tool 이 디스크에 늦게 쓰는 경우가 있어, 첫 fetch 가
# 0장이면 잠깐 쉬고 재시도한다(rcp 등록 키는 항상 존재하므로 재시도 대상 아님).
# 동기 gather 라 이 대기만큼 cycle 시작이 늦지만, 빈 msr 트리로 feasibility 를 오판하는
# 것보다 낫다. env 로 끄거나(횟수 0) 조절 가능.
MSR_RETRY_COUNT = int(os.environ.get("ALIGN_FAIL_MSR_RETRY_COUNT", "1"))
MSR_RETRY_DELAY_SEC = float(os.environ.get("ALIGN_FAIL_MSR_RETRY_DELAY_SEC", "2.0"))

# 기존 다운로드 함수 재사용 — 둘 다 (image_path_list, image_cond_path_list) 반환.
# 개발 PC(office_rich_notify 부재)에서도 이 파일이 import 되도록 가드한다.
try:
    from poc.workflow_3.monitor.office_rich_notify import (
        download_align_images_from_rcp,
        download_align_images_from_msr,
    )
    _OFFICE_FNS_AVAILABLE = True
except ImportError:
    download_align_images_from_rcp = None
    download_align_images_from_msr = None
    _OFFICE_FNS_AVAILABLE = False


class RcpMsrDownloader:
    """recipe 의 등록 align key(rcp) + 측정 궤적(msr)을 align_images 트리로 받는다."""

    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir) -> int:
        """eqp_id + recipe_id('<class>/<recipe>') 의 rcp/msr 이미지를 받고 총개수를 반환.

        Case 1(내부 경로 계산): download_align_images_from_rcp/msr 가 align_images 트리에
        직접 적재하므로 dest_dir 는 검증용으로만 쓴다(쓰는 곳==읽는 곳 점검).

        전제: 두 함수는 idempotent(이미 있으면 FTP skip, 기존 경로 반환) — 모듈 docstring
        의 [중요] 참고. send_cube_align_fail_info 도 같은 함수를 embed 용으로 부르므로,
        가드가 없으면 알람당 같은 이미지를 두 번 받는다.
        """
        if not _OFFICE_FNS_AVAILABLE:
            raise RuntimeError(
                "office_rich_notify 의 download_align_images_from_rcp/msr 를 찾을 수 없습니다 "
                "(개발 PC). 오피스 PC 에서만 동작합니다."
            )

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Case 1 — 함수가 (eqp_id, recipe_id) 로 내부에서 경로를 계산해 적재.
        # (함수가 class/recipe 를 따로 받으면: class_name, recipe_name = recipe_id.split("/", 1))
        rcp_imgs, rcp_conds = download_align_images_from_rcp(eqp_id, recipe_id)
        # msr 은 tool 저장 지연 대비 0장이면 잠깐 뒤 재시도(아래 헬퍼 참고).
        msr_imgs, msr_conds = self._download_msr_with_retry(eqp_id, recipe_id)

        # '쓰는 곳 != 읽는 곳' 조기 발견 — 이 다운로더가 막으려는 바로 그 버그 클래스.
        self._warn_if_outside(list(rcp_imgs) + list(msr_imgs), dest_dir)

        n_images = len(rcp_imgs) + len(msr_imgs)
        print(f"[INFO] rcp/msr 다운로드: rcp={len(rcp_imgs)}장 "
              f"msr={len(msr_imgs)}장 (cond rcp={len(rcp_conds)}/msr={len(msr_conds)}) -> {dest_dir}")
        return n_images

    @staticmethod
    def _download_msr_with_retry(eqp_id, recipe_id):
        """msr 측정 이미지를 받되, 0장이면 tool 저장 지연을 감안해 잠깐 뒤 재시도한다.

        align fail 직후엔 측정(E) 이미지가 tool 내부에서 디스크에 늦게 써질 수 있어,
        첫 호출이 0장이면 MSR_RETRY_DELAY_SEC 쉬고 최대 MSR_RETRY_COUNT 회 더 받아본다.
        idempotent 전제라 재호출은 안전(이미 있으면 그 경로 반환). (img_paths, cond_paths) 반환.

        주의: 0장이 '아직 안 써짐'이 아니라 '진짜 측정 없음'일 수도 있다 — 그 경우 재시도는
        헛대기(최대 MSR_RETRY_COUNT*MSR_RETRY_DELAY_SEC 초)지만, 빈 트리로 feasibility 를
        오판하는 비용이 더 크므로 1회 재시도를 기본값으로 둔다. 끄려면 env 횟수 0.
        """
        msr_imgs, msr_conds = download_align_images_from_msr(eqp_id, recipe_id)
        attempt = 0
        while len(msr_imgs) == 0 and attempt < MSR_RETRY_COUNT:
            attempt += 1
            print(f"[INFO] msr 이미지 0장 - tool 저장 지연 추정, {MSR_RETRY_DELAY_SEC}s 후 재시도 "
                  f"({attempt}/{MSR_RETRY_COUNT}) EQP_ID={eqp_id} recipe={recipe_id}")
            time.sleep(MSR_RETRY_DELAY_SEC)
            msr_imgs, msr_conds = download_align_images_from_msr(eqp_id, recipe_id)
        return msr_imgs, msr_conds

    @staticmethod
    def _warn_if_outside(paths, dest_dir):
        """산출물이 dest_dir 밖에 떨어지면 경고. align.assets 가 읽는 경로와 불일치하면
        다운로드는 '성공'해도 cycle 이 빈 트리를 본다 — 이 경고가 그걸 즉시 드러낸다."""
        dest_dir = Path(dest_dir).resolve()
        for p in paths:
            try:
                Path(p).resolve().relative_to(dest_dir)
            except ValueError:
                print(f"[WARNING] 다운로드 경로가 dest_dir 밖: {p} "
                      f"(align.assets 읽기 경로 {dest_dir} 와 불일치 → cycle 이 못 읽음)")
                break


def make_rcp_msr_downloader():
    """glue(rcp_msr_gather)가 호출하는 인자 없는 팩토리."""
    return RcpMsrDownloader()
