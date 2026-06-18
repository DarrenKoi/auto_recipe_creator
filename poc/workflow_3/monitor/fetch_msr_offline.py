"""오프라인 벤치(golden set 확장)용 rcp+msr 동기 다운로드 스크립트.

프로덕션 루프는 align_img_from_msr 을 받지 않는다(런타임 미소비). 하지만 workflow_2/_3
오프라인 벤치(golden localization/consensus eval)는 측정 궤적(S*/E*)을 정답 근거로 쓰므로,
golden set 을 새 recipe 로 넓힐 때 이 스크립트로 그 recipe 의 rcp+msr 을 받아둔다.

설정은 env 로만 받는다(CLAUDE.md: argparse 금지):
  - ALIGN_EQP_ID       : 장비 ID (필수)
  - ALIGN_RECIPE_NAME  : '<class>/<recipe>' 형태의 recipe_id (필수)
office 다운로더(office_rcp_msr_downloader)가 있어야 실제로 받는다 - 개발 PC 에서는
게이트에서 막혀 [WARNING] 후 종료한다(루프와 동일 best-effort 철학).

    uv run python poc/workflow_3/monitor/fetch_msr_offline.py
"""

import os

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.rcp_msr_gather import gather_rcp_msr


def main():
    eqp_id = os.environ.get("ALIGN_EQP_ID", "").strip()
    recipe_id = os.environ.get("ALIGN_RECIPE_NAME", "").strip()
    if not eqp_id or not recipe_id:
        print("[ERROR] ALIGN_EQP_ID 와 ALIGN_RECIPE_NAME('<class>/<recipe>') 를 env 로 지정하세요.")
        return 1

    settings = load_workflow3_settings()
    print(f"[INFO] 오프라인 msr fetch: EQP_ID={eqp_id} recipe={recipe_id} (include_msr=True)")
    ok = gather_rcp_msr(eqp_id, recipe_id, settings, include_msr=True)
    if ok:
        print("[INFO] rcp+msr 다운로드 완료.")
        return 0
    print("[WARNING] 다운로드 안 됨 (downloader 부재/게이트 off/예외). 콘솔 로그 확인.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
