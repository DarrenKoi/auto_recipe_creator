"""CV align-key 엔진 — 매칭/ensemble/자산 해석/보정/라이브 탐색.

설계 규칙(2026-05-25 확정): OpenCV 가 정량 점수와 최종 좌표를 결정하고, VLM 은
영역 식별·모호한 FOV 설명·feasibility 평가만 한다. VLM 답변이 낮은 CV 점수를
뒤집거나 반복 가능한 stage 전환을 결정하게 하지 않는다.
"""

# 오피스 MES 가 align fail 시 생성하는 이미지 루트 (우리 코드가 읽기만 하는 입력).
# 실제 레이아웃:
#   align_images/<eqp_id>/<class_name>/<recipe_name>/
#     ├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe 등록 align key
#     └─ align_img_from_msr/   S*/E*                             # 측정 궤적 (E 접두 = fail step)
# 경로 해석은 `align_fail_assets.py` 가 단일 창구로 담당한다 (최신 자동 + override).
from poc.workflow_3 import ALIGN_IMAGES_DIR as ALIGN_IMAGES_ROOT

# from_rcp / from_msr 서브폴더명.
FROM_RCP_DIRNAME = "align_img_from_rcp"
FROM_MSR_DIRNAME = "align_img_from_msr"

# from_rcp 안의 표준 stem (IMAP0001=OM, IMAP0002=SEM). 확장자는 자동 탐색.
RCP_OM_STEM = "IMAP0001"
RCP_SEM_STEM = "IMAP0002"

__all__ = [
    "ALIGN_IMAGES_ROOT",
    "FROM_MSR_DIRNAME",
    "FROM_RCP_DIRNAME",
    "RCP_OM_STEM",
    "RCP_SEM_STEM",
]
