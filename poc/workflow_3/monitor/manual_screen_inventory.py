"""열린 tool 창의 현재 화면을 **레지스트리 기준으로** 읽어 표로 출력한다. 클릭 없음.

  - 등록 버튼마다 그 자리(버튼 한 개 크기)를 PaddleOCR 로 읽어 label_seen / not_seen /
    read_error / no_region 을 낸다(`button_registry.inventory`).
  - 앞에 떠 있는 창 하나의 제목줄을 VLM 으로 찾아 OCR 로 읽는다. **한 번만** 본다 -
    못 찾았다고 열린 창이 없다는 뜻은 아니다(unknown).

이 표는 관찰용이다. 클릭 승인에는 쓰지 않는다 - 클릭은 `manual_click_button` 이 매번
현재 프레임에서 VLM 좌표 + OCR 확인을 새로 한다. not_seen 에는 가림·이동·커서 겹침·OCR
누락이 섞여 있어 그것만으로 '가려졌다' 고 판정하지 않는다.

실행: uv run python poc/workflow_3/monitor/manual_screen_inventory.py
종료 코드: 0=표 출력, 2=tool 창 없음
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.config import load_workflow3_settings  # noqa: E402
from poc.workflow_3.monitor.button_registry import BUTTONS, inventory  # noqa: E402
from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window  # noqa: E402
from poc.workflow_3.util import make_timestamp_tag  # noqa: E402
from poc.workflow_3.util.event_dir import debug_root  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 셸 env(괄호 안 이름)가 있으면 env 가 이긴다.
# ===========================================================================

EQP_ID = "MCD513"                  # (MANUAL_INVENTORY_EQP_ID) 제목에 이 ID 가 든 tool 창
FRONT_WINDOW_DESCRIPTION = (
    "the TITLE BAR of the front-most separate window or dialog floating on top of the "
    "Remote Monitoring screen (for example 'SECS Terminal' or 'File Manager'). Point at "
    "the middle of that title bar. Do not point at buttons of the main screen."
)

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2


def main() -> int:
    os.environ.setdefault("SAFE_MODE", "1")  # 클릭하지 않는 진입점 - 이동/클릭 협력자도 안 쓴다
    load_workflow3_settings()
    eqp_id = os.environ.get("MANUAL_INVENTORY_EQP_ID", "").strip() or EQP_ID

    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] tool 창이 없습니다: EQP_ID={eqp_id}. 먼저 직접 접속하세요.")
        return EXIT_PREFLIGHT_FAILED
    print(f"[INFO] 열린 tool 창: title={title!r}")

    from poc.workflow_3.util.image_utils import capture_window
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target

    debug_dir = debug_root() / "manual_screen_inventory" / make_timestamp_tag()
    image = capture_window(window)
    debug_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(debug_dir / "screen.jpg", quality=90)

    def _read(img, box, key):
        read = read_text_near_point(
            img, box, debug_image_dir=debug_dir, timestamp_tag=make_timestamp_tag(),
            artifact_label=key, log_name="manual_screen_inventory",
        )
        if not read.ok:
            raise RuntimeError(read.error)
        return tokens_from_text(read.raw_text)

    rows = inventory(image, BUTTONS, read_fn=_read)
    print(f"[INFO] 등록 버튼 {len(rows)}개 (image {image.width}x{image.height})")
    for row in rows:
        detail = row.get("error") or " ".join(row["tokens"][:8])
        print(f"  {row['status']:<11} {row['key']:<14} '{row['label']}' [{row['window']}] {detail}")

    front = analyze_window_target(
        None, "Remote Monitoring System", "uia",
        TargetConfig(key="front_window_title", description=FRONT_WINDOW_DESCRIPTION),
        debug_image_dir=debug_dir, log_name="manual_screen_inventory",
        component_name="manual_screen_inventory", artifact_prefix="front_window_title",
        image=image,
    ).point
    if front is None:
        front_text = "unknown(미검출 - 열린 창이 없다는 증거 아님)"
    else:
        box = crop_box_around_point(front, image.width, image.height,
                                    left_ratio=0.15, right_ratio=0.15, half_height_ratio=0.015)
        try:
            front_text = f"px={front} 제목={' '.join(_read(image, box, 'front_window_title'))!r}"
        except RuntimeError as exc:
            front_text = f"px={front} 제목 판독 실패: {exc}"
    print(f"  front_window  {front_text}")

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[DIGEST] screen_inventory eqp={eqp_id} "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" front={'found' if front else 'unknown'} debug={debug_dir}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
