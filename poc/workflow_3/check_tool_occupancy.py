"""RCS List의 Remote / Control User 판독. 클릭 없이 단독 점검 가능.

uv run python -m poc.workflow_3.check_tool_occupancy
아래 ACTION_TARGET_TOOL_NAME 수정, 선택: TOOL_OCCUPANCY_IMAGE=/path/list.jpg
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json, save_debug_text
from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER, UNKNOWN
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_SERVICE
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

# 단독 점검 대상: 이 값을 직접 수정한다. 알람 루프는 전달받은 EQP_ID를 사용한다.
ACTION_TARGET_TOOL_NAME = "MCDA23"


def classify_reading(reading: dict, tool_name: str) -> str:
    """같은 행임이 확인된 명시적 판독만 사용한다. 누락/null은 공백이 아니다."""
    if (reading.get("row_confirmed") is not True
            or not isinstance(reading.get("mc_id"), str)
            or reading["mc_id"].strip().upper() != tool_name.strip().upper()):
        return UNKNOWN
    remote = reading.get("remote_text")
    user = reading.get("control_user_text")
    if remote is not None and not isinstance(remote, str):
        return UNKNOWN
    if user is not None and not isinstance(user, str):
        return UNKNOWN
    if isinstance(remote, str) and re.fullmatch(r"[0-9]+", remote.strip()):
        if int(remote.strip()) > 0:
            return OCCUPIED_BY_OTHER
    if isinstance(user, str) and user.strip():
        return OCCUPIED_BY_OTHER
    if remote == "" and user == "":
        return FREE
    return UNKNOWN


def check_tool_occupancy(image, tool_name: str, *, client=None) -> dict:
    """전체 List와 헤더를 함께 판독한다. 인접 행/잘린 컬럼/실패는 unknown."""
    report = {"target_tool_name": tool_name, "occupancy": UNKNOWN}
    artifact_dir = DEBUG_IMAGE_DIR / "tool_occupancy" / str(time.time_ns())
    report["artifact_dir"] = str(artifact_dir)
    try:
        if image is None or not tool_name.strip():
            raise ValueError("List image and target MC ID are required")
        save_debug_jpeg(image, artifact_dir / "list.jpg")
        image_b64, _, _ = encode_image_webp(image.convert("RGB"), quality=90)
        client = client or Workflow1VLMClient(
            service_slug=os.getenv("TOOL_OCCUPANCY_SERVICE", DEFAULT_SCREEN_ANALYSIS_SERVICE),
        )
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            image_mime="image/webp",
            system_message="Read RCS table cells precisely. Return only a JSON object. Never guess blank cells.",
            user_text=(
                f"Find the exact MC ID {json.dumps(tool_name)} in the RCS List table. "
                "Read ONLY that equipment's row, using the headers and row boundaries. "
                "Read the count immediately next to Remote (such as 1 or 2), and the "
                "Control User cell in that SAME row. Read both independently; when the "
                "Remote count is hard to read, Control User is the second occupancy signal. "
                "Do not copy a count or user from another row. Do not return the headers "
                "Remote or Control User as cell values. Return this schema: "
                '{"mc_id":"exact visible ID", "row_confirmed":true, '
                '"remote_text":"1", "control_user_text":"visible user"}. '
                "Use empty string ONLY for a fully visible, confidently empty cell. "
                "Use null for an unreadable, clipped, hidden or unidentified field. "
                "Set row_confirmed=false if the exact target row cannot be identified "
                "unambiguously. Never infer empty from missing OCR text."
            ),
            temperature=0.0,
        )
        save_debug_text(artifact_dir / "response.txt", response.text)
        reading = extract_json(response.text)
        report["reading"] = reading
        report["occupancy"] = classify_reading(reading, tool_name)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[WARNING] List 점유 판독 실패: {exc}")
    save_debug_json(artifact_dir / "result.json", report)
    print(f"[INFO] List 점유: {json.dumps(report, ensure_ascii=False)}")
    return report


def main() -> int:
    """현재 보이는 List 또는 저장 이미지 판독만 한다. tool을 열지 않는다."""
    load_dotenv()
    from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
    from poc.workflow_3.util import capture_window

    tool_name = ACTION_TARGET_TOOL_NAME.strip()
    try:
        image_path = os.getenv("TOOL_OCCUPANCY_IMAGE", "").strip()
        if image_path:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        else:
            window, _, _ = wait_for_rcs_main_window(timeout_sec=15.0)
            if window is None or not callable(capture_window):
                raise RuntimeError("RCS List 창을 열고 대상 MC ID 행이 보이게 해주세요")
            image = capture_window(window)
        report = check_tool_occupancy(image, tool_name)
    except Exception as exc:
        print(f"[ERROR] 점유 점검 불가: {exc}")
        return 2
    return {FREE: 0, OCCUPIED_BY_OTHER: 1, UNKNOWN: 2}[report["occupancy"]]


if __name__ == "__main__":
    raise SystemExit(main())
