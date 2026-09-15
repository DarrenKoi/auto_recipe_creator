"""RCS List의 Remote / Control User 판독. 클릭 없이 단독 점검 가능.

uv run python -m poc.workflow_3.check_tool_occupancy
아래 ACTION_TARGET_TOOL_NAME 수정, 선택: TOOL_OCCUPANCY_IMAGE=/path/list.jpg
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json, save_debug_text
from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER, UNKNOWN
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_SERVICE
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

# 단독 점검 대상: 이 값을 직접 수정한다. 알람 루프는 전달받은 EQP_ID를 사용한다.
ACTION_TARGET_TOOL_NAME = "MCDA23"
# 한 행만 포함하도록 제한한다. 오피스 DPI/행 높이에 맞춰 조정한다.
MAX_ROW_HEIGHT_PX = 48
CELL_UPSCALE = 3


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


def build_row_read_image(image, layout: dict, tool_name: str):
    """세 컬럼을 동일 y 범위로 잘라 확대한다. 좌우 위치는 헤더로 찾는다."""
    if layout.get("mc_id") != tool_name:
        raise ValueError("coarse MC ID mismatch")
    top, bottom = layout.get("row_top"), layout.get("row_bottom")
    if (type(top) is not int or type(bottom) is not int
            or not 0 <= top < bottom <= image.height
            or bottom - top > MAX_ROW_HEIGHT_PX):
        raise ValueError("invalid or multi-row crop height")
    columns = layout.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("missing column bounds")
    cells = []
    spans = []
    for name in ("mc_id", "remote", "control_user"):
        span = columns.get(name)
        if (not isinstance(span, list) or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= image.width):
            raise ValueError(f"invalid {name} column bounds")
        left, right = span
        if any(left < end and right > start for start, end in spans):
            raise ValueError("overlapping columns")
        spans.append(span)
        cell = image.crop((left, top, right, bottom)).convert("RGB")
        cells.append(cell.resize((cell.width * CELL_UPSCALE, cell.height * CELL_UPSCALE)))
    panel_height = cells[0].height + 28
    fine = Image.new("RGB", (max(cell.width for cell in cells) + 16, panel_height * 3), "white")
    draw = ImageDraw.Draw(fine)
    for index, (label, cell) in enumerate(zip(("MC ID", "Remote", "Control User"), cells)):
        y = index * panel_height
        draw.text((8, y + 4), label, fill="black")
        fine.paste(cell, (8, y + 24))
    return fine


def check_tool_occupancy(image, tool_name: str, *, client=None) -> dict:
    """coarse 행/컬럼 위치 → 동일 행 cell crop → fine 텍스트 판독."""
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
        coarse = client.chat_with_image_b64(
            image_b64=image_b64,
            image_mime="image/webp",
            system_message="Locate RCS table geometry. Return only JSON. Do not classify occupancy.",
            user_text=(
                f"Image size is {image.width} x {image.height} pixels. "
                f"Locate the exact MC ID {json.dumps(tool_name)} using the MC ID header. "
                "Do not assume MC ID is on the left: columns may be far apart. "
                "Locate the Remote count field and Control User column using their labels. "
                "Return one shared row_top/row_bottom pixel band containing ONLY the target "
                "row, excluding text from rows above and below. Return horizontal pixel "
                "bounds for the three cells. Remote bounds must cover the count next to "
                "Remote, not merely the Remote label. Control User bounds must cover the "
                "whole user cell, not just a blank part. Exclude table headers. "
                "All coordinates are absolute image pixels, NOT normalized 0-1000. Schema: "
                '{"mc_id":"exact visible ID","row_top":100,"row_bottom":120,'
                '"columns":{"mc_id":[10,110],"remote":[400,500],"control_user":[700,900]}}. '
                "These are example coordinates only. Use null for anything uncertain. "
                "Do not choose a similarly named neighbor (MCDA01 and MCDA23 are different)."
            ),
            temperature=0.0,
        )
        save_debug_text(artifact_dir / "coarse_response.txt", coarse.text)
        layout = extract_json(coarse.text)
        report["layout"] = layout
        fine_image = build_row_read_image(image, layout, tool_name)
        save_debug_jpeg(image.crop((0, layout["row_top"], image.width, layout["row_bottom"])),
                        artifact_dir / "row.jpg")
        save_debug_jpeg(fine_image, artifact_dir / "cells.jpg")
        fine_b64, _, _ = encode_image_webp(fine_image, quality=90)
        response = client.chat_with_image_b64(
            image_b64=fine_b64,
            image_mime="image/webp",
            system_message="Transcribe three cropped RCS cells. Return only JSON. Never guess blank cells.",
            user_text=(
                "The image contains three labelled panels: MC ID, Remote, Control User. "
                "Each panel contains an enlarged cell cropped from the SAME pixel row band. "
                "Read only cell content below each label; labels are not cell values. "
                "Transcribe ALL equipment IDs visible in the MC ID panel into visible_mc_ids. "
                "If two rows or partial neighboring text are visible, row_confirmed=false. "
                "Read the Remote count and Control User text independently. "
                "Use empty string ONLY for a completely visible, confidently empty cell. "
                "Use null for clipped, partial or unreadable content. Never infer empty "
                "from failed recognition. Schema: "
                '{"mc_id":"transcribed ID","visible_mc_ids":["transcribed ID"],'
                '"row_confirmed":true,"remote_text":"1","control_user_text":"visible user"}. '
                "Transcribe the actual MC ID, never substitute an expected ID."
            ),
            temperature=0.0,
        )
        save_debug_text(artifact_dir / "response.txt", response.text)
        reading = extract_json(response.text)
        report["reading"] = reading
        if reading.get("visible_mc_ids") == [tool_name]:
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
