"""RCS List의 Connection User(최우측 컬럼) 판독으로 점유를 판정한다. 클릭 없이 단독 점검 가능.

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
from poc.workflow_3.rcs.workflow_select_tool import _locate_tool_via_vlm
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_SERVICE
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient
from poc.workflow_3.vlm.label_verify import read_text_near_point

# 단독 점검 대상: 이 값을 직접 수정한다. 알람 루프는 전달받은 EQP_ID를 사용한다.
ACTION_TARGET_TOOL_NAME = "MCDA23"
# 한 행만 포함하도록 제한한다. 오피스 DPI/행 높이에 맞춰 조정한다.
MAX_ROW_HEIGHT_PX = 48
CELL_UPSCALE = 3
# MC ID 글자 중심 위/아래 픽셀. 인접 행이 섞이지 않도록 실제 행 간격보다 작게 설정.
ROW_HALF_HEIGHT_PX = 8


def classify_reading(reading: dict, tool_name: str) -> str:
    """같은 행임이 확인된 명시적 판독만 사용한다. 누락/null은 공백이 아니다."""
    if (reading.get("row_confirmed") is not True
            or not isinstance(reading.get("mc_id"), str)
            or reading["mc_id"].strip().upper() != tool_name.strip().upper()):
        return UNKNOWN
    user = reading.get("connection_user_text")
    if not isinstance(user, str):
        return UNKNOWN
    return OCCUPIED_BY_OTHER if user.strip() else FREE


COLUMN_KEYS = {"mcid": "mc_id", "connectionuser": "connection_user"}


def columns_from_headers(headers, image_width: int) -> dict:
    """헤더 중심 x(0-1000) 목록을 인접 헤더의 중점으로 갈라 컬럼 경계를 만든다.

    mai-ui 는 점(grounding)은 잘 찍지만 폭은 어림한다 - 오피스 실측에서 MC ID 는 2배,
    Connection User 는 왼쪽이 잘렸다. 폭을 모델에 묻지 않고 이웃 헤더
    간격에서 파생하면 어느 컬럼도 서로 겹치거나 비지 않는다. 첫/마지막 컬럼은 이미지 가장자리까지.
    """
    if not isinstance(headers, list) or not headers:
        raise ValueError("missing headers")
    centers = []
    for header in headers:
        if (not isinstance(header, dict) or not isinstance(header.get("name"), str)
                or type(header.get("x")) not in (int, float)):
            raise ValueError(f"invalid header entry: {header!r}")
        name = re.sub(r"[^a-z]", "", header["name"].lower())
        centers.append((int(round(header["x"] * image_width / 1000)), name))
    centers.sort()
    columns = {}
    for index, (x, name) in enumerate(centers):
        left = 0 if index == 0 else (centers[index - 1][0] + x) // 2
        right = image_width if index == len(centers) - 1 else (x + centers[index + 1][0]) // 2
        if name in COLUMN_KEYS:
            columns[COLUMN_KEYS[name]] = [left, right]
    return columns


def validate_columns(columns, image_width: int):
    """VLM 컬럼 좌표의 타입, 이미지 경계, 겹침을 검증한다."""
    if not isinstance(columns, dict):
        raise ValueError("missing column bounds")
    spans = []
    for name in ("mc_id", "connection_user"):
        span = columns.get(name)
        if (not isinstance(span, list) or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= image_width):
            raise ValueError(f"invalid {name} column bounds")
        left, right = span
        if any(left < end and right > start for start, end in spans):
            raise ValueError("overlapping columns")
        spans.append(span)


def build_row_read_image(image, layout: dict, tool_name: str):
    """두 컬럼을 동일 y 범위로 잘라 확대한다. 좌우 경계는 헤더 중점 분할이다."""
    if layout.get("mc_id") != tool_name:
        raise ValueError("layout MC ID mismatch")
    top, bottom = layout.get("row_top"), layout.get("row_bottom")
    if (type(top) is not int or type(bottom) is not int
            or not 0 <= top < bottom <= image.height
            or bottom - top > MAX_ROW_HEIGHT_PX):
        raise ValueError("invalid or multi-row crop height")
    columns = layout.get("columns")
    validate_columns(columns, image.width)
    cells = []
    for name in ("mc_id", "connection_user"):
        left, right = columns[name]
        cell = image.crop((left, top, right, bottom)).convert("RGB")
        cells.append(cell.resize((cell.width * CELL_UPSCALE, cell.height * CELL_UPSCALE)))
    panel_height = cells[0].height + 28
    fine = Image.new("RGB", (max(cell.width for cell in cells) + 16, panel_height * len(cells)), "white")
    draw = ImageDraw.Draw(fine)
    for index, (label, cell) in enumerate(zip(("MC ID", "Connection User"), cells)):
        y = index * panel_height
        draw.text((8, y + 4), label, fill="black")
        fine.paste(cell, (8, y + 24))
    return fine


def locate_row_point(image, tool_name: str, artifact_dir):
    """클릭 경로와 같은 로케이터를 쓴다: list 영역 전체 + tool-list 결과 모드 + 행 확인 게이트.

    MC ID 컬럼만 잘라 넣으면 좁고 긴 strip 안의 행들이 전부 같아 보여 coarse bbox 가
    무의미해지고 확인 게이트도 없어 mai-ui 가 이웃 행을 고른다(오피스 실측).
    """
    located, attempts = _locate_tool_via_vlm(
        None, "RCS List", "image", tool_name, image,
        debug_image_dir=artifact_dir / "locator", log_name="tool_occupancy",
        component_name="tool_occupancy", timestamp_tag="mc_id",
    )
    if located is None:
        raise ValueError(f"MC ID locator failed: {json.dumps(attempts.get('iters'), ensure_ascii=False)}")
    return located["full_image_point"]


def check_tool_occupancy(image, tool_name: str, *, client=None, row_point=None, ocr_client=None) -> dict:
    """헤더 → MC ID 위치 → PaddleOCR 엄격 검증 → 같은 행 Connection User 판독."""
    report = {"target_tool_name": tool_name, "occupancy": UNKNOWN,
              "diagnosis": "column_location_failed"}
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
                "This is an RCS equipment list. Find the table HEADER row and return the "
                "horizontal center x of EVERY column header text in it, on a 0-1000 scale "
                "(0 = left image edge, 1000 = right image edge). Include all headers, e.g. "
                "MC ID, RCS IP, Location, Model, Status, Count, DVR, Remote, Control User, "
                "Connection User, and any others you see. Control User and Connection User "
                "are different columns; Connection User is the LAST column at the far right "
                "edge of the table and must be included. "
                "Use the header text exactly as displayed. Do not select any equipment row. "
                'Schema: {"headers":[{"name":"MC ID","x":40},{"name":"Remote","x":520}]}. '
                "The numbers are examples only."
            ),
            temperature=0.0,
        )
        save_debug_text(artifact_dir / "coarse_response.txt", coarse.text)
        column_reading = extract_json(coarse.text)
        report["column_reading"] = column_reading
        columns = columns_from_headers(column_reading.get("headers"), image.width)
        report["columns_px"] = columns
        validate_columns(columns, image.width)
        report["diagnosis"] = "row_location_failed"
        point = row_point if row_point is not None else locate_row_point(image, tool_name, artifact_dir)
        report["row_point"] = point
        report["row_source"] = "click_locator" if row_point is not None else "mc_id_column_locator"
        left, right = columns["mc_id"]
        if (not isinstance(point, dict)
                or any(type(point.get(axis)) is not int for axis in ("x", "y"))
                or not left <= point["x"] < right
                or not ROW_HALF_HEIGHT_PX <= point["y"] < image.height - ROW_HALF_HEIGHT_PX):
            raise ValueError("refined point is outside MC ID column or image")
        layout = {"mc_id": tool_name, "columns": columns,
                  "row_top": point["y"] - ROW_HALF_HEIGHT_PX,
                  "row_bottom": point["y"] + ROW_HALF_HEIGHT_PX}
        report["layout"] = layout
        fine_image = build_row_read_image(image, layout, tool_name)
        save_debug_jpeg(image.crop((0, layout["row_top"], image.width, layout["row_bottom"])),
                        artifact_dir / "row.jpg")
        save_debug_jpeg(fine_image, artifact_dir / "cells.jpg")
        report["diagnosis"] = "mc_id_ocr_failed"
        id_read = read_text_near_point(
            image, {"left": left, "right": right,
                    "top": layout["row_top"], "bottom": layout["row_bottom"]},
            debug_image_dir=artifact_dir, timestamp_tag="id", artifact_label="mc_id",
            log_name="tool_occupancy", client=ocr_client,
        )
        report["mc_id_ocr"] = {"raw_text": id_read.raw_text, "ok": id_read.ok,
                               "error": id_read.error, "crop": id_read.crop_image_path}
        if not id_read.ok:
            raise ValueError(f"PaddleOCR MC ID read failed: {id_read.error}")
        # D/0 등을 보정하지 않는다. 다른 ID나 인접 행 혼입도 엄격히 거부한다.
        tokens = id_read.raw_text.upper().split()
        if tokens != [tool_name.upper()]:
            report["diagnosis"] = "mc_id_mismatch" if tokens else "mc_id_unreadable"
            raise ValueError(f"expected {tool_name!r}, PaddleOCR read {id_read.raw_text!r}")
        report["diagnosis"] = "occupancy_read_failed"
        fine_b64, _, _ = encode_image_webp(fine_image, quality=90)
        response = client.chat_with_image_b64(
            image_b64=fine_b64,
            image_mime="image/webp",
            system_message="Transcribe two cropped RCS cells. Return only JSON. Never guess blank cells.",
            user_text=(
                "The image contains two labelled panels: MC ID, Connection User. "
                "Each panel contains an enlarged cell cropped from the SAME pixel row band. "
                "Read only cell content below each label; labels are not cell values. "
                "Transcribe ALL equipment IDs visible in the MC ID panel into visible_mc_ids. "
                "If two rows or partial neighboring text are visible, row_confirmed=false. "
                "Transcribe the Connection User text exactly as shown. "
                "Use empty string ONLY for a completely visible, confidently empty cell. "
                "Use null for clipped, partial or unreadable content. Never infer empty "
                "from failed recognition. Schema: "
                '{"mc_id":"transcribed ID","visible_mc_ids":["transcribed ID"],'
                '"row_confirmed":true,"connection_user_text":"visible user"}. '
                "Transcribe the actual MC ID, never substitute an expected ID."
            ),
            temperature=0.0,
        )
        save_debug_text(artifact_dir / "response.txt", response.text)
        reading = extract_json(response.text)
        report["reading"] = reading
        if reading.get("visible_mc_ids") == [tool_name]:
            report["occupancy"] = classify_reading(reading, tool_name)
            report["diagnosis"] = "ok" if report["occupancy"] != UNKNOWN else "occupancy_unreadable"
        else:
            report["diagnosis"] = "fine_mc_id_mismatch"
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
