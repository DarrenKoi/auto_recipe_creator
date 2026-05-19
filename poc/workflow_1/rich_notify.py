"""Align Fail 감지 시 API Cube DM 으로 보내는 rich notification 모듈.

송신 페이로드 구성:
  - 텍스트: EQP_ID, RECIPE_ID, ALID, UTC9, OPERATION_DESC, LOT_TYPE_CD, history
  - 첨부 이미지 2장:
      1) recipe 에 저장된 Align 이미지 (툴 FTP 에서 다운로드)
      2) 알람 시점의 툴 RCS 창 캡처 (Remote Monitoring System 창)

환경변수 (미설정 시 해당 단계는 skip — 텍스트만 전송):
  APICUBE_BASE_URL          예: https://apicube.internal
  APICUBE_BOT_TOKEN         Bearer 토큰
  APICUBE_DM_USER_IDS       콤마 구분 user id 목록
  APICUBE_MESSAGE_PATH      기본 "/api/v1/messages" (사내 스펙에 맞춰 조정)

  RCS_FTP_HOST              FTP 호스트 (예: 사내 파일 서버, 혹은 "{eqp_id}" 템플릿)
  RCS_FTP_PORT              기본 21
  RCS_FTP_USER, RCS_FTP_PASS
  RCS_FTP_RECIPE_IMAGE_PATH FTP 경로 템플릿 (예: "/recipes/{recipe_id}/align.bmp")
  RCS_FTP_TIMEOUT_SEC       기본 10

  ALIGN_FAIL_HISTORY_LIMIT  history 줄 수 (기본 5)
"""

import ftplib
import os
import socket
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from poc.workflow_1 import DEBUG_IMAGE_DIR, LOG_DIR
from poc.workflow_1.util import env_int, make_timestamp_tag


RECIPE_IMAGE_CACHE_DIR = DEBUG_IMAGE_DIR / "rich_notify" / "recipe_image"
TOOL_CAPTURE_DIR = DEBUG_IMAGE_DIR / "rich_notify" / "tool_capture"
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"

APICUBE_TIMEOUT_SEC = 15


def _safe_name(text: str) -> str:
    """파일명에 안전한 문자만 남긴다."""
    safe = []
    for ch in (text or "").strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe).strip("-._") or "x"


# ---------------------------------------------------------------- FTP 다운로드


def fetch_recipe_align_image(eqp_id: str, recipe_id: str) -> Path | None:
    """레시피에 저장된 Align 이미지를 툴 FTP 에서 받아 로컬 파일로 저장한다.

    설정이 없거나 실패하면 None 을 반환하고 caller 는 텍스트만 전송한다.
    """
    host_template = os.getenv("RCS_FTP_HOST", "").strip()
    user = os.getenv("RCS_FTP_USER", "").strip()
    password = os.getenv("RCS_FTP_PASS", "")
    path_template = os.getenv("RCS_FTP_RECIPE_IMAGE_PATH", "").strip()
    if not host_template or not path_template:
        print("[INFO] FTP 미설정 — recipe align 이미지 skip")
        return None

    port = env_int("RCS_FTP_PORT", 21)
    timeout = env_int("RCS_FTP_TIMEOUT_SEC", 10)

    host = host_template.format(eqp_id=eqp_id, recipe_id=recipe_id)
    remote_path = path_template.format(eqp_id=eqp_id, recipe_id=recipe_id)
    suffix = Path(remote_path).suffix or ".bmp"
    RECIPE_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = (
        RECIPE_IMAGE_CACHE_DIR
        / f"{make_timestamp_tag()}_{_safe_name(eqp_id)}_{_safe_name(recipe_id)}{suffix}"
    )

    print(f"[INFO] FTP 다운로드 시도: host={host}, path={remote_path}")
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(host=host, port=port, timeout=timeout)
            ftp.login(user=user or "anonymous", passwd=password or "")
            with local_path.open("wb") as fp:
                ftp.retrbinary(f"RETR {remote_path}", fp.write)
    except (ftplib.all_errors + (socket.timeout, OSError)) as exc:
        print(f"[WARNING] FTP 다운로드 실패: {exc}")
        try:
            local_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    # BMP/TIFF 같은 큰 포맷이면 JPEG 으로 변환하여 첨부 부피 축소.
    return _convert_to_jpeg_if_large(local_path)


def _convert_to_jpeg_if_large(path: Path, threshold_bytes: int = 1_000_000) -> Path:
    """원본 그대로 전송하면 큰 BMP/PNG 는 JPEG 으로 변환해 반환한다."""
    try:
        if path.stat().st_size < threshold_bytes:
            return path
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            jpeg_path = path.with_suffix(".jpg")
            rgb.save(jpeg_path, format="JPEG", quality=90)
        if jpeg_path != path:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return jpeg_path
    except (OSError, ValueError) as exc:
        print(f"[WARNING] JPEG 변환 실패 (원본 유지): {exc}")
        return path


# ----------------------------------------------------------- 툴 RCS 창 캡처


def capture_tool_align_image(eqp_id: str) -> Path | None:
    """알람 시점의 Remote Monitoring System 창을 캡처해 JPEG 로 저장한다."""
    if os.name != "nt":
        print(f"[INFO] 비 Windows 환경 — 툴 캡처 skip: os={os.name}")
        return None

    try:
        from poc.workflow_1.login_rcs_common import wait_for_remote_monitoring_window
        from poc.workflow_1.util import capture_window
    except ImportError as exc:
        print(f"[WARNING] 창 캡처 의존성 누락: {exc}")
        return None

    print(f"[INFO] 툴 창 탐색: eqp_id={eqp_id}")
    tool_window, window_title, backend = wait_for_remote_monitoring_window(
        eqp_id,
        timeout_sec=3.0,
        poll_interval_sec=0.3,
    )
    if tool_window is None:
        print(f"[WARNING] Remote Monitoring System 창 미발견: eqp_id={eqp_id}")
        return None

    try:
        image = capture_window(tool_window)
    except Exception as exc:
        print(f"[WARNING] 창 캡처 실패: title={window_title!r}, backend={backend}, error={exc}")
        return None

    TOOL_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TOOL_CAPTURE_DIR / f"{make_timestamp_tag()}_{_safe_name(eqp_id)}.jpg"
    try:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        rgb.save(out_path, format="JPEG", quality=90)
    except Exception as exc:
        print(f"[WARNING] 캡처 이미지 저장 실패: {exc}")
        return None

    print(f"[INFO] 툴 창 캡처 저장: {out_path}")
    return out_path


# ----------------------------------------------------------------- 이력 조회


def collect_recipe_history(recipe_id: str, limit: int | None = None) -> list[str]:
    """`align_fail_alarms.txt` 에서 같은 RECIPE_ID 의 최근 알람 라인을 반환한다.

    deque(maxlen) 로 last-N 만 메모리에 유지하여 로그가 커져도 안전.
    """
    if not recipe_id:
        return []

    if limit is None:
        limit = env_int("ALIGN_FAIL_HISTORY_LIMIT", 5)
    if limit <= 0:
        return []

    needle = f"RECIPE_ID={recipe_id}"
    matched: deque[str] = deque(maxlen=limit)
    try:
        with ALARM_LOG_PATH.open("r", encoding="utf-8") as fp:
            for line in fp:
                if needle in line:
                    matched.append(line.rstrip("\n"))
    except FileNotFoundError:
        return []
    except OSError as exc:
        print(f"[WARNING] 알람 로그 읽기 실패: {exc}")
        return []

    return list(matched)


# --------------------------------------------------------------- API Cube 송신


def _build_message_text(
    eqp_id: str,
    recipe_id: str,
    alarm_time: str,
    alarm_name: str,
    alid: str,
    operation_desc: str,
    lot_type_cd: str,
    history: list[str],
) -> str:
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "🚨 CD-SEM Align Fail",
        f"감지시각 : {detected_at}",
        f"EQP_ID   : {eqp_id}",
        f"RECIPE_ID: {recipe_id}",
        f"ALARM    : {alarm_name} (ALID={alid})",
        f"UTC9     : {alarm_time}",
        f"OPERATION: {operation_desc}",
        f"LOT_TYPE : {lot_type_cd}",
    ]
    if history:
        lines.append("")
        lines.append(f"이 RECIPE_ID 의 최근 알람 {len(history)}건:")
        lines.extend(f"  • {line}" for line in history)
    return "\n".join(lines)


def _read_image_attachments(
    image_paths: list[Path],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """첨부할 이미지를 한 번만 디스크에서 읽어 multipart files 리스트로 만든다."""
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for img_path in image_paths:
        try:
            with img_path.open("rb") as fp:
                data = fp.read()
        except OSError as exc:
            print(f"[WARNING] 첨부 파일 읽기 실패: {img_path}, error={exc}")
            continue
        files.append(("files", (img_path.name, data, "image/jpeg")))
    return files


def _send_apicube_dm(
    user_id: str,
    text: str,
    files: list[tuple[str, tuple[str, bytes, str]]],
    *,
    url: str,
    bot_token: str,
) -> bool:
    """API Cube 에 DM 1건 전송. 사내 REST 스펙은 환경변수 + 아래 페이로드 키로 조정한다."""
    headers = {"Authorization": f"Bearer {bot_token}"}
    data = {"recipient": user_id, "text": text}

    print(f"[INFO] API Cube DM 전송: url={url}, user={user_id}, files={len(files)}")
    try:
        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files or None,
            timeout=APICUBE_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        print(f"[WARNING] API Cube 전송 예외: user={user_id}, error={exc}")
        return False

    if response.status_code >= 400:
        print(
            f"[WARNING] API Cube 응답 오류: user={user_id}, "
            f"status={response.status_code}, body={response.text[:200]!r}"
        )
        return False
    print(f"[INFO] API Cube DM 전송 완료: user={user_id}, status={response.status_code}")
    return True


def _resolve_apicube_endpoint() -> tuple[str, str] | None:
    """환경변수에서 API Cube 호출 URL 과 토큰을 한 번에 해석한다."""
    base_url = os.getenv("APICUBE_BASE_URL", "").strip().rstrip("/")
    bot_token = os.getenv("APICUBE_BOT_TOKEN", "").strip()
    if not base_url or not bot_token:
        return None
    message_path = os.getenv("APICUBE_MESSAGE_PATH", "/api/v1/messages").strip()
    if not message_path.startswith("/"):
        message_path = "/" + message_path
    return f"{base_url}{message_path}", bot_token


# ------------------------------------------------------------------- 오케스트


def send_rich_align_fail_notification(
    eqp_id: str,
    alarm_time: str,
    alarm_name: str,
    alid: str = "9006",
    recipe_id: str = "",
    operation_desc: str = "",
    lot_type_cd: str = "",
) -> None:
    """Align Fail 1건에 대한 rich 알림(이미지 2장 + 이력)을 DM 으로 전송."""
    user_ids_raw = os.getenv("APICUBE_DM_USER_IDS", "").strip()
    user_ids = [uid.strip() for uid in user_ids_raw.split(",") if uid.strip()]
    if not user_ids:
        print("[INFO] APICUBE_DM_USER_IDS 미설정 — rich notify skip")
        return

    endpoint = _resolve_apicube_endpoint()
    if endpoint is None:
        print("[INFO] API Cube 미설정 — 전송 skip")
        return
    url, bot_token = endpoint

    # FTP / 창 캡처 / 로그 grep 은 서로 독립이므로 동시 실행해 wall-clock 을 줄인다.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="rich-notify-fetch") as pool:
        ftp_future = pool.submit(fetch_recipe_align_image, eqp_id, recipe_id) if recipe_id else None
        capture_future = pool.submit(capture_tool_align_image, eqp_id)
        history_future = pool.submit(collect_recipe_history, recipe_id) if recipe_id else None

        recipe_image_path = ftp_future.result() if ftp_future is not None else None
        tool_image_path = capture_future.result()
        history = history_future.result() if history_future is not None else []

    image_paths = [p for p in (recipe_image_path, tool_image_path) if p is not None]
    files = _read_image_attachments(image_paths)
    text = _build_message_text(
        eqp_id=eqp_id,
        recipe_id=recipe_id,
        alarm_time=alarm_time,
        alarm_name=alarm_name,
        alid=alid,
        operation_desc=operation_desc,
        lot_type_cd=lot_type_cd,
        history=history,
    )

    if len(user_ids) == 1:
        _send_apicube_dm(user_ids[0], text, files, url=url, bot_token=bot_token)
        return

    with ThreadPoolExecutor(
        max_workers=min(5, len(user_ids)),
        thread_name_prefix="rich-notify-send",
    ) as pool:
        for uid in user_ids:
            pool.submit(_send_apicube_dm, uid, text, files, url=url, bot_token=bot_token)
