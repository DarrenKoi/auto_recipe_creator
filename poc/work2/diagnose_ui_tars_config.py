"""UI-TARS 요청 형상/토큰 진단 스크립트.

목적:
- `completion_tokens=1` 현상이 stream 설정, 이미지 포함 여부, system role,
  혹은 generation field 차이 때문인지 분리 진단한다.
- 사내 Windows/office 환경에서 UI-TARS 프록시에 직접 붙어 raw 응답을 저장한다.

기본 사용:
  uv run python poc/work2/diagnose_ui_tars_config.py

선택 환경변수:
- `UI_TARS_DIAG_IMAGE_PATH` : 로컬 이미지 경로. 비우면 더미 로그인 이미지를 생성한다.
- `UI_TARS_DIAG_SEND_REQUESTS` : `0` 이면 요청을 보내지 않고 payload 만 저장한다.
- `UI_TARS_DIAG_VARIANTS` : 실행 variant 목록. 예: `script_nonstream,script_stream`
- `UI_TARS_DIAG_MAX_TOKENS` : 기본 256
- `UI_TARS_DIAG_TIMEOUT_SEC` : 기본 60
- `UI_TARS_DIAG_INCLUDE_TOKEN_ALIAS_VARIANTS` : `1` 이면 `max_new_tokens` 변형도 시도한다.
"""

import json
import os
import time
import base64
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from poc.work2.flask_vlm import (
    get_service_by_slug,
    resolve_service_api_key,
    resolve_service_proxy_url,
)
from poc.work2.logger import log_work2_event
from poc.work2.prompts.login_rcs_ui_tars import (
    build_login_rcs_ui_tars_prompt,
    build_single_element_prompt,
)
from poc.work2.util.debug_image_utils import save_debug_jpeg, save_debug_webp
from poc.work2.vlm_client import OpenAICompatibleVLMClient

load_dotenv()

SERVICE_SLUG = "ui-tars"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_VARIANTS = (
    "text_only_nonstream",
    "text_only_stream",
    "script_nonstream",
    "script_stream",
    "script_stream_with_system",
    "single_element_stream",
)


def env_flag(name: str, default: bool = False) -> bool:
    """bool 환경변수를 읽는다."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    """int 환경변수를 읽는다."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다: {raw!r}. 기본값 {default} 사용")
        return default


def env_float(name: str, default: float) -> float:
    """float 환경변수를 읽는다."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다: {raw!r}. 기본값 {default} 사용")
        return default


def parse_variants() -> tuple[str, ...]:
    """실행할 variant 목록을 반환한다."""
    raw = os.getenv("UI_TARS_DIAG_VARIANTS", "").strip()
    if not raw:
        variants = list(DEFAULT_VARIANTS)
    else:
        variants = [
            item.strip()
            for item in raw.replace(";", ",").split(",")
            if item.strip()
        ]

    if env_flag("UI_TARS_DIAG_INCLUDE_TOKEN_ALIAS_VARIANTS"):
        variants.extend(
            [
                "script_stream_max_new_tokens",
                "script_stream_both_token_keys",
            ]
        )

    seen: set[str] = set()
    deduped: list[str] = []
    for item in variants:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


def make_timestamp_tag(now: float | None = None) -> str:
    """파일명용 타임스탬프를 만든다."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(now or time.time()))


def _write_text(path: Path, text: str) -> None:
    """텍스트 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    """JSON 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _truncate(text: str, limit: int = 240) -> str:
    """로그 출력용 문자열 길이를 제한한다."""
    normalized = (text or "").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _load_font(size: int = 20) -> "ImageFont.ImageFont":
    """가능한 폰트를 선택한다."""
    for name in ("arial.ttf", "Arial.ttf", "AppleGothic.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_dummy_login_image(width: int = 1280, height: int = 900) -> "Image.Image":
    """간단한 RCS 로그인 유사 더미 이미지를 생성한다."""
    image = Image.new("RGB", (width, height), "#e9edf3")
    draw = ImageDraw.Draw(image)
    font_title = _load_font(24)
    font_body = _load_font(20)
    font_small = _load_font(16)

    draw.rectangle((0, 0, width, 52), fill="#cad5e3")
    draw.text((28, 14), "Remote Control System", fill="#1e2b37", font=font_title)
    draw.rectangle((width - 58, 10, width - 18, 40), outline="#8d1e2f", width=2)
    draw.text((width - 45, 15), "X", fill="#8d1e2f", font=font_body)

    dialog = (300, 170, 980, 660)
    draw.rounded_rectangle(dialog, radius=18, fill="#ffffff", outline="#8fa2b8", width=3)

    rows = [
        ("Server", "ETCH-01"),
        ("User ID", "engineer"),
        ("Password", "********"),
    ]
    row_y = 280
    for label, value in rows:
        draw.text((380, row_y), label, fill="#24384b", font=font_body)
        draw.rounded_rectangle((520, row_y - 12, 820, row_y + 28), radius=8, outline="#6e8397", width=2)
        draw.text((540, row_y - 2), value, fill="#4a5e70", font=font_body)
        row_y += 92

    draw.rounded_rectangle((560, 555, 700, 603), radius=10, fill="#2e6fd2")
    draw.text((610, 568), "Log In", fill="#ffffff", font=font_body)
    draw.rounded_rectangle((722, 555, 862, 603), radius=10, outline="#8aa0b5", width=2)
    draw.text((772, 568), "Cancel", fill="#43576b", font=font_body)
    draw.text((560, 620), "바로가기 설정", fill="#2f7a8f", font=font_small)
    return image


def load_probe_image() -> tuple["Image.Image", str]:
    """진단용 이미지를 준비한다."""
    image_path = os.getenv("UI_TARS_DIAG_IMAGE_PATH", "").strip()
    if image_path:
        path = Path(image_path).expanduser()
        image = Image.open(path)
        print(f"[INFO] 진단 이미지 로드: {path}")
        return image, path.stem

    print("[INFO] UI_TARS_DIAG_IMAGE_PATH 미설정. 더미 로그인 이미지를 생성합니다.")
    return build_dummy_login_image(), "dummy_login"


def encode_image_to_webp_b64(image: "Image.Image", quality: int = 90) -> tuple[str, int, int]:
    """이미지를 WebP base64 로 인코딩한다."""
    width, height = image.size
    converted = image.convert("RGB") if image.mode != "RGB" else image
    buffer = BytesIO()
    converted.save(buffer, format="WEBP", quality=quality)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("utf-8")
    print(
        f"[INFO] 진단 이미지 인코딩: {width}x{height}, "
        f"WebP q={quality}, {len(payload) / 1024:.1f}KB"
    )
    return encoded, width, height


def sanitize_payload_for_log(payload: dict) -> dict:
    """요청 payload 를 로그 친화적으로 축약한다."""
    cloned = json.loads(json.dumps(payload))
    messages = cloned.get("messages")
    if not isinstance(messages, list):
        return cloned

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url", "") or "")
            part["image_url"] = {
                "url_prefix": url[:48] + "..." if url else "",
                "url_length": len(url),
            }
    return cloned


def extract_finish_reasons(data: object) -> list[str]:
    """응답 body 에서 finish_reason 을 추출한다."""
    return OpenAICompatibleVLMClient._extract_finish_reasons(data)


def extract_text(data: object, raw_text: str) -> str:
    """응답 body 에서 assistant text 를 추출한다."""
    if isinstance(data, dict):
        return OpenAICompatibleVLMClient._extract_text_from_json_body(data)
    return OpenAICompatibleVLMClient._extract_text_from_sse_body(raw_text)


def build_messages(
    *,
    system_message: str,
    user_text: str,
    image_b64: str = "",
) -> list[dict]:
    """진단 payload messages 를 구성한다."""
    messages: list[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    if image_b64:
        user_content: object = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
            },
        ]
    else:
        user_content = user_text

    messages.append({"role": "user", "content": user_content})
    return messages


def build_variant_payload(
    variant: str,
    *,
    model_name: str,
    image_b64: str,
    max_tokens: int,
) -> dict:
    """variant 이름으로 진단 payload 를 만든다."""
    system_message = ""
    if variant == "text_only_nonstream":
        messages = build_messages(
            system_message="",
            user_text="Hello. Describe what you can do in one sentence.",
            image_b64="",
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }

    if variant == "text_only_stream":
        messages = build_messages(
            system_message="",
            user_text="Hello. Describe what you can do in one sentence.",
            image_b64="",
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
        }

    if variant == "script_nonstream":
        _, user_text = build_login_rcs_ui_tars_prompt(
            target_keys=("userid_input", "password_input", "login_button")
        )
        messages = build_messages(
            system_message="",
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }

    if variant == "script_stream":
        _, user_text = build_login_rcs_ui_tars_prompt(
            target_keys=("userid_input", "password_input", "login_button")
        )
        messages = build_messages(
            system_message="",
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
        }

    if variant == "script_stream_with_system":
        _, user_text = build_login_rcs_ui_tars_prompt(
            target_keys=("userid_input", "password_input", "login_button")
        )
        messages = build_messages(
            system_message=(
                "GROUNDING task for a desktop GUI screenshot. "
                "Output only action lines."
            ),
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
        }

    if variant == "single_element_stream":
        system_message, user_text = build_single_element_prompt("login_button")
        messages = build_messages(
            system_message=system_message,
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
        }

    if variant == "script_stream_max_new_tokens":
        _, user_text = build_login_rcs_ui_tars_prompt(
            target_keys=("userid_input", "password_input", "login_button")
        )
        messages = build_messages(
            system_message="",
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "stream": True,
        }

    if variant == "script_stream_both_token_keys":
        _, user_text = build_login_rcs_ui_tars_prompt(
            target_keys=("userid_input", "password_input", "login_button")
        )
        messages = build_messages(
            system_message="",
            user_text=user_text,
            image_b64=image_b64,
        )
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "max_new_tokens": max_tokens,
            "stream": True,
        }

    raise ValueError(f"알 수 없는 variant: {variant}")


def run_variant(
    variant: str,
    *,
    endpoint: str,
    headers: dict[str, str],
    run_dir: Path,
    model_name: str,
    image_b64: str,
    max_tokens: int,
    timeout_sec: float,
    send_requests: bool,
) -> dict:
    """단일 variant 를 실행하고 결과를 저장한다."""
    payload = build_variant_payload(
        variant,
        model_name=model_name,
        image_b64=image_b64,
        max_tokens=max_tokens,
    )
    sanitized_payload = sanitize_payload_for_log(payload)
    request_path = run_dir / f"{variant}_request.json"
    response_path = run_dir / f"{variant}_response.txt"
    summary_path = run_dir / f"{variant}_summary.json"
    _write_json(request_path, sanitized_payload)

    if not send_requests:
        summary = {
            "variant": variant,
            "status": "dry_run",
            "request_path": str(request_path),
            "endpoint": endpoint,
        }
        _write_json(summary_path, summary)
        print(f"[DRYRUN] {variant:28s} endpoint={endpoint}")
        return summary

    started_at = time.time()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout_sec,
        )
        body_text = response.text
        elapsed_ms = (time.time() - started_at) * 1000
        _write_text(response_path, body_text)

        try:
            body_json = response.json()
        except ValueError:
            body_json = None

        text = extract_text(body_json, body_text)
        usage = body_json.get("usage") if isinstance(body_json, dict) else {}
        finish_reasons = extract_finish_reasons(body_json)
        summary = {
            "variant": variant,
            "status": "ok" if response.status_code < 400 else "http_error",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "elapsed_ms": round(elapsed_ms, 1),
            "usage": usage or {},
            "finish_reasons": finish_reasons,
            "response_chars": len(body_text),
            "extracted_text": text,
            "response_path": str(response_path),
        }
        _write_json(summary_path, summary)
        print(
            f"[RESULT] {variant:28s} status={response.status_code} "
            f"usage={usage or {}} finish={finish_reasons or []} "
            f"text={_truncate(text)}"
        )
        if response.status_code >= 400:
            print(f"[ERROR] {variant} raw body: {_truncate(body_text, limit=400)}")
        return summary
    except Exception as exc:
        elapsed_ms = (time.time() - started_at) * 1000
        summary = {
            "variant": variant,
            "status": "request_error",
            "elapsed_ms": round(elapsed_ms, 1),
            "error": str(exc),
            "endpoint": endpoint,
        }
        _write_json(summary_path, summary)
        print(f"[ERROR] {variant:28s} request failed: {exc}")
        return summary


def main() -> str:
    """UI-TARS 요청 형상 진단을 실행한다."""
    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        print(f"[ERROR] 서비스 {SERVICE_SLUG} 를 찾을 수 없습니다.")
        return "service_not_found"

    endpoint = f"{resolve_service_proxy_url(SERVICE_SLUG).rstrip('/')}/v1/chat/completions"
    api_key = resolve_service_api_key(SERVICE_SLUG)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    send_requests = env_flag("UI_TARS_DIAG_SEND_REQUESTS", default=True)
    max_tokens = env_int("UI_TARS_DIAG_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    timeout_sec = env_float("UI_TARS_DIAG_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    variants = parse_variants()
    debug_stamp = make_timestamp_tag()
    run_dir = DEBUG_IMAGE_DIR / "ui_tars_diagnostics" / debug_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    image, image_label = load_probe_image()
    jpeg_path = run_dir / f"{image_label}.jpg"
    webp_path = run_dir / f"{image_label}.webp"
    save_debug_jpeg(image, jpeg_path, log_name=LOG_NAME)
    save_debug_webp(image, webp_path, log_name=LOG_NAME)
    image_b64, width, height = encode_image_to_webp_b64(image)

    _write_json(
        run_dir / "run_config.json",
        {
            "endpoint": endpoint,
            "service": SERVICE_SLUG,
            "model_name": service_entry.model_name,
            "send_requests": send_requests,
            "max_tokens": max_tokens,
            "timeout_sec": timeout_sec,
            "image_width": width,
            "image_height": height,
            "variants": list(variants),
        },
    )

    print(
        f"[INFO] UI-TARS 진단 시작: endpoint={endpoint}, "
        f"image={width}x{height}, send_requests={send_requests}"
    )
    print(f"[INFO] 결과 저장 디렉터리: {run_dir}")

    summaries: list[dict] = []
    for variant in variants:
        summary = run_variant(
            variant,
            endpoint=endpoint,
            headers=headers,
            run_dir=run_dir,
            model_name=service_entry.model_name,
            image_b64=image_b64,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            send_requests=send_requests,
        )
        summaries.append(summary)

    _write_json(run_dir / "result_index.json", {"results": summaries})
    log_work2_event(
        component=COMPONENT_NAME,
        message="ui_tars_diagnostic_finished",
        log_name=LOG_NAME,
        endpoint=endpoint,
        send_requests=send_requests,
        variant_count=len(variants),
        run_dir=run_dir,
    )
    print("[INFO] UI-TARS 진단 완료")
    return "success"


if __name__ == "__main__":
    result = main()
    if result != "success":
        raise SystemExit(1)
