"""UI-TARS completion_tokens=1 원인 진단 스크립트.

UI-TARS-1.5-7B 에서 completion_tokens=1 로 응답이 잘리는 문제를 분리 테스트한다.
5가지 케이스를 순차 실행하여 어느 조합에서 정상 응답이 나오는지 확인한다.

테스트 케이스:
  1) 텍스트 전용 (이미지 없음) — 모델/vLLM 자체 문제 분리
  2) 이미지 + 텍스트 (text → image 순서) — 현재 vlm_client 기본 순서
  3) 이미지 + 텍스트 (image → text 순서) — UI-TARS 권장 순서
  4) system message 포함 + image → text 순서
  5) system message 포함 + image → text 순서 + frequency_penalty=1.0

사용법:
  uv run python poc/work2/test_ui_tars_config.py
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from poc.work2.flask_vlm import get_service_by_slug, resolve_service_proxy_url

load_dotenv()

SERVICE_SLUG = "ui-tars"
SEPARATOR = "=" * 72
THIN_SEPARATOR = "-" * 72


def _resolve_endpoint() -> tuple[str, str]:
    """UI-TARS 의 proxy URL 과 model name 을 반환한다."""
    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        print(f"[ERROR] 서비스 {SERVICE_SLUG} 를 찾을 수 없습니다.")
        sys.exit(1)

    proxy_url = resolve_service_proxy_url(SERVICE_SLUG)
    endpoint = f"{proxy_url.rstrip('/')}/v1/chat/completions"
    return endpoint, service_entry.model_name


def _create_synthetic_image() -> str:
    """간단한 로그인 대화상자 모양의 합성 이미지를 만들어 base64 WebP 로 반환한다."""
    import base64

    width, height = 400, 300
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # 타이틀 바
    draw.rectangle([0, 0, width, 30], fill=(0, 80, 160))
    draw.text((10, 8), "Remote Control System", fill="white")
    draw.text((width - 20, 8), "X", fill="white")

    # Server 필드
    draw.text((20, 50), "Server", fill="black")
    draw.rectangle([100, 45, 370, 65], outline="gray")

    # User ID 필드
    draw.text((20, 90), "User ID", fill="black")
    draw.rectangle([100, 85, 370, 105], outline="gray")

    # Password 필드
    draw.text((20, 130), "Password", fill="black")
    draw.rectangle([100, 125, 370, 145], outline="gray")

    # 버튼
    draw.rectangle([100, 200, 200, 230], fill=(0, 120, 200))
    draw.text((130, 208), "Log In", fill="white")
    draw.rectangle([220, 200, 320, 230], fill=(180, 180, 180))
    draw.text((245, 208), "Cancel", fill="black")

    buf = BytesIO()
    img.save(buf, format="WEBP", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[INFO] 합성 이미지 생성: {width}x{height}, {len(buf.getvalue()) / 1024:.1f}KB")
    return b64


def _send_request(
    endpoint: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 500,
    frequency_penalty: float | None = None,
) -> dict:
    """OpenAI-compatible 요청을 보내고 응답 dict 를 반환한다."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty

    started_at = time.time()
    try:
        resp = requests.post(endpoint, json=payload, timeout=60)
    except Exception as exc:
        return {"error": str(exc), "elapsed_ms": (time.time() - started_at) * 1000}

    elapsed_ms = (time.time() - started_at) * 1000

    try:
        body = resp.json()
    except Exception:
        return {
            "error": f"status={resp.status_code}, non-JSON body: {resp.text[:300]}",
            "elapsed_ms": elapsed_ms,
        }

    usage = body.get("usage") or {}
    choices = body.get("choices") or []
    assistant_text = ""
    if choices:
        msg = choices[0].get("message") or choices[0].get("delta") or {}
        assistant_text = msg.get("content", "") or ""

    return {
        "status_code": resp.status_code,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "assistant_text": assistant_text,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "elapsed_ms": elapsed_ms,
    }


def _print_result(label: str, result: dict) -> None:
    """테스트 결과를 출력한다."""
    if "error" in result:
        print(f"  결과: ERROR — {result['error']}")
        print(f"  경과: {result['elapsed_ms']:.0f}ms")
        return

    ct = result.get("completion_tokens")
    text = (result.get("assistant_text") or "").strip()
    preview = text[:200] + ("..." if len(text) > 200 else "")

    verdict = "OK" if ct is not None and ct > 1 else "FAIL (1 token)"
    print(f"  판정: {verdict}")
    print(f"  completion_tokens: {ct}")
    print(f"  prompt_tokens:     {result.get('prompt_tokens')}")
    print(f"  finish_reason:     {result.get('finish_reason')}")
    print(f"  경과: {result['elapsed_ms']:.0f}ms")
    print(f"  응답: {preview!r}")


# ── 테스트 케이스 ────────────────────────────────────────────────────

def test_1_text_only(endpoint: str, model: str) -> dict:
    """텍스트 전용 요청 — 모델/vLLM 자체 문제 분리."""
    messages = [
        {"role": "user", "content": "Hello, describe what you can do in one sentence."},
    ]
    return _send_request(endpoint, model, messages)


def test_2_text_then_image(endpoint: str, model: str, image_b64: str) -> dict:
    """이미지 포함: text → image 순서 (현재 vlm_client 기본 순서)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Click on the 'Log In' button."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                },
            ],
        },
    ]
    return _send_request(endpoint, model, messages)


def test_3_image_then_text(endpoint: str, model: str, image_b64: str) -> dict:
    """이미지 포함: image → text 순서 (UI-TARS 권장 순서)."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                },
                {"type": "text", "text": "Click on the 'Log In' button."},
            ],
        },
    ]
    return _send_request(endpoint, model, messages)


def test_4_system_msg_image_then_text(endpoint: str, model: str, image_b64: str) -> dict:
    """system message 포함 + image → text 순서."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                },
                {"type": "text", "text": "Click on the 'Log In' button."},
            ],
        },
    ]
    return _send_request(endpoint, model, messages)


def test_5_full_config(endpoint: str, model: str, image_b64: str) -> dict:
    """system message + image → text + frequency_penalty=1.0."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                },
                {"type": "text", "text": "Click on the 'Log In' button."},
            ],
        },
    ]
    return _send_request(endpoint, model, messages, frequency_penalty=1.0)


# ── 메인 ─────────────────────────────────────────────────────────────

TESTS = [
    ("1. 텍스트 전용 (이미지 없음)", test_1_text_only, False),
    ("2. text → image 순서 (현재 기본)", test_2_text_then_image, True),
    ("3. image → text 순서 (UI-TARS 권장)", test_3_image_then_text, True),
    ("4. system msg + image → text", test_4_system_msg_image_then_text, True),
    ("5. system msg + image → text + freq_penalty=1.0", test_5_full_config, True),
]


def main() -> None:
    """모든 테스트 케이스를 실행하고 결과 요약을 출력한다."""
    endpoint, model = _resolve_endpoint()
    print(SEPARATOR)
    print("UI-TARS completion_tokens=1 진단")
    print(f"  endpoint: {endpoint}")
    print(f"  model:    {model}")
    print(SEPARATOR)

    image_b64 = _create_synthetic_image()

    results: list[tuple[str, dict]] = []

    for label, test_fn, needs_image in TESTS:
        print(f"\n{THIN_SEPARATOR}")
        print(f"테스트: {label}")
        print(THIN_SEPARATOR)

        if needs_image:
            result = test_fn(endpoint, model, image_b64)
        else:
            result = test_fn(endpoint, model)

        _print_result(label, result)
        results.append((label, result))

    # ── 요약 테이블 ──────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("요약")
    print(SEPARATOR)
    print(f"{'테스트':<50s} {'tokens':>6s}  {'판정':<12s}")
    print(THIN_SEPARATOR)
    for label, result in results:
        ct = result.get("completion_tokens")
        if "error" in result:
            verdict = "ERROR"
        elif ct is not None and ct > 1:
            verdict = "OK"
        else:
            verdict = "FAIL"
        ct_str = str(ct) if ct is not None else "err"
        print(f"{label:<50s} {ct_str:>6s}  {verdict:<12s}")

    print(SEPARATOR)

    # 진단 힌트
    print("\n진단 가이드:")
    r1 = results[0][1]
    r2 = results[1][1]
    r3 = results[2][1]
    r4 = results[3][1]

    ct1 = r1.get("completion_tokens") if "error" not in r1 else None
    ct2 = r2.get("completion_tokens") if "error" not in r2 else None
    ct3 = r3.get("completion_tokens") if "error" not in r3 else None
    ct4 = r4.get("completion_tokens") if "error" not in r4 else None

    if ct1 is not None and ct1 <= 1:
        print("  → 테스트 1 FAIL: 모델/vLLM 자체 문제. chat_template 또는 vLLM 설정 확인 필요.")
    elif ct2 is not None and ct2 <= 1 and ct3 is not None and ct3 > 1:
        print("  → content 순서 문제: image → text 순서로 변경하면 해결됨.")
        print("    vlm_client.py 의 content 배열 순서를 수정하세요.")
    elif ct3 is not None and ct3 <= 1 and ct4 is not None and ct4 > 1:
        print("  → system message 누락 문제: system role 메시지를 추가하면 해결됨.")
    elif ct1 is not None and ct1 > 1 and ct2 is not None and ct2 <= 1 and ct3 is not None and ct3 <= 1:
        print("  → 이미지 처리 문제: 텍스트는 되지만 이미지가 포함되면 실패.")
        print("    vLLM 의 이미지 전처리 또는 --limit-mm-per-prompt 설정 확인 필요.")
    else:
        print("  → 결과 조합을 비교하여 어떤 변수가 영향을 주는지 확인하세요.")


if __name__ == "__main__":
    main()
