"""sample_code 공용 VLM 클라이언트 및 OCR 헬퍼.

이 모듈은 같은 폴더의 스크립트들이 공유하는 상수, VLM 호출, OCR 함수를 제공한다.
외부 의존성: requests 만 필요.
"""

import base64
import json
import time
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# 설정 — 필요 시 여기만 수정
# ──────────────────────────────────────────────
FLASK_API_BASE = "http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api"

VLM_SERVICE = "ui-venus"
VLM_MODEL = "ui-venus-1.5-8b"
VLM_URL = f"{FLASK_API_BASE}/vlm_serve/{VLM_SERVICE}"

OCR_SERVICE = "paddleocr-vl-1.5"
OCR_MODEL = "paddleocr-vl-1.5"
OCR_URL = f"{FLASK_API_BASE}/vlm_serve/{OCR_SERVICE}"

TIMEOUT_SEC = 120.0
SAMPLE_DIR = Path(__file__).parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ──────────────────────────────────────────────
# 최소 VLM 클라이언트
# ──────────────────────────────────────────────
def _detect_mime(image_bytes: bytes) -> str:
    """이미지 header 로 MIME type 을 추정한다."""
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def vlm_chat(
    base_url: str,
    model: str,
    image_path: Path,
    system_message: str,
    user_text: str,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """OpenAI-compatible VLM endpoint 에 이미지 + 텍스트를 보내고 응답을 반환한다.

    Returns:
        (response_text, token_usage_dict)
    """
    image_bytes = image_path.read_bytes()
    mime = _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ],
    })

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        if endpoint.endswith("/v1"):
            endpoint += "/chat/completions"
        else:
            endpoint += "/v1/chat/completions"

    resp = requests.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"VLM 응답에 choices 없음: {json.dumps(data, ensure_ascii=False)[:300]}")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        ).strip()

    return str(content), data.get("usage") or {}


def clean_model_text(raw_text: str) -> str:
    """모델 응답에서 코드 펜스를 제거한다."""
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def parse_json_response(raw_text: str, label: str) -> dict | None:
    """모델 응답을 JSON 으로 파싱한다."""
    cleaned = clean_model_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[WARNING] {label} JSON 파싱 실패: {exc}")
        print(f"[INFO] {label} Raw 응답:\n{cleaned}")
        return None


# ──────────────────────────────────────────────
# OCR 헬퍼
# ──────────────────────────────────────────────
def run_ocr(image_path: Path) -> str:
    """PaddleOCR-VL 로 이미지에서 텍스트를 추출한다."""
    print(f"[INFO] [OCR] 텍스트 추출 중... ({OCR_SERVICE})")
    start = time.time()
    try:
        text, _ = vlm_chat(OCR_URL, OCR_MODEL, image_path, "", "OCR:")
    except Exception as exc:
        print(f"[WARNING] OCR 호출 실패: {exc}")
        return ""
    elapsed = time.time() - start
    text = text.strip()
    print(f"[INFO] OCR 완료: {elapsed:.1f}초, {len(text)}자 추출")
    return text


def run_chart_ocr(image_path: Path) -> str:
    """PaddleOCR-VL 의 Chart Recognition 으로 차트 데이터를 추출한다."""
    print(f"[INFO] [OCR] 차트 인식 중... ({OCR_SERVICE})")
    start = time.time()
    try:
        text, _ = vlm_chat(OCR_URL, OCR_MODEL, image_path, "", "Chart Recognition:")
    except Exception as exc:
        print(f"[WARNING] Chart OCR 호출 실패: {exc}")
        return ""
    elapsed = time.time() - start
    text = text.strip()
    print(f"[INFO] Chart OCR 완료: {elapsed:.1f}초, {len(text)}자 추출")
    return text


def build_combined_ocr(image_path: Path) -> str:
    """OCR 텍스트와 차트 인식 결과를 하나로 합친다."""
    ocr_text = run_ocr(image_path)
    chart_text = run_chart_ocr(image_path)

    combined = ocr_text
    if chart_text:
        combined += f"\n\n[Chart Data]\n{chart_text}"
    if not combined.strip():
        combined = "(OCR 결과 없음 — 이미지만으로 분석)"
    return combined


def collect_images() -> list[Path]:
    """SAMPLE_DIR 내 이미지 파일 목록을 반환한다."""
    return sorted(
        p for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_output_dir(image_path: Path) -> Path:
    """이미지 이름(확장자 제외)으로 출력 폴더를 생성하고 반환한다."""
    output_dir = image_path.parent / image_path.stem
    output_dir.mkdir(exist_ok=True)
    return output_dir


def print_token_usage(usage: dict, prefix: str = "") -> None:
    """토큰 사용량을 출력한다."""
    if not usage:
        return
    label = f"[INFO] {prefix}토큰" if prefix else "[INFO] 토큰"
    print(
        f"{label}: prompt={usage.get('prompt_tokens', '?')}, "
        f"completion={usage.get('completion_tokens', '?')}, "
        f"total={usage.get('total_tokens', '?')}"
    )
