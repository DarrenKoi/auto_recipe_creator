"""RCS 로그인 스크린샷용 OCR 비교 스크립트.

목적:
- 보유 중인 OCR 서비스(`paddleocr-vl-1.5`, `got-ocr`)를 한 장의 로그인 스크린샷으로 비교한다.
- 원본 JPEG, 전송 WebP, raw 응답, 요약 JSON을 `poc/work2/debug_images/ocr_login_check/` 아래에 저장한다.
- `PaddleOCR-VL-1.5`는 `OCR:`와 `Spotting:`를 모두 호출해
  텍스트 파싱만 되는지, 위치 힌트가 나오는지 같이 확인한다.

사용법:
  uv run python poc/work2/ocr_login_check.py

주요 환경변수:
- `OCR_TEST_IMAGE_PATH`: 테스트할 로그인 스크린샷 절대/상대 경로
- `OCR_TEST_SERVICES`: `paddleocr-vl-1.5,got-ocr` 같은 서비스 목록
- `OCR_TEST_PADDLE_TASKS`: `OCR,Spotting`
- `OCR_TEST_TARGET_WORDS`: `Server,User ID,Password,Log In`
- `OCR_TEST_GOT_BOX`: `x1,y1,x2,y2` 형식. GOT-OCR에 특정 영역 box를 줄 때 사용
- `OCR_TEST_GOT_CROP_TO_PATCHES`: `true/false`
- `OCR_TEST_GOT_FORMAT_OUTPUT`: `true/false`
"""

import base64
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from PIL import Image

from poc.work2.flask_vlm import get_service_by_slug, resolve_service_api_key, resolve_service_proxy_url
from poc.work2.logger import log_work2_event
from poc.work2.util.debug_image_utils import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
    save_debug_text,
    save_debug_webp,
)
from poc.work2.vlm_client import Work2VLMClient


WORK2_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK2_DIR.parent.parent
DEBUG_ROOT_DIR = WORK2_DIR / "debug_images" / "ocr_login_check"
DEFAULT_SERVICES = ("paddleocr-vl-1.5", "got-ocr")
DEFAULT_PADDLE_TASKS = ("OCR", "Spotting")
DEFAULT_TARGET_WORDS = ("Server", "User ID", "Password", "Log In", "Login")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSITION_HINT_PATTERN = re.compile(
    r"(<loc_|bbox|polygon|quad|box|x1|y1|x2|y2|\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OCRRunSpec:
    """OCR 1회 실행 사양."""

    service_slug: str
    run_label: str
    request_kind: str
    prompt_text: str = ""
    got_box: list[int] | None = None
    got_format_output: bool = True
    got_crop_to_patches: bool = False


@dataclass(frozen=True)
class OCRRunResult:
    """OCR 1회 실행 결과."""

    service_slug: str
    model_name: str
    run_label: str
    request_kind: str
    prompt_text: str
    endpoint: str
    raw_response_path: str
    result_json_path: str
    normalized_lines: list[str]
    focus_hits: list[str]
    position_hints_detected: bool
    elapsed_ms: float
    token_usage: dict[str, int]
    error: str = ""


def _env_flag(name: str, default: bool = False) -> bool:
    """bool 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """쉼표/세미콜론 구분 환경변수를 튜플로 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default

    values: list[str] = []
    seen: set[str] = set()
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values) or default


def _normalize_paddle_task_name(raw_task: str) -> str:
    """PaddleOCR-VL 태스크 이름을 공식 keyword 형식으로 정규화한다."""
    task = raw_task.strip()
    lowered = task.lower().rstrip(":")
    mapping = {
        "ocr": "OCR:",
        "spotting": "Spotting:",
        "table recognition": "Table Recognition:",
        "formula recognition": "Formula Recognition:",
        "chart recognition": "Chart Recognition:",
        "seal recognition": "Seal Recognition:",
    }
    if lowered in mapping:
        return mapping[lowered]
    if not task.endswith(":"):
        return f"{task}:"
    return task


def _parse_got_box(raw: str) -> list[int] | None:
    """GOT-OCR box 환경변수를 `[x1, y1, x2, y2]`로 파싱한다."""
    text = raw.strip()
    if not text:
        return None

    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("OCR_TEST_GOT_BOX 는 x1,y1,x2,y2 형식이어야 합니다.")

    try:
        box = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("OCR_TEST_GOT_BOX 는 정수 4개여야 합니다.") from exc

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("OCR_TEST_GOT_BOX 는 x2>x1, y2>y1 이어야 합니다.")
    return box


def _candidate_score(path: Path) -> tuple[int, float]:
    """자동 탐색용 이미지 후보 점수를 계산한다."""
    name = path.name.lower()
    score = 0
    if "login" in name:
        score += 10
    if "capture" in name:
        score += 5
    if "dummy" in name:
        score -= 4
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        score += 2
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return score, mtime


def resolve_image_path() -> Path:
    """입력 이미지 경로를 결정한다."""
    configured = os.environ.get("OCR_TEST_IMAGE_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OCR_TEST_IMAGE_PATH 파일이 없습니다: {path}")
        return path

    search_roots = [
        WORK2_DIR / "debug_images",
        REPO_ROOT / "output",
        REPO_ROOT / "logs",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "login" not in path.name.lower():
                continue
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "로그인 스크린샷을 자동으로 찾지 못했습니다. "
            "OCR_TEST_IMAGE_PATH 환경변수로 경로를 지정하세요."
        )

    candidates.sort(key=_candidate_score, reverse=True)
    return candidates[0]


def _relative_text(path: Path) -> str:
    """출력용 경로 문자열을 정리한다."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _normalize_lines(raw_text: str, max_items: int = 40) -> list[str]:
    """raw OCR 응답을 사람이 보기 쉬운 줄 목록으로 정리한다."""
    if not raw_text.strip():
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", line)
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_items:
            break
    return lines


def _match_focus_words(raw_text: str, lines: list[str], target_words: tuple[str, ...]) -> list[str]:
    """관심 텍스트가 OCR 결과에 포함됐는지 찾는다."""
    search_space = "\n".join(lines) if lines else raw_text
    lowered = search_space.casefold()
    hits: list[str] = []
    seen: set[str] = set()
    for word in target_words:
        normalized_word = word.strip()
        if not normalized_word:
            continue
        if normalized_word.casefold() in lowered and normalized_word not in seen:
            hits.append(normalized_word)
            seen.add(normalized_word)
    return hits


def _detect_position_hints(raw_text: str) -> bool:
    """응답에 좌표/박스 힌트가 보이는지 대략 판단한다."""
    return bool(POSITION_HINT_PATTERN.search(raw_text))


def _load_image(image_path: Path) -> Image.Image:
    """이미지를 RGB PIL Image 로 읽는다."""
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            return image.convert("RGB")
        return image.copy()


def _build_got_ocr_endpoint(base_url: str) -> str:
    """GOT-OCR `/v1/ocr` 엔드포인트를 구성한다."""
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/ocr"
    return f"{normalized}/v1/ocr"


def build_run_specs() -> tuple[OCRRunSpec, ...]:
    """환경변수 기준 실행 사양 목록을 만든다."""
    services = _parse_csv_env("OCR_TEST_SERVICES", DEFAULT_SERVICES)
    paddle_tasks = tuple(
        _normalize_paddle_task_name(task)
        for task in _parse_csv_env("OCR_TEST_PADDLE_TASKS", DEFAULT_PADDLE_TASKS)
    )
    got_box = _parse_got_box(os.environ.get("OCR_TEST_GOT_BOX", ""))
    got_format_output = _env_flag("OCR_TEST_GOT_FORMAT_OUTPUT", default=True)
    got_crop_to_patches = _env_flag("OCR_TEST_GOT_CROP_TO_PATCHES", default=False)

    specs: list[OCRRunSpec] = []
    for service_slug in services:
        if service_slug == "paddleocr-vl-1.5":
            for task in paddle_tasks:
                run_label = f"{service_slug}_{task.rstrip(':').lower().replace(' ', '_')}"
                specs.append(
                    OCRRunSpec(
                        service_slug=service_slug,
                        run_label=run_label,
                        request_kind="chat_keyword",
                        prompt_text=task,
                    )
                )
            continue

        if service_slug == "got-ocr":
            specs.append(
                OCRRunSpec(
                    service_slug=service_slug,
                    run_label="got-ocr_ocr",
                    request_kind="ocr_endpoint",
                    got_box=got_box,
                    got_format_output=got_format_output,
                    got_crop_to_patches=got_crop_to_patches,
                )
            )
            continue

        raise ValueError(
            f"이 스크립트는 OCR 서비스만 지원합니다: {service_slug}. "
            "현재 지원: paddleocr-vl-1.5, got-ocr"
        )

    return tuple(specs)


def _save_input_artifacts(
    image: Image.Image,
    debug_dir: Path,
    model_name: str,
    run_label: str,
    timestamp_tag: str,
) -> Path:
    """원본 JPEG와 전송용 WebP를 저장한다."""
    jpeg_path = debug_image_path(
        debug_dir,
        f"{run_label}_capture.jpg",
        model_name=model_name,
        timestamp_tag=timestamp_tag,
    )
    webp_path = debug_image_path(
        debug_dir,
        f"{run_label}_input.webp",
        model_name=model_name,
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, jpeg_path, log_name="ocr_login_check")
    save_debug_webp(image, webp_path, quality=90, log_name="ocr_login_check")
    return webp_path


def _run_paddleocr_chat(
    spec: OCRRunSpec,
    image: Image.Image,
    debug_dir: Path,
    timestamp_tag: str,
    target_words: tuple[str, ...],
    timeout_sec: float,
) -> OCRRunResult:
    """PaddleOCR-VL chat-completions 호출."""
    service_entry = get_service_by_slug(spec.service_slug)
    if service_entry is None:
        raise ValueError(f"알 수 없는 service slug: {spec.service_slug}")

    webp_path = _save_input_artifacts(
        image,
        debug_dir,
        service_entry.model_name,
        spec.run_label,
        timestamp_tag,
    )
    raw_response_path = debug_image_path(
        debug_dir,
        f"{spec.run_label}_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        debug_dir,
        f"{spec.run_label}_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )

    client = Work2VLMClient(
        service_slug=spec.service_slug,
        timeout_sec=timeout_sec,
        log_name="ocr_login_check",
    )

    started_at = time.time()
    response = client.chat_with_image_path(
        image_path=webp_path,
        system_message="",
        user_text=spec.prompt_text,
        image_mime="image/webp",
        temperature=0.0,
        max_tokens=4096,
    )
    elapsed_ms = round((time.time() - started_at) * 1000, 1)

    raw_text = response.text.strip()
    normalized_lines = _normalize_lines(raw_text)
    focus_hits = _match_focus_words(raw_text, normalized_lines, target_words)
    position_hints_detected = _detect_position_hints(raw_text)

    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": spec.service_slug,
            "model_name": response.model_name,
            "run_label": spec.run_label,
            "request_kind": spec.request_kind,
            "prompt_text": spec.prompt_text,
            "endpoint": client.endpoint,
            "normalized_lines": normalized_lines,
            "focus_hits": focus_hits,
            "position_hints_detected": position_hints_detected,
            "token_usage": response.token_usage,
            "elapsed_ms": elapsed_ms,
        },
    )

    return OCRRunResult(
        service_slug=spec.service_slug,
        model_name=response.model_name,
        run_label=spec.run_label,
        request_kind=spec.request_kind,
        prompt_text=spec.prompt_text,
        endpoint=client.endpoint,
        raw_response_path=_relative_text(raw_response_path),
        result_json_path=_relative_text(result_json_path),
        normalized_lines=normalized_lines,
        focus_hits=focus_hits,
        position_hints_detected=position_hints_detected,
        elapsed_ms=elapsed_ms,
        token_usage=response.token_usage,
    )


def _run_got_ocr(
    spec: OCRRunSpec,
    image: Image.Image,
    debug_dir: Path,
    timestamp_tag: str,
    target_words: tuple[str, ...],
    timeout_sec: float,
) -> OCRRunResult:
    """GOT-OCR `/v1/ocr` 호출."""
    service_entry = get_service_by_slug(spec.service_slug)
    if service_entry is None:
        raise ValueError(f"알 수 없는 service slug: {spec.service_slug}")

    webp_path = _save_input_artifacts(
        image,
        debug_dir,
        service_entry.model_name,
        spec.run_label,
        timestamp_tag,
    )
    raw_response_path = debug_image_path(
        debug_dir,
        f"{spec.run_label}_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        debug_dir,
        f"{spec.run_label}_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )

    base_url = resolve_service_proxy_url(spec.service_slug)
    endpoint = _build_got_ocr_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    api_key = resolve_service_api_key(spec.service_slug)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    image_bytes = webp_path.read_bytes()
    payload: dict[str, object] = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "format_output": spec.got_format_output,
        "crop_to_patches": spec.got_crop_to_patches,
    }
    if spec.got_box is not None:
        payload["box"] = spec.got_box

    started_at = time.time()
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )
    elapsed_ms = round((time.time() - started_at) * 1000, 1)
    response.raise_for_status()

    body = response.json()
    raw_text = str(body.get("text", "")).strip()
    normalized_lines = _normalize_lines(raw_text)
    focus_hits = _match_focus_words(raw_text, normalized_lines, target_words)
    position_hints_detected = _detect_position_hints(raw_text)

    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": spec.service_slug,
            "model_name": str(body.get("model", service_entry.model_name)),
            "run_label": spec.run_label,
            "request_kind": spec.request_kind,
            "prompt_text": spec.prompt_text,
            "endpoint": endpoint,
            "got_box": spec.got_box,
            "format_output": spec.got_format_output,
            "crop_to_patches": spec.got_crop_to_patches,
            "normalized_lines": normalized_lines,
            "focus_hits": focus_hits,
            "position_hints_detected": position_hints_detected,
            "elapsed_ms": elapsed_ms,
            "response_body": body,
        },
    )

    return OCRRunResult(
        service_slug=spec.service_slug,
        model_name=str(body.get("model", service_entry.model_name)),
        run_label=spec.run_label,
        request_kind=spec.request_kind,
        prompt_text=spec.prompt_text,
        endpoint=endpoint,
        raw_response_path=_relative_text(raw_response_path),
        result_json_path=_relative_text(result_json_path),
        normalized_lines=normalized_lines,
        focus_hits=focus_hits,
        position_hints_detected=position_hints_detected,
        elapsed_ms=elapsed_ms,
        token_usage={},
    )


def run_ocr_check() -> None:
    """OCR 비교 실행."""
    image_path = resolve_image_path()
    image = _load_image(image_path)
    target_words = _parse_csv_env("OCR_TEST_TARGET_WORDS", DEFAULT_TARGET_WORDS)
    timeout_sec = float(os.environ.get("OCR_TEST_TIMEOUT_SEC", "120"))
    run_specs = build_run_specs()

    timestamp_tag = time.strftime("%y%m%d_%H%M%S")
    debug_dir = DEBUG_ROOT_DIR / timestamp_tag
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 입력 이미지: {_relative_text(image_path)}")
    print(f"[INFO] 이미지 크기: {image.size[0]}x{image.size[1]}")
    print(f"[INFO] OCR 실행 수: {len(run_specs)}")
    print(f"[INFO] 디버그 출력: {_relative_text(debug_dir)}")

    log_work2_event(
        component="ocr_login_check",
        message="started",
        log_name="ocr_login_check",
        image_path=_relative_text(image_path),
        debug_dir=_relative_text(debug_dir),
        run_count=len(run_specs),
    )

    results: list[OCRRunResult] = []
    for spec in run_specs:
        print(
            f"[INFO] OCR 실행 시작: service={spec.service_slug}, "
            f"run={spec.run_label}, kind={spec.request_kind}"
        )
        try:
            if spec.request_kind == "chat_keyword":
                result = _run_paddleocr_chat(
                    spec,
                    image,
                    debug_dir,
                    timestamp_tag,
                    target_words,
                    timeout_sec,
                )
            elif spec.request_kind == "ocr_endpoint":
                result = _run_got_ocr(
                    spec,
                    image,
                    debug_dir,
                    timestamp_tag,
                    target_words,
                    timeout_sec,
                )
            else:
                raise ValueError(f"지원하지 않는 request_kind: {spec.request_kind}")
        except Exception as exc:
            error_json_path = debug_image_path(
                debug_dir,
                f"{spec.run_label}_error.json",
                model_name=get_service_by_slug(spec.service_slug).model_name,
                timestamp_tag=timestamp_tag,
            )
            save_debug_json(
                error_json_path,
                {
                    "service_slug": spec.service_slug,
                    "run_label": spec.run_label,
                    "request_kind": spec.request_kind,
                    "prompt_text": spec.prompt_text,
                    "error": str(exc),
                },
            )
            print(
                f"[ERROR] OCR 실행 실패: service={spec.service_slug}, "
                f"run={spec.run_label}, error={exc}"
            )
            log_work2_event(
                component="ocr_login_check",
                message="run_failed",
                level="error",
                log_name="ocr_login_check",
                service=spec.service_slug,
                run_label=spec.run_label,
                error=str(exc),
            )
            results.append(
                OCRRunResult(
                    service_slug=spec.service_slug,
                    model_name=get_service_by_slug(spec.service_slug).model_name,
                    run_label=spec.run_label,
                    request_kind=spec.request_kind,
                    prompt_text=spec.prompt_text,
                    endpoint="",
                    raw_response_path="",
                    result_json_path=_relative_text(error_json_path),
                    normalized_lines=[],
                    focus_hits=[],
                    position_hints_detected=False,
                    elapsed_ms=0.0,
                    token_usage={},
                    error=str(exc),
                )
            )
            continue

        print(
            f"[INFO] OCR 실행 완료: service={result.service_slug}, "
            f"run={result.run_label}, elapsed_ms={result.elapsed_ms}, "
            f"focus_hits={', '.join(result.focus_hits) if result.focus_hits else '(없음)'}"
        )
        results.append(result)

    summary_path = debug_dir / "summary.json"
    save_debug_json(
        summary_path,
        {
            "image_path": _relative_text(image_path),
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "target_words": list(target_words),
            "results": [asdict(result) for result in results],
            "grounding_notes": {
                "paddleocr_vl_ocr": "기본 OCR: 텍스트 파싱 중심",
                "paddleocr_vl_spotting": "텍스트 + 위치 힌트 가능성 확인용. click grounding 계약은 아님",
                "got_ocr": "기본은 텍스트 OCR. box를 줄 때 특정 영역 OCR 가능, 스스로 GUI click target grounding을 하지는 않음",
            },
        },
    )

    print("\n[INFO] OCR 요약")
    for result in results:
        if result.error:
            print(
                f"[ERROR] {result.run_label}: {result.error} "
                f"(detail: {result.result_json_path})"
            )
            continue

        print(
            f"[INFO] {result.run_label}: lines={len(result.normalized_lines)}, "
            f"focus_hits={', '.join(result.focus_hits) if result.focus_hits else '(없음)'}, "
            f"position_hints={result.position_hints_detected}, "
            f"raw={result.raw_response_path}"
        )

    print("[INFO] Grounding 해석:")
    print("[INFO] - PaddleOCR-VL `OCR:`는 사실상 텍스트 파싱 확인용입니다.")
    print("[INFO] - PaddleOCR-VL `Spotting:`는 위치 힌트가 나올 수 있지만 GUI click grounding 전용 계약은 아닙니다.")
    print("[INFO] - GOT-OCR는 기본적으로 텍스트 OCR이며, `box`로 특정 영역을 읽게 할 수는 있어도 스스로 버튼 좌표를 고르는 모델은 아닙니다.")
    print(f"[INFO] 요약 JSON: {_relative_text(summary_path)}")

    log_work2_event(
        component="ocr_login_check",
        message="completed",
        log_name="ocr_login_check",
        image_path=_relative_text(image_path),
        summary_path=_relative_text(summary_path),
        success_count=sum(1 for item in results if not item.error),
        failure_count=sum(1 for item in results if item.error),
    )


if __name__ == "__main__":
    run_ocr_check()
