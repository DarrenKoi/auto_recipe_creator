"""RCS 로그인 화면에서 여러 VLM 모델의 좌표 검출 정확도를 비교하는 스크립트 (Windows 전용).

Flask health 에서 현재 serving 상태인 UI VLM 서비스를 자동 발견하고,
동일한 로그인 화면 스크린샷으로 각 모델의 UI 요소 좌표 검출 능력을 비교한다.
OCR assist 파이프라인은 보조 힌트 생성에 선택적으로 사용된다.
pywinauto 는 창 실행·탐색에만 사용한다.

환경변수:
  RCS_LOGIN_SERVICES=ui-venus,mai-ui   # 비교 대상 서비스 (미지정 시 health 에서 자동 선택)
  RCS_LOGIN_INCLUDE_OCR_SERVICE=true   # OCR 계열 서비스도 비교 대상에 포함
  RCS_TARGET_CLICK_KEY=login_button    # 벤치마크 후 클릭할 요소 키

사용법:
  uv run python poc/work2/automate_rcs_login.py
"""

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from pywinauto import Desktop, mouse
from pywinauto.application import Application
from pywinauto.keyboard import send_keys

from flask_api.vlm_serve.config import get_service_by_slug
from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient
from poc.work2.flask_vlm import (
    apply_pipeline_env_defaults,
    fetch_vlm_health,
    normalize_vlm_health_entries,
    resolve_service_proxy_url,
)
from poc.work2.logger import log_vlm_call
from poc.work2.pipeline_ocr import build_ocr_extra_instructions, collect_ocr_hint_result
from poc.work2.prompts import build_rcs_login_locator_prompt
from poc.work2.rcs_utils import (
    capture_window,
    click_at,
    debug_image_path,
    encode_image_webp,
    extract_json,
    find_existing_main_window,
    is_main_window_title,
    parse_coords,
    save_marked_image,
    scan_window_list,
)

PIPELINE_CONFIG = apply_pipeline_env_defaults()

# ─────────────────────────── 설정 ───────────────────────────

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "win32"
MAIN_WINDOW_TITLE_REGEX = (
    os.environ.get("RCS_MAIN_WINDOW_REGEX", r"\brcs\b.*\[server\s*:[^\]]+\]").strip()
    or r"\brcs\b.*\[server\s*:[^\]]+\]"
)
DEBUG_MAIN_WINDOW_TITLES = (
    os.environ.get("RCS_DEBUG_MAIN_WINDOW_TITLES", "0").strip().lower()
    not in {"0", "false", "no", "off"}
)
_desktop_backends_raw = [
    item.strip().lower()
    for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "win32,uia").split(",")
    if item.strip()
]
_desktop_backends = _desktop_backends_raw + [PYWINAUTO_BACKEND]
DESKTOP_SCAN_BACKENDS = tuple(
    dict.fromkeys(b for b in _desktop_backends if b in {"uia", "win32"})
) or ("uia", "win32")

LAUNCH_TIMEOUT = 30.0
WINDOW_TITLE_PREFIX = "Remote Control System"

# 벤치마크 후 클릭할 요소 키
TARGET_CLICK_KEY = os.environ.get("RCS_TARGET_CLICK_KEY", "login_button").strip() or "login_button"

# health 에서 서비스를 찾지 못할 때 기본 비교 대상
DEFAULT_TARGET_SERVICE_SLUGS = ("ui-venus", "mai-ui")

# OCR assist 에서 검증할 키워드
LOGIN_OCR_FOCUS_WORDS = (
    "Server",
    "User ID",
    "Password",
    "Log In",
    "Cancel",
)

# 검출 대상: 텍스트 라벨 + 입력 필드 + 버튼
TARGET_ELEMENTS = [
    "server_label",
    "server_input",
    "userid_label",
    "userid_input",
    "password_label",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
]

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0

try:
    POST_LOGIN_DELAY_SEC = float(os.getenv("RCS_POST_LOGIN_DELAY_SEC", "4.0"))
except ValueError:
    POST_LOGIN_DELAY_SEC = 4.0

try:
    POST_LOGIN_MAIN_TIMEOUT_SEC = float(os.getenv("RCS_POST_LOGIN_MAIN_TIMEOUT_SEC", "240.0"))
except ValueError:
    POST_LOGIN_MAIN_TIMEOUT_SEC = 240.0

try:
    POST_LOGIN_POLL_SEC = float(os.getenv("RCS_POST_LOGIN_POLL_SEC", "0.5"))
except ValueError:
    POST_LOGIN_POLL_SEC = 0.5

try:
    CLICK_RETRY_COUNT = int(os.getenv("RCS_CLICK_RETRY_COUNT", "2"))
except ValueError:
    CLICK_RETRY_COUNT = 2

try:
    CLICK_RETRY_DELAY_SEC = float(os.getenv("RCS_CLICK_RETRY_DELAY_SEC", "0.25"))
except ValueError:
    CLICK_RETRY_DELAY_SEC = 0.25

try:
    POST_LOGIN_SCROLL_STEPS = int(os.getenv("RCS_POST_LOGIN_SCROLL_STEPS", "0"))
except ValueError:
    POST_LOGIN_SCROLL_STEPS = 0

try:
    POST_LOGIN_SCROLL_INTERVAL = float(os.getenv("RCS_POST_LOGIN_SCROLL_INTERVAL", "0.3"))
except ValueError:
    POST_LOGIN_SCROLL_INTERVAL = 0.3

POST_LOGIN_SCROLL_MODE = (
    os.getenv("RCS_POST_LOGIN_SCROLL_MODE", "wheel").strip().lower() or "wheel"
)  # wheel | keys | combo

DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"

ELEMENT_COLORS = {
    "server_label": "red",
    "server_input": "salmon",
    "userid_label": "blue",
    "userid_input": "deepskyblue",
    "password_label": "green",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}


# ─────────────────────────── 환경변수 헬퍼 ───────────────────────────


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """콤마/세미콜론 구분 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    values: list[str] = []
    seen: set[str] = set()
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(values) or default


def _env_flag(name: str, default: bool = False) -> bool:
    """bool 형 환경변수를 해석한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _looks_like_ocr_service(row: dict[str, Any]) -> bool:
    """service/model 이름 기반으로 OCR 계열 여부를 추정한다."""
    haystack = " ".join(
        [
            str(row.get("service") or ""),
            str(row.get("display_name") or ""),
            str(row.get("expected_model") or ""),
        ]
    ).lower()
    return "ocr" in haystack


# ─────────────────────────── VLM 서비스 탐색 ───────────────────────────


def _discover_benchmark_targets(
    pipeline_config: dict[str, object],
) -> list[dict[str, Any]]:
    """Flask health 에서 벤치마크 대상 VLM 서비스를 발견한다."""
    flask_base_url = str(pipeline_config.get("flask_api_base_url") or "").rstrip("/")
    explicit_services = _parse_csv_env("RCS_LOGIN_SERVICES", ())
    include_ocr = _env_flag("RCS_LOGIN_INCLUDE_OCR_SERVICE", False)

    # 1) Health 에서 서비스 목록 가져오기
    health_rows: list[dict[str, Any]] = []
    if flask_base_url:
        try:
            health_body = fetch_vlm_health(flask_base_url=flask_base_url, timeout_sec=10.0)
            health_rows = normalize_vlm_health_entries(health_body, flask_base_url=flask_base_url)
        except (requests.RequestException, ValueError) as exc:
            print(f"[WARNING] health 기반 서비스 탐색 실패: {exc}")

    if health_rows:
        print("[INFO] health 기반 서비스 상태:")
        for row in health_rows:
            print(
                f"  {row['service']} -> health={row.get('health_status') or '-'}, "
                f"model={row.get('expected_model') or '-'}, "
                f"proxy_registered={row.get('proxy_registered')}"
            )

    # 2) 환경변수로 명시된 서비스가 있으면 사용
    if explicit_services:
        health_map = {str(row["service"]): row for row in health_rows}
        targets: list[dict[str, Any]] = []
        for slug in explicit_services:
            if slug in health_map:
                targets.append(health_map[slug])
            else:
                entry = get_service_by_slug(slug)
                if entry and entry.enabled:
                    targets.append(
                        {
                            "service": slug,
                            "display_name": entry.display_name,
                            "expected_model": entry.model_name,
                            "health_status": "",
                            "proxy_registered": True,
                            "api_url": resolve_service_proxy_url(
                                slug, flask_base_url=flask_base_url
                            ),
                        }
                    )
                else:
                    print(f"[WARNING] 알 수 없거나 비활성 서비스: {slug}")
        return targets

    # 3) Health 에서 serving + proxy_registered 인 서비스 자동 선택
    if health_rows:
        serving = [
            row
            for row in health_rows
            if str(row.get("health_status") or "").strip().lower() == "serving"
            and bool(row.get("proxy_registered"))
        ]
        if not include_ocr:
            ui_rows = [row for row in serving if not _looks_like_ocr_service(row)]
            if ui_rows:
                serving = ui_rows
        if serving:
            return serving

    # 4) Fallback: 기본 서비스 슬러그 사용
    fallback_targets: list[dict[str, Any]] = []
    for slug in DEFAULT_TARGET_SERVICE_SLUGS:
        entry = get_service_by_slug(slug)
        api_url = ""
        if flask_base_url:
            api_url = resolve_service_proxy_url(slug, flask_base_url=flask_base_url)
        elif entry and entry.enabled:
            api_url = resolve_service_proxy_url(slug)
        fallback_targets.append(
            {
                "service": slug,
                "display_name": entry.display_name if entry else slug,
                "expected_model": entry.model_name if entry else "",
                "health_status": "",
                "proxy_registered": bool(entry and entry.enabled),
                "api_url": api_url,
            }
        )
    return fallback_targets


# ─────────────────────────── 창 탐색 ───────────────────────────


def _main_title_matcher(title: str) -> bool:
    """모듈 설정 MAIN_WINDOW_TITLE_REGEX 기준 매칭 함수."""
    return is_main_window_title(title, MAIN_WINDOW_TITLE_REGEX)


def _python_bitness() -> int:
    """현재 Python 인터프리터의 비트 수를 반환한다."""
    return 64 if sys.maxsize > 2**32 else 32


def _exe_bitness(exe_path: Path) -> int | None:
    """PE 헤더를 읽어 실행 파일 비트 수를 판별한다."""
    try:
        with exe_path.open("rb") as fp:
            if fp.read(2) != b"MZ":
                return None
            fp.seek(0x3C)
            e_lfanew = struct.unpack("<I", fp.read(4))[0]
            fp.seek(e_lfanew + 4)
            machine = struct.unpack("<H", fp.read(2))[0]
    except OSError:
        return None

    if machine == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
        return 64
    if machine == 0x14C:  # IMAGE_FILE_MACHINE_I386
        return 32
    return None


def _resolve_backend(exe_path: Path) -> str:
    """혼합 비트 환경에서 32/64비트 호환성이 높은 백엔드를 선택한다."""
    backend = PYWINAUTO_BACKEND
    exe_bits = _exe_bitness(exe_path)
    py_bits = _python_bitness()

    if exe_bits and exe_bits != py_bits and backend == "win32":
        print(
            f"[INFO] 비트 수 불일치 감지 (Python={py_bits}-bit, RCS EXE={exe_bits}-bit). "
            "win32 백엔드 대신 uia를 사용해 32비트 앱 자동화 이슈를 우회합니다."
        )
        return "uia"

    return backend


def _wait_for_login_window(app):
    """'Remote Control System' 으로 시작하는 창이 나타날 때까지 대기."""
    deadline = time.time() + LAUNCH_TIMEOUT
    while time.time() < deadline:
        for win in app.windows():
            try:
                title = win.window_text() or ""
            except Exception:
                continue
            if title.startswith(WINDOW_TITLE_PREFIX):
                return win
        time.sleep(0.5)
    raise TimeoutError(f"로그인 창을 {LAUNCH_TIMEOUT:.0f}초 내에 찾지 못했습니다")


def _scan_main_window_candidates(app):
    """현재 app.windows()의 타이틀과 매칭 결과를 수집한다."""
    debug_rows = []

    # 1) 시작 프로세스(app)에서 먼저 탐색
    app_window, app_title = scan_window_list(
        app.windows(), "app", debug_rows, _main_title_matcher
    )
    if app_window is not None:
        return app_window, app_title, debug_rows

    # 2) RCS가 재기동되어 프로세스가 바뀌는 경우를 위해 데스크톱 전체 창에서 탐색
    for backend in DESKTOP_SCAN_BACKENDS:
        try:
            desktop_windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True
            )
        except Exception as exc:
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        desktop_window, desktop_title = scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows, _main_title_matcher
        )
        if desktop_window is not None:
            return desktop_window, desktop_title, debug_rows

    return None, "", debug_rows


def _wait_for_post_login_windows(app):
    """로그인 버튼 클릭 후 메인 RCS 창이 나타날 때까지 대기한다."""
    if POST_LOGIN_DELAY_SEC > 0:
        print(f"[INFO] 로그인 직후 안정화 대기: {POST_LOGIN_DELAY_SEC:.1f}초")
        time.sleep(POST_LOGIN_DELAY_SEC)

    print(
        f"[INFO] 메인 RCS 창 대기 시작 (최대 {POST_LOGIN_MAIN_TIMEOUT_SEC:.0f}초, "
        f"poll={POST_LOGIN_POLL_SEC:.1f}s)"
    )
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")

    main_deadline = time.time() + POST_LOGIN_MAIN_TIMEOUT_SEC
    attempt = 0
    while time.time() < main_deadline:
        attempt += 1
        main_window, main_title, debug_rows = _scan_main_window_candidates(app)
        if DEBUG_MAIN_WINDOW_TITLES:
            elapsed = POST_LOGIN_MAIN_TIMEOUT_SEC - max(0.0, main_deadline - time.time())
            print(f"[DEBUG] title-scan attempt={attempt}, elapsed={elapsed:.1f}s")
            if not debug_rows:
                print("[DEBUG] app.windows() returned no visible top-level windows")
            else:
                for row in debug_rows:
                    print(f"[DEBUG] {row}")

        if main_window is not None:
            print(f"[INFO] 로그인 성공 창 발견: '{main_title}'")
            return main_window
        time.sleep(POST_LOGIN_POLL_SEC)

    print(f"[ERROR] 메인 RCS 창을 {POST_LOGIN_MAIN_TIMEOUT_SEC:.0f}초 내에 찾지 못했습니다.")
    return None


# ─────────────────────────── 스크롤 ───────────────────────────


def _scroll_to_reveal_more(window) -> None:
    """로그인 후 큰 화면에서 더 많은 텍스트 항목을 보도록 스크롤한다."""
    if POST_LOGIN_SCROLL_STEPS <= 0:
        return

    mode = POST_LOGIN_SCROLL_MODE
    if mode not in {"wheel", "keys", "combo"}:
        mode = "wheel"

    rect = window.rectangle()
    center_x = (rect.left + rect.right) // 2
    center_y = (rect.top + rect.bottom) // 2
    steps = POST_LOGIN_SCROLL_STEPS

    if mode == "combo":
        print(f"[INFO] Combo mode scroll: 단계 수={steps}")
        send_keys("{F4}")
        time.sleep(0.25)
        for _ in range(steps):
            send_keys("{DOWN}")
            time.sleep(POST_LOGIN_SCROLL_INTERVAL)
        send_keys("{ESC}")
        return

    if mode == "keys":
        print(f"[INFO] Key mode scroll: 단계 수={steps}, 대상=PageDown")
        for _ in range(steps):
            send_keys("{PGDN}")
            time.sleep(POST_LOGIN_SCROLL_INTERVAL)
        return

    print(f"[INFO] Wheel mode scroll: 단계 수={steps}, 좌표=({center_x}, {center_y})")
    for _ in range(steps):
        mouse.scroll(coords=(center_x, center_y), wheel_dist=-1)
        time.sleep(POST_LOGIN_SCROLL_INTERVAL)


# ─────────────────────────── 벤치마크 실행 ───────────────────────────


def _run_benchmark(
    window,
    service_targets: list[dict[str, Any]],
    pipeline_config: dict[str, object],
) -> dict | None:
    """여러 VLM 모델에 동일 스크린샷을 보내 좌표 검출 결과를 비교한다."""
    image = capture_window(window)
    img_b64, w, h = encode_image_webp(image)

    rect = window.rectangle()
    print(
        f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
        f"size={rect.right - rect.left}x{rect.bottom - rect.top}"
    )

    # OCR assist (한 번만 실행, 모든 모델에 동일 힌트 제공)
    ocr_result = collect_ocr_hint_result(
        image_b64=img_b64,
        image_width=w,
        image_height=h,
        image_mime="image/webp",
        pipeline_config=pipeline_config,
        context_label="RCS login dialog",
        focus_words=LOGIN_OCR_FOCUS_WORDS,
    )
    ocr_extra = build_ocr_extra_instructions(ocr_result)

    system_msg, prompt = build_rcs_login_locator_prompt(
        width=w,
        height=h,
        target_keys=TARGET_ELEMENTS,
        extra_instructions=ocr_extra,
    )

    api_key = str(pipeline_config.get("primary_api_key") or "")
    results: list[dict[str, Any]] = []

    for target in service_targets:
        service_slug = str(target["service"])
        model_name = str(target.get("expected_model") or "")
        target_api_url = str(target.get("api_url") or "").strip()

        print(f"\n{'=' * 60}")
        print(f"[INFO] 모델 테스트: {service_slug} ({model_name})")
        print("=" * 60)

        if not target_api_url or not model_name:
            reason = "api_url missing" if not target_api_url else "model_name missing"
            print(f"[ERROR] {service_slug}: {reason} — 건너뜀")
            results.append(
                {
                    "service": service_slug,
                    "model": model_name,
                    "ok": False,
                    "error": reason,
                    "detected": 0,
                    "latency_ms": 0,
                    "data": {},
                }
            )
            continue

        client = LangChainOpenAICompatibleVLMClient(
            base_url=target_api_url,
            api_key=api_key,
            timeout_sec=120.0,
        )

        try:
            request = ChatImageRequest(
                model=model_name,
                system_message=system_msg,
                user_text=prompt,
                image_b64=img_b64,
                temperature=VLM_TEMPERATURE,
            )
            print(f"[INFO] VLM 호출: model={model_name}, endpoint={client.endpoint}")
            start = time.time()
            raw = client.chat_with_image(request)
            elapsed = (time.time() - start) * 1000

            log_vlm_call(
                service=service_slug,
                model=model_name,
                status="ok",
                latency_ms=elapsed,
                token_usage=client.last_token_usage,
                endpoint=client.endpoint,
            )

            print(f"[INFO] 응답 수신 ({elapsed:.0f}ms)")
            print(f"[INFO] 원문 응답:\n{raw}\n")

            data = extract_json(raw)
            print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
            data = parse_coords(data, TARGET_ELEMENTS, w, h)

            detected = sum(
                1 for k in TARGET_ELEMENTS if k in data and isinstance(data[k], dict)
            )
            print(f"[INFO] 검출률: {detected}/{len(TARGET_ELEMENTS)}")

            out_path = debug_image_path(
                DEBUG_IMAGE_DIR,
                "debug_login.png",
                model_name=model_name,
            )
            save_marked_image(image, data, ELEMENT_COLORS, out_path)

            results.append(
                {
                    "service": service_slug,
                    "model": model_name,
                    "ok": True,
                    "error": None,
                    "detected": detected,
                    "latency_ms": round(elapsed, 1),
                    "data": data,
                }
            )

        except Exception as exc:
            log_vlm_call(
                service=service_slug,
                model=model_name,
                status="error",
                latency_ms=0,
                error=str(exc),
                endpoint=client.endpoint,
            )
            print(f"[ERROR] {service_slug} 실패: {exc}")
            results.append(
                {
                    "service": service_slug,
                    "model": model_name,
                    "ok": False,
                    "error": str(exc),
                    "detected": 0,
                    "latency_ms": 0,
                    "data": {},
                }
            )

    _print_benchmark_summary(results)

    # 첫 번째 성공 결과의 좌표 데이터를 반환
    for result in results:
        if result["ok"] and result["detected"] > 0:
            return result["data"]
    return None


def _print_benchmark_summary(results: list[dict[str, Any]]) -> None:
    """벤치마크 결과를 비교 테이블로 출력한다."""
    total = len(TARGET_ELEMENTS)

    print(f"\n{'=' * 90}")
    print("  RCS 로그인 VLM 좌표 검출 비교 결과")
    print("=" * 90)
    print(
        f"  {'서비스':<18} {'모델':<24} {'상태':<8} "
        f"{'검출':<10} {'응답시간':<12} {'비고'}"
    )
    print(
        f"  {'─' * 18} {'─' * 24} {'─' * 8} "
        f"{'─' * 10} {'─' * 12} {'─' * 24}"
    )

    for result in results:
        service = str(result["service"])
        model = str(result["model"])[:24]
        status = "OK" if result["ok"] else "FAIL"
        detected = f"{result['detected']}/{total}"
        latency = f"{result['latency_ms']}ms" if result["latency_ms"] else "-"

        note = ""
        if result.get("error"):
            note = str(result["error"])[:36]
        elif result["ok"]:
            data = result.get("data", {})
            missing = [
                k
                for k in TARGET_ELEMENTS
                if k not in data or not isinstance(data.get(k), dict)
            ]
            if missing:
                note = f"miss: {', '.join(missing[:3])}"
            else:
                note = "all elements detected"

        print(
            f"  {service:<18} {model:<24} {status:<8} "
            f"{detected:<10} {latency:<12} {note}"
        )

    print("=" * 90)


# ─────────────────────────── 메인 ───────────────────────────


def main() -> int:
    print(
        f"[INFO] pipeline: primary={PIPELINE_CONFIG['primary_service']} "
        f"({PIPELINE_CONFIG['primary_model_name']}), "
        f"ocr={PIPELINE_CONFIG['ocr_service']} "
        f"({PIPELINE_CONFIG['ocr_model_name']})"
    )

    existing_window, existing_title, existing_debug_rows = find_existing_main_window(
        DESKTOP_SCAN_BACKENDS, _main_title_matcher
    )
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")
        if not existing_debug_rows:
            print("[DEBUG] existing-check: no visible top-level windows")
        else:
            for row in existing_debug_rows:
                print(f"[DEBUG] existing-check {row}")
    if existing_window is not None:
        print(f"[INFO] 이미 로그인된 RCS 창 감지: '{existing_title}'")
        print("[INFO] 로그인 절차를 건너뜁니다.")
        return 0

    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return 1

    print(f"[INFO] RCS 시작: {RCS_EXE}")
    backend = _resolve_backend(RCS_EXE)
    cmd_str = subprocess.list2cmdline([str(RCS_EXE)])
    print(f"[INFO] pywinauto 백엔드: {backend}")
    app = Application(backend=backend).start(cmd_str, wait_for_idle=False)

    try:
        login_window = _wait_for_login_window(app)
        print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    time.sleep(1.0)

    # VLM 서비스 탐색
    benchmark_targets = _discover_benchmark_targets(PIPELINE_CONFIG)
    if not benchmark_targets:
        print("[ERROR] 벤치마크 대상 VLM 서비스가 없습니다.")
        return 2

    print(
        f"[INFO] 벤치마크 대상: "
        f"{', '.join(str(t['service']) for t in benchmark_targets)}"
    )

    data = _run_benchmark(login_window, benchmark_targets, PIPELINE_CONFIG)
    if not data:
        return 4

    if not click_at(
        TARGET_CLICK_KEY,
        login_window,
        data,
        retry_count=CLICK_RETRY_COUNT,
        retry_delay_sec=CLICK_RETRY_DELAY_SEC,
    ):
        return 5

    if TARGET_CLICK_KEY != "login_button":
        print(
            f"[WARN] TARGET_CLICK_KEY='{TARGET_CLICK_KEY}' 이므로 "
            "로그인 성공 창 검증을 건너뜁니다."
        )
        return 0

    main_window = _wait_for_post_login_windows(app)
    if main_window is None:
        return 6

    _scroll_to_reveal_more(main_window)

    return 0


if __name__ == "__main__":
    sys.exit(main())
