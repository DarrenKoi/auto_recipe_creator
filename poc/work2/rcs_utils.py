"""poc.work2 RCS 자동화 공통 유틸리티.

VLM 응답 파싱, 화면 캡처, 마우스 클릭, 디버그 이미지 생성, 창 탐색 등
여러 자동화 스크립트에서 공통으로 사용하는 함수를 모아둔다.
"""

from __future__ import annotations

import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

try:
    import mss
    import mss.tools

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pywinauto import Desktop, mouse

    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False


__all__ = [
    "capture_window",
    "click_at",
    "debug_image_path",
    "encode_image_webp",
    "extract_json",
    "find_existing_main_window",
    "is_main_window_title",
    "parse_coords",
    "save_marked_image",
    "scan_window_list",
]


# ─────────────────────────── VLM 응답 파싱 ───────────────────────────


def extract_json(text: str) -> dict:
    """VLM 응답 텍스트에서 JSON 객체를 추출한다."""
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        if e != -1:
            return json.loads(text[s:e].strip())
    if "{" in text:
        s = text.find("{")
        e = text.rfind("}")
        if e > s:
            return json.loads(text[s : e + 1])
    return json.loads(text)


def parse_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """VLM 응답 좌표를 정수로 변환하고 범위를 검증한다."""
    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"  [MISS] {key:20s} — VLM 응답에 없음")
            continue
        raw_x, raw_y = pt.get("x", 0), pt.get("y", 0)
        x, y = int(raw_x), int(raw_y)
        data[key] = {"x": x, "y": y}
        out = ""
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            out = " ← OUT OF BOUNDS"
        print(f"  [RAW ] {key:20s} — raw=({raw_x}, {raw_y}) → px=({x}, {y}){out}")
    return data


# ─────────────────────────── 화면 캡처 ───────────────────────────


def capture_window(window) -> "Image.Image":
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
    rect = window.rectangle()
    region = {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_data))
    print(f"[INFO] 창 캡처 완료: {image.size[0]}x{image.size[1]} px")
    return image


def encode_image_webp(
    image: "Image.Image", quality: int = 90
) -> tuple[str, int, int]:
    """PIL Image를 base64 WebP로 인코딩한다 (원본 해상도 유지)."""
    w, h = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="WEBP", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(
        f"[INFO] 이미지 인코딩: {w}x{h}, WebP q={quality}, "
        f"{len(buf.getvalue()) / 1024:.1f}KB"
    )
    return b64, w, h


# ─────────────────────────── 마우스 클릭 ───────────────────────────


def click_at(
    element_key: str,
    window,
    elements: dict,
    *,
    retry_count: int = 2,
    retry_delay_sec: float = 0.25,
) -> bool:
    """VLM 좌표를 스크린 좌표로 변환해 요소를 클릭한다."""
    pt = elements.get(element_key)
    if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
        print(f"[ERROR] 클릭 대상 '{element_key}' 좌표가 없습니다.")
        return False

    rect = window.rectangle()
    x = int(pt["x"]) + rect.left
    y = int(pt["y"]) + rect.top
    x = max(rect.left, min(x, rect.right - 1))
    y = max(rect.top, min(y, rect.bottom - 1))
    rel_x = max(0, min(int(pt["x"]), rect.right - rect.left - 1))
    rel_y = max(0, min(int(pt["y"]), rect.bottom - rect.top - 1))

    print(f"[INFO] '{element_key}' 클릭: screen=({x}, {y})")
    for attempt in range(1, max(1, retry_count) + 1):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] click_input 실패 (attempt={attempt}): {exc}")

        try:
            mouse.move(coords=(x, y))
            time.sleep(0.08)
            mouse.press(button="left", coords=(x, y))
            time.sleep(0.05)
            mouse.release(button="left", coords=(x, y))
            print(f"[INFO] mouse press/release 실행 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] mouse press/release 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, retry_delay_sec))

    return False


# ─────────────────────────── 디버그 이미지 ───────────────────────────


def debug_image_path(
    debug_dir: Path,
    filename: str,
    model_name: str | None = None,
) -> Path:
    """모델명 하위 디렉터리를 포함한 디버그 이미지 경로를 반환한다."""
    from poc.work2 import debug_image_path as resolve_debug_image_path

    return resolve_debug_image_path(
        debug_dir,
        filename,
        model_name=model_name,
    )


def save_marked_image(
    image: "Image.Image",
    elements: dict,
    colors: dict,
    out_path: Path,
) -> None:
    """좌표를 원본 스크린샷 위에 십자선+원으로 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    r = 12
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        x, y = int(pt["x"]), int(pt["y"])
        color = colors.get(name, "white")
        # 십자선
        draw.line([(x - r, y), (x + r, y)], fill=color, width=2)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=2)
        # 원
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=2)
        # 라벨: input/button 류는 아래에, 나머지는 위에 표시
        label = f"{name} ({x},{y})"
        if "input" in name or "button" in name:
            draw.text((x + r + 3, y + 4), label, fill=color, font=font)
        else:
            draw.text((x + r + 3, y - 16), label, fill=color, font=font)

    debug_dir = out_path.parent
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


# ─────────────────────────── 창 탐색 ───────────────────────────


def is_main_window_title(title: str, regex: str) -> bool:
    """메인 RCS 창 제목인지 정규식으로 판별한다."""
    try:
        return re.search(regex, title, flags=re.IGNORECASE) is not None
    except re.error:
        t = title.lower()
        return "rcs" in t and "[server" in t


def scan_window_list(
    windows,
    source_name: str,
    debug_rows: list[str],
    matcher: Callable[[str], bool],
) -> tuple[Any, str]:
    """창 목록을 순회하며 matcher 조건에 맞는 첫 번째 창을 반환한다."""
    for idx, win in enumerate(windows, start=1):
        try:
            title = win.window_text() or ""
        except Exception as exc:
            debug_rows.append(f"{source_name}[{idx}] title-read-error={exc}")
            continue

        matched = matcher(title)
        debug_rows.append(f"{source_name}[{idx}] matched={matched} title={title!r}")
        if matched:
            return win, title

    return None, ""


def find_existing_main_window(
    backends: tuple[str, ...],
    matcher: Callable[[str], bool],
) -> tuple[Any, str, list[str]]:
    """이미 떠 있는 메인 RCS 창을 데스크톱 전체에서 탐색한다."""
    debug_rows: list[str] = []
    for backend in backends:
        try:
            desktop_windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True
            )
        except Exception as exc:
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        main_window, main_title = scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows, matcher
        )
        if main_window is not None:
            return main_window, main_title, debug_rows

    return None, "", debug_rows
