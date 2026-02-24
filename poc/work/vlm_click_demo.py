"""
VLM 클릭 포인트 시각화 데모

화면 캡처 → VLM 분석 → 클릭 지점 마킹 → 결과 이미지 저장.
매니저 시연용: VLM이 화면의 어디를 클릭해야 하는지 시각적으로 보여줍니다.

듀얼 모니터 (1920x1080 x 2) 환경 대응:
  MONITOR_INDEX=1 → 주 모니터 캡처 (오프셋 0,0)
  MONITOR_INDEX=2 → 보조 모니터 캡처 (오프셋 1920,0)

좌표 변환 체인:
  VLM 좌표 (리사이즈 이미지, 예: 1280x720)
    ÷ resize_scale (0.6667)
  스크린샷 좌표 (캡처 픽셀, 1920x1080)  ← 이미지에 마킹
    ÷ dpi_scale (1.0 for 표준 1080p)
  모니터 로컬 좌표 (1920x1080)
    + monitor_offset (예: 1920,0 for 2번 모니터)
  마우스 절대 좌표  ← 실제 클릭 위치

Usage:
    python -m poc.work.vlm_click_demo

.env 설정:
    VLM_API_URL=http://...       # VLM API 엔드포인트
    VLM_API_KEY=...              # API 키
    VLM_MODEL_NAME=Qwen3-VL-30B-Instruct
    CLICK_TASK=Settings 버튼을 클릭하세요
    SAFE_MODE=true               # true면 실제 클릭 안 함
    MONITOR_INDEX=1              # 캡처할 모니터 (1=주, 2=보조)
    MAX_IMAGE_SIZE=1280
    USE_WEBP=true
"""

import os
import sys
import time
import json
import base64
import platform
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, List

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] Pillow 미설치. pip install Pillow")

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[WARNING] mss 미설치. pip install mss")

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# ─────────────────────────── DPI 스케일 감지 ───────────────────────────

def get_dpi_scale() -> float:
    """
    DPI 스케일 팩터 감지

    mss는 물리 픽셀로 캡처하고, pynput은 논리 좌표(pt)로 동작하므로
    이 비율을 알아야 정확한 클릭 위치를 계산할 수 있습니다.

    일반적 값:
      - Windows 1920x1080 (100% 배율): 1.0
      - macOS Retina: 2.0
      - macOS 외장 1080p 모니터: 1.0
    """
    system = platform.system()

    # Windows: 표준 1080p (100% 배율)에서는 1.0
    # DPI 배율(125%, 150%) 사용 시에는 별도 보정 필요
    if system == "Windows":
        try:
            import ctypes
            # DPI Awareness 설정 (mss가 물리 픽셀로 캡처하도록)
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        return 1.0

    # macOS
    if system == "Darwin":
        # 방법 1: AppKit.NSScreen.backingScaleFactor (가장 정확)
        try:
            import AppKit
            scale = AppKit.NSScreen.mainScreen().backingScaleFactor()
            return float(scale)
        except Exception:
            pass

        # 방법 2: Quartz 논리 해상도 vs mss 물리 해상도 비교
        try:
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            logical_w = Quartz.CGDisplayPixelsWide(main_display)
            if MSS_AVAILABLE:
                with mss.mss() as sct:
                    physical_w = sct.monitors[1]["width"]
                return physical_w / logical_w
        except Exception:
            pass

        print("[WARNING] DPI 스케일 자동 감지 불가, 기본값 1.0 사용")

    return 1.0


# ─────────────────────────── 모니터 정보 ───────────────────────────

def print_monitor_info() -> list:
    """
    연결된 모니터 목록을 출력합니다.

    듀얼 모니터 구성 예시 (1920x1080 x 2):
      Monitor 0: 가상 전체 화면 (3840x1080) — 캡처하지 않음
      Monitor 1: 주 모니터 (0, 0) 1920x1080
      Monitor 2: 보조 모니터 (1920, 0) 1920x1080

    Returns:
        mss monitors 리스트
    """
    if not MSS_AVAILABLE:
        print("[ERROR] mss 라이브러리 필요")
        return []

    with mss.mss() as sct:
        monitors = list(sct.monitors)

    print(f"[INFO] 감지된 모니터: {len(monitors) - 1}대")
    for i, mon in enumerate(monitors):
        if i == 0:
            label = "가상 전체"
        else:
            label = f"모니터 {i}"
        print(
            f"  [{i}] {label}: "
            f"({mon['left']}, {mon['top']}) "
            f"{mon['width']}x{mon['height']}"
        )

    return monitors


# ─────────────────────────── 화면 캡처 ───────────────────────────

def capture_screen(
    monitor_index: int = 1,
) -> Tuple[Optional[Image.Image], int, int, int, int]:
    """
    특정 모니터 화면 캡처 (물리 픽셀 해상도)

    듀얼 모니터에서 monitor_index로 캡처 대상 선택:
      1 = 주 모니터, 2 = 보조 모니터

    Returns:
        (PIL Image, width, height, offset_x, offset_y)
        offset_x/y: 가상 스크린 내 모니터 시작 좌표 (마우스 절대 좌표 계산용)
        실패 시 (None, 0, 0, 0, 0)
    """
    if not MSS_AVAILABLE or not PIL_AVAILABLE:
        print("[ERROR] mss 또는 Pillow 라이브러리 필요")
        return None, 0, 0, 0, 0

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            print(f"[ERROR] 모니터 인덱스 {monitor_index} 초과 (최대: {len(monitors)-1})")
            return None, 0, 0, 0, 0

        mon = monitors[monitor_index]
        offset_x = mon["left"]
        offset_y = mon["top"]

        start = time.time()
        screenshot = sct.grab(mon)
        elapsed_ms = (time.time() - start) * 1000
        png_data = mss.tools.to_png(screenshot.rgb, screenshot.size)

    image = Image.open(BytesIO(png_data))
    w, h = image.size
    print(
        f"[INFO] 모니터 {monitor_index} 캡처: {w}x{h} px, "
        f"오프셋 ({offset_x}, {offset_y}), {elapsed_ms:.1f}ms"
    )
    return image, w, h, offset_x, offset_y


# ─────────────────────────── 이미지 리사이즈 ───────────────────────────

def resize_for_vlm(image: Image.Image, max_size: int = 1280) -> Tuple[Image.Image, float]:
    """
    VLM 입력용 리사이즈

    Returns:
        (resized_image, scale_factor)
        scale_factor = resized / original (< 1.0 이면 축소됨)
    """
    w, h = image.size
    max_dim = max(w, h)

    if max_dim <= max_size:
        print(f"[INFO] 리사이즈 불필요: {w}x{h} (max_size={max_size})")
        return image.copy(), 1.0

    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    print(f"[INFO] VLM용 리사이즈: {w}x{h} → {new_w}x{new_h} (scale={scale:.4f})")
    return resized, scale


# ─────────────────────────── VLM API 호출 ───────────────────────────

def ask_vlm_click_point(
    image: Image.Image,
    task: str,
    api_url: str,
    api_key: str = "",
    model_name: str = "Qwen3-VL-30B-Instruct",
    use_webp: bool = True,
) -> Optional[dict]:
    """
    VLM에 클릭 지점 질의

    Returns:
        {
            "reasoning": str,
            "target_name": str,
            "bbox": [x1, y1, x2, y2],
            "click_point": [cx, cy],
            "confidence": float
        }
        또는 None
    """
    w, h = image.size

    # 이미지 → bytes
    buffer = BytesIO()
    if use_webp:
        image.save(buffer, format="WEBP", quality=85, method=4)
        img_format = "webp"
    else:
        image.save(buffer, format="PNG", optimize=True)
        img_format = "png"

    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_size_kb = len(buffer.getvalue()) / 1024
    print(f"[INFO] VLM 전송 이미지: {w}x{h}, {img_format}, {img_size_kb:.1f}KB")

    prompt = f"""당신은 GUI 화면 분석 전문가입니다.

현재 화면에서 다음 작업을 수행하려면 어디를 클릭해야 하는지 분석하세요:
「{task}」

이미지 해상도: {w}x{h} 픽셀

반드시 다음 JSON 형식으로만 응답하세요:
{{
    "reasoning": "화면 분석 및 클릭 지점 선택 이유",
    "target_name": "클릭 대상 UI 요소 이름",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.0
}}

bbox는 클릭 대상의 바운딩 박스 [x1, y1, x2, y2]를 픽셀 좌표로 반환하세요.
좌표 범위: x는 0~{w}, y는 0~{h}."""

    vlm_client = LangChainOpenAICompatibleVLMClient(
        base_url=api_url,
        api_key=api_key,
        timeout_sec=60.0,
    )
    request = ChatImageRequest(
        model=model_name,
        system_message=(
            "당신은 GUI 자동화 에이전트입니다. "
            f"이 이미지의 해상도는 {w}x{h} 픽셀입니다. "
            f"좌표는 반드시 0~{w}(x), 0~{h}(y) 범위의 픽셀 값으로 반환하세요. "
            "반드시 JSON 형식으로만 응답하세요."
        ),
        user_text=prompt,
        image_b64=img_base64,
        image_mime=f"image/{img_format}",
        temperature=0.1,
    )

    try:
        print(f"[INFO] VLM API 호출 중... ({api_url})")
        start = time.time()

        response_text = vlm_client.chat_with_image(request)
        elapsed_ms = (time.time() - start) * 1000

        print(f"[INFO] VLM 응답 수신 ({elapsed_ms:.0f}ms)")

        parsed = _parse_vlm_json(response_text, w, h)
        if parsed:
            bbox = parsed.get("bbox", [])
            if len(bbox) == 4:
                parsed["click_point"] = [
                    (bbox[0] + bbox[2]) // 2,
                    (bbox[1] + bbox[3]) // 2,
                ]
            return parsed

        return None

    except Exception as e:
        print(f"[ERROR] VLM API 호출 실패: {e}")
        return None


def _parse_vlm_json(response_text: str, screen_w: int, screen_h: int) -> Optional[dict]:
    """VLM 응답에서 JSON 파싱 + 정규화 좌표 변환"""
    try:
        json_str = response_text
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_str = response_text[start:end].strip()
        elif "{" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if end > start:
                json_str = response_text[start : end + 1]

        data = json.loads(json_str)

        # 정규화 좌표 감지 (모든 값이 0~1 범위 → 픽셀 좌표로 변환)
        bbox = data.get("bbox", [])
        if bbox and all(isinstance(v, (int, float)) and 0 <= v <= 1.0 for v in bbox):
            data["bbox"] = [
                int(bbox[0] * screen_w),
                int(bbox[1] * screen_h),
                int(bbox[2] * screen_w),
                int(bbox[3] * screen_h),
            ]
            print("[INFO] VLM이 정규화 좌표(0~1) 반환 → 픽셀 좌표로 변환 완료")

        # 화면 경계 클램핑
        if data.get("bbox") and len(data["bbox"]) == 4:
            data["bbox"][0] = max(0, min(data["bbox"][0], screen_w))
            data["bbox"][1] = max(0, min(data["bbox"][1], screen_h))
            data["bbox"][2] = max(0, min(data["bbox"][2], screen_w))
            data["bbox"][3] = max(0, min(data["bbox"][3], screen_h))

        return data

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[ERROR] VLM 응답 파싱 실패: {e}")
        print(f"[DEBUG] 응답: {response_text[:500]}")
        return None


# ─────────────────────────── 클릭 마커 그리기 ───────────────────────────

def draw_click_marker(
    image: Image.Image,
    x: int,
    y: int,
    bbox: Optional[List[int]] = None,
    label: str = "",
    confidence: float = 0.0,
) -> Image.Image:
    """
    스크린샷에 클릭 지점 마커 표시

    - 빨간 크로스헤어 + 원 (클릭 지점)
    - 초록 사각형 (bbox 영역)
    - 라벨 텍스트 (대상 이름 + 신뢰도)

    Args:
        image: 원본 스크린샷 (물리 픽셀 좌표 기준)
        x, y: 클릭 지점 (물리 픽셀)
        bbox: 바운딩 박스 [x1,y1,x2,y2] 물리 픽셀 (없으면 None)
        label: 대상 이름
        confidence: 신뢰도

    Returns:
        마커가 표시된 이미지 사본
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    img_w, img_h = annotated.size
    marker_radius = max(20, min(img_w, img_h) // 40)
    crosshair_len = marker_radius * 2
    line_width = max(3, marker_radius // 6)

    color_red = (255, 50, 50)
    color_green = (50, 200, 50)
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)

    # 1. bbox 사각형 (초록)
    if bbox and len(bbox) == 4:
        draw.rectangle(bbox, outline=color_green, width=line_width)

    # 2. 크로스헤어 (빨강)
    draw.line(
        [(x - crosshair_len, y), (x + crosshair_len, y)],
        fill=color_red,
        width=line_width,
    )
    draw.line(
        [(x, y - crosshair_len), (x, y + crosshair_len)],
        fill=color_red,
        width=line_width,
    )

    # 3. 원 (빨강)
    draw.ellipse(
        [
            (x - marker_radius, y - marker_radius),
            (x + marker_radius, y + marker_radius),
        ],
        outline=color_red,
        width=line_width,
    )

    # 4. 라벨 텍스트
    if label or confidence > 0:
        font_size = max(20, marker_radius)
        font = _load_font(font_size)

        label_text = label if label else ""
        if confidence > 0:
            label_text += f" ({confidence:.0%})"
        label_text = label_text.strip()

        if label_text:
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # 라벨 위치: 클릭 지점 위쪽 (화면 밖이면 아래로)
            padding = 6
            label_x = x - text_w // 2
            label_y = y - marker_radius - text_h - padding * 2 - 8

            if label_x < padding:
                label_x = padding
            if label_x + text_w + padding > img_w:
                label_x = img_w - text_w - padding * 2
            if label_y < padding:
                label_y = y + marker_radius + 10

            # 배경 사각형
            draw.rectangle(
                [
                    label_x - padding,
                    label_y - padding,
                    label_x + text_w + padding,
                    label_y + text_h + padding,
                ],
                fill=color_black,
                outline=color_red,
                width=2,
            )
            draw.text((label_x, label_y), label_text, fill=color_white, font=font)

    # 5. 좌표 텍스트 (작게, 클릭 지점 옆)
    coord_font = _load_font(max(14, marker_radius // 2))
    coord_text = f"({x}, {y})"
    draw.text(
        (x + marker_radius + 5, y - 8),
        coord_text,
        fill=color_red,
        font=coord_font,
    )

    return annotated


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """시스템 폰트 로드 (실패 시 기본 폰트)"""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",       # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc", # macOS 한글
        "C:\\Windows\\Fonts\\arial.ttf",              # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ─────────────────────────── 좌표 변환 정보 ───────────────────────────

def print_coordinate_info(
    vlm_point: Tuple[int, int],
    vlm_image_size: Tuple[int, int],
    screenshot_size: Tuple[int, int],
    resize_scale: float,
    dpi_scale: float,
    monitor_offset: Tuple[int, int] = (0, 0),
) -> Tuple[int, int, int, int]:
    """
    좌표 변환 체인을 상세히 출력

    듀얼 모니터에서 monitor_offset은 캡처한 모니터의 가상 스크린 내 시작 좌표입니다.
    예: 모니터 2가 (1920, 0) 위치 → 마우스 좌표에 +1920 보정

    Returns:
        (screenshot_x, screenshot_y, mouse_abs_x, mouse_abs_y)
    """
    vlm_x, vlm_y = vlm_point
    off_x, off_y = monitor_offset

    # VLM → 스크린샷 물리 픽셀
    phys_x = int(vlm_x / resize_scale)
    phys_y = int(vlm_y / resize_scale)

    # 물리 픽셀 → 모니터 내 논리 좌표
    local_x = int(phys_x / dpi_scale)
    local_y = int(phys_y / dpi_scale)

    # + 모니터 오프셋 → 절대 마우스 좌표
    mouse_x = local_x + off_x
    mouse_y = local_y + off_y

    logical_w = int(screenshot_size[0] / dpi_scale)
    logical_h = int(screenshot_size[1] / dpi_scale)

    print()
    print("=" * 60)
    print("  좌표 변환 상세")
    print("=" * 60)
    print(f"  VLM 이미지:         {vlm_image_size[0]}x{vlm_image_size[1]} px")
    print(f"  스크린샷 (캡처):    {screenshot_size[0]}x{screenshot_size[1]} px")
    print(f"  모니터 해상도:      {logical_w}x{logical_h}")
    print(f"  모니터 오프셋:      ({off_x}, {off_y})")
    print(f"  리사이즈 비율:      {resize_scale:.4f}")
    print(f"  DPI 스케일:         {dpi_scale:.1f}x")
    print()
    print(f"  VLM 클릭 좌표:      ({vlm_x}, {vlm_y})")
    print(f"    ÷ resize {resize_scale:.4f}")
    print(f"  스크린샷 좌표:      ({phys_x}, {phys_y})  ← 이미지에 마킹")
    print(f"    ÷ DPI {dpi_scale:.1f}")
    print(f"  모니터 로컬 좌표:   ({local_x}, {local_y})")
    if off_x != 0 or off_y != 0:
        print(f"    + offset ({off_x}, {off_y})")
    print(f"  마우스 절대 좌표:   ({mouse_x}, {mouse_y})  ← 실제 클릭 위치")
    print("=" * 60)

    return phys_x, phys_y, mouse_x, mouse_y


# ─────────────────────────── 메인 ───────────────────────────

def main():
    """데모 실행"""
    # .env 로드
    if DOTENV_AVAILABLE:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()

    # 설정
    api_url = os.environ.get("VLM_API_URL", "")
    api_key = os.environ.get("VLM_API_KEY", "")
    model_name = os.environ.get("VLM_MODEL_NAME", "Qwen3-VL-30B-Instruct")
    max_image_size = int(os.environ.get("MAX_IMAGE_SIZE", "1280"))
    use_webp = os.environ.get("USE_WEBP", "true").lower() in ("true", "1", "yes")
    safe_mode = os.environ.get("SAFE_MODE", "true").lower() in ("true", "1", "yes")
    click_task = os.environ.get(
        "CLICK_TASK", "이 화면에서 가장 눈에 띄는 클릭 가능한 버튼을 찾아주세요"
    )
    output_dir = os.environ.get("OUTPUT_DIR", "./captures")
    monitor_index = int(os.environ.get("MONITOR_INDEX", "1"))
    output_dir_path = Path(output_dir).expanduser()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_dir = output_dir_path

    print()
    print("+" + "=" * 58 + "+")
    print("|   VLM Click Point Visualization Demo                     |")
    print("+" + "=" * 58 + "+")
    print(f"  Task:       {click_task}")
    print(f"  Model:      {model_name}")
    print(f"  Safe mode:  {safe_mode}")
    print(f"  Monitor:    {monitor_index}")
    print(f"  API URL:    {api_url or '(미설정 → Mock)'}")
    print()

    # ── Step 1: 모니터 감지 + DPI 스케일 ──
    print("[Step 1] 모니터 감지")
    monitors = print_monitor_info()

    dpi_scale = get_dpi_scale()
    print(f"[INFO] DPI 스케일: {dpi_scale:.1f}x", end="")
    if dpi_scale > 1.0:
        print(" (HiDPI 디스플레이)")
    else:
        print(" (표준 디스플레이)")

    # ── Step 2: 화면 캡처 ──
    print()
    print(f"[Step 2] 모니터 {monitor_index} 화면 캡처")
    screenshot, phys_w, phys_h, off_x, off_y = capture_screen(monitor_index)
    if screenshot is None:
        print("[ERROR] 화면 캡처 실패")
        sys.exit(1)

    logical_w = int(phys_w / dpi_scale)
    logical_h = int(phys_h / dpi_scale)
    print(f"[INFO] 모니터 해상도: {logical_w}x{logical_h}")
    if off_x != 0 or off_y != 0:
        print(f"[INFO] 모니터 오프셋: ({off_x}, {off_y}) — 듀얼 모니터 보정 적용")

    # 원본 스크린샷 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_path = output_dir / f"screenshot_{timestamp}.png"
    screenshot.save(orig_path, format="PNG")
    print(f"[INFO] 원본 스크린샷 저장: {orig_path}")

    # ── Step 3: VLM용 리사이즈 ──
    print()
    print("[Step 3] VLM용 이미지 준비")
    vlm_image, resize_scale = resize_for_vlm(screenshot, max_image_size)
    vlm_w, vlm_h = vlm_image.size

    # ── Step 4: VLM 분석 ──
    print()
    print("[Step 4] VLM 분석")
    print(f"[INFO] 작업: \"{click_task}\"")

    if not api_url:
        print("[WARNING] VLM_API_URL 미설정 → Mock 응답 사용")
        vlm_result = {
            "reasoning": "Mock: 화면 중앙 부근의 버튼을 대상으로 선택합니다.",
            "target_name": "중앙 버튼 (Mock)",
            "bbox": [
                vlm_w // 2 - 50,
                vlm_h // 2 - 20,
                vlm_w // 2 + 50,
                vlm_h // 2 + 20,
            ],
            "confidence": 0.85,
            "click_point": [vlm_w // 2, vlm_h // 2],
        }
    else:
        vlm_result = ask_vlm_click_point(
            vlm_image, click_task, api_url, api_key, model_name, use_webp
        )

    if vlm_result is None:
        print("[ERROR] VLM 분석 실패")
        sys.exit(1)

    print()
    print(f"  추론:   {vlm_result.get('reasoning', 'N/A')}")
    print(f"  대상:   {vlm_result.get('target_name', 'N/A')}")
    print(f"  bbox:   {vlm_result.get('bbox', 'N/A')}")
    print(f"  신뢰도: {vlm_result.get('confidence', 0):.0%}")

    # ── Step 5: 좌표 변환 ──
    click_point = vlm_result.get("click_point")
    bbox = vlm_result.get("bbox", [])

    if not click_point and bbox and len(bbox) == 4:
        click_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    if not click_point:
        print("[ERROR] 클릭 지점을 결정할 수 없음")
        sys.exit(1)

    vlm_x, vlm_y = click_point

    phys_x, phys_y, mouse_x, mouse_y = print_coordinate_info(
        (vlm_x, vlm_y),
        (vlm_w, vlm_h),
        (phys_w, phys_h),
        resize_scale,
        dpi_scale,
        monitor_offset=(off_x, off_y),
    )

    # ── Step 6: 스크린샷에 마킹 ──
    print()
    print("[Step 6] 스크린샷에 클릭 지점 마킹")

    # bbox를 물리 픽셀로 변환
    phys_bbox = None
    if bbox and len(bbox) == 4:
        phys_bbox = [
            int(bbox[0] / resize_scale),
            int(bbox[1] / resize_scale),
            int(bbox[2] / resize_scale),
            int(bbox[3] / resize_scale),
        ]

    annotated = draw_click_marker(
        screenshot,
        phys_x,
        phys_y,
        bbox=phys_bbox,
        label=vlm_result.get("target_name", ""),
        confidence=vlm_result.get("confidence", 0),
    )

    # 결과 저장
    result_path = output_dir / f"vlm_click_{timestamp}.png"
    annotated.save(result_path, format="PNG")
    print(f"[INFO] 결과 이미지 저장: {result_path}")

    # 좌표 매핑 JSON 저장
    coord_info = {
        "task": click_task,
        "vlm_result": vlm_result,
        "coordinate_mapping": {
            "vlm_image_size": [vlm_w, vlm_h],
            "screenshot_physical_size": [phys_w, phys_h],
            "monitor_resolution": [logical_w, logical_h],
            "monitor_offset": [off_x, off_y],
            "monitor_index": monitor_index,
            "resize_scale": resize_scale,
            "dpi_scale": dpi_scale,
            "vlm_click_point": [vlm_x, vlm_y],
            "screenshot_click_point": [phys_x, phys_y],
            "mouse_absolute_point": [mouse_x, mouse_y],
        },
        "safe_mode": safe_mode,
        "timestamp": timestamp,
    }

    json_path = output_dir / f"vlm_click_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(coord_info, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 좌표 정보 저장: {json_path}")

    # 이미지 미리보기
    if platform.system() == "Darwin":
        print()
        print("[INFO] 결과 이미지를 미리보기로 엽니다...")
        os.system(f'open "{result_path}"')
    elif platform.system() == "Windows":
        print()
        print("[INFO] 결과 이미지를 엽니다...")
        os.system(f'start "" "{result_path}"')

    # ── Step 7: 실제 클릭 (옵션) ──
    print()
    if safe_mode:
        print(f"[SAFE MODE] 실제 클릭 생략 — 마우스 절대 좌표: ({mouse_x}, {mouse_y})")
    else:
        print(f"[LIVE] 마우스 클릭 실행: ({mouse_x}, {mouse_y})")
        try:
            from pynput.mouse import Button, Controller
            mouse = Controller()
            mouse.position = (mouse_x, mouse_y)
            time.sleep(0.3)
            mouse.click(Button.left)
            print("[INFO] 클릭 실행 완료")
        except ImportError:
            print("[ERROR] pynput 미설치, 클릭 생략")

    print()
    print("데모 완료!")
    return coord_info


if __name__ == "__main__":
    main()
