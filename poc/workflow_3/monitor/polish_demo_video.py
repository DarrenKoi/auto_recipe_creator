"""시연 영상 마무리 - 녹화 조각(clip)과 설명 카드(card)를 이어 PPT 용 mp4 한 편으로 만든다.

입력은 `demo_record_rcs.py` 가 남긴 폴더(`_demo/rcs_<tag>/`: raw.mp4 + events.json +
stages.json + subtitles.json)다. 아래 SEQUENCE 순서대로 카드와 clip 을 이어 붙이며 clip 마다:

  * 커서를 그린다 - 화면 캡처에는 마우스 커서가 없다(녹화 때 기록한 위치로 그린다).
  * 클릭 지점에 퍼지는 원(click ring)을 그리고, 클릭 부근을 부드럽게 확대한다. 클릭마다
    확대하면 정신없으므로 **샷** 으로 묶는다: SHOT_GAP_SEC 안에 이어지고 확대 화면 한 장에
    다 들어오는 클릭들(로그인 입력칸들)은 한 번 확대해 그 샷이 끝날 때까지 움직이지 않는다.
  * 화면이 IDLE_MAX_SEC 넘게 멈춘 구간(VLM 판독 대기 등)은 잘라낸다. 자막이 떠 있거나
    카메라가 움직이는 중에는 자르지 않는다.
  * 자막은 하단 반투명 바에 페이드로, 카드는 어두운 배경의 제목+본문으로 넣는다(한글,
    맑은 고딕). clip/카드 경계는 검은색 페이드.
  * BLUR_REGIONS 로 직원 이름(List 탭 Connection User 열) 같은 영역을 가린다.
  * notes.json(순찰 판독 결과)이 있으면 판독한 영역에 강조 박스를, 우상단에 "AI 판독" 패널을
    NOTE_SEC 동안 띄운다. 판독 영역이 확대 화면에 들어가면 카메라도 그쪽으로 간다.

출력은 H.264(yuv420p, faststart) 1920x1080 30fps mp4 - PowerPoint 에 그대로 삽입된다.
오프라인 전용(Mac/오피스 어디서나). 오래 걸리면 먼저 PREVIEW_WIDTH 로 빠르게 확인한다.

사용법:
  1) 아래 SEQUENCE 를 채운다 (clip 은 _demo 아래 폴더 이름 앞부분 또는 절대 경로).
  2) uv run python poc/workflow_3/monitor/polish_demo_video.py
"""

import json
import math
import sys
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor.demo_log_panel import resolve_font  # noqa: E402
from poc.workflow_3.monitor.screen_video import (  # noqa: E402
    DEMO_ROOT,
    EVENTS_NAME,
    NOTES_NAME,
    STAGES_NAME,
    SUBTITLES_NAME,
    VIDEO_NAME,
)

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다.
#
# SEQUENCE 항목 두 종류:
#   카드: {"card": "제목", "body": "본문 (\n 줄바꿈)", "sec": 4.0}
#   clip: {"clip": "rcs_"}                      _demo 아래 이름 앞부분(가장 최근) 또는 경로
#         + "stage": "view_tab" 또는 ["login", "view_tab"]   stages.json 의 그 단계만 자른다
#         + "detail": "MCD019"                  stage 가 visit/in_tool 일 때 장비 지정
#         + "start"/"end": 초                   직접 자르기(stage 보다 우선, 음수=끝에서부터)
#         + "subtitles": [(시작, 끝, "문구")]   원본 영상 초 기준. 주면 subtitles.json 대신 쓴다
#         + "crop": (x0, y0, x1, y1)            화면 비율 0~1 - tool 창만 크게 보이게
#         + "zoom": 1.0                         이 clip 만 확대 배율(1.0 = 확대 끔)
#         + "blur": [(x0, y0, x1, y1), ...]     이 clip 만 가림 영역(BLUR_REGIONS 에 더함)
#
# clip 폴더 두 종류:
#   rcs_<tag>            demo_record_rcs.py   stage = login/view_tab/list_tab/visit/in_tool
#   alarm_<EQP>_<tag>    demo_record_alarm.py stage = alarm/connect_tool/.../correction/teardown
#     보정이 실패하면 teardown 에 엔지니어 수동 작업이 녹화된다 - 그때는 stage 를 correction 까지만
#     자르고, "모든 입력은 Agent" 카드가 그 구간에 걸리지 않게 한다.
# List 탭 Connection User 열에는 직원 이름이 보인다 - "blur" 로 가린다(판독 패널은 이름을 안 싣는다).
# ===========================================================================

SEQUENCE = [
    {"card": "엔지니어 개입 없는 AI 기반 자동화",
     "body": "이 영상의 모든 키보드·마우스 입력은 엔지니어가 아니라\nAgent가 직접 수행합니다", "sec": 4.0},
    {"card": "Align Fail 자동 대응",
     "body": "R3 CD-SEM Align Fail에 대한 24/7 현업 대응을 AI Agent가 대신 처리\n"
             "처리 불가능한 경우에만 엔지니어에게 알람 후 인계하는 시스템", "sec": 5.0},
    {"clip": "rcs_", "stage": ["login", "view_tab"]},
    {"card": "AI 순찰", "body": "화면을 읽고 판단해 장비를 오갑니다", "sec": 4.0},
    {"clip": "rcs_", "stage": "visit"},
    {"card": "Align Fail 알람 대응", "body": "알람 발생부터 보정까지 사람 없이 진행됩니다", "sec": 4.0},
    {"clip": "alarm_", "stage": ["alarm", "correction"]},
]
OUTPUT = ""               # 비우면 _demo/final_<시각>.mp4
OUT_SIZE = (1920, 1080)   # PPT 16:9
FPS = 30
CRF = 18                  # 낮을수록 고화질(파일 큼). 18 = 눈으로 구분 안 되는 수준
PREVIEW_WIDTH = 0         # >0 이면 그 폭으로 빠르게 미리보기(예: 960)
FADE_SEC = 0.5            # clip/카드 시작·끝 페이드
STAGE_PAD_SEC = 0.8       # stage 로 자를 때 앞뒤 여유
IDLE_MAX_SEC = 1.0        # 화면이 이보다 오래 멈추면 나머지를 잘라낸다. 0 = 안 자름
CURSOR = 1                # 커서 그리기
CLICK_RING = 1            # 클릭 강조 원
ZOOM = 1.3                # 클릭 지점 확대 배율. 1.0 = 끔
ZOOM_LEAD_SEC = 0.6       # 클릭 전 미리 다가가는 시간
ZOOM_HOLD_SEC = 1.5       # 샷의 마지막 클릭 뒤 머무는 시간
SHOT_GAP_SEC = 10.0       # 이 간격 안에 이어지는 클릭은 한 샷(확대 한 번)으로 묶는다
SHOT_MARGIN = 0.12        # 묶인 클릭들이 확대 화면 가장자리에서 떨어질 여백(화면 비율)
CAMERA_TAU_SEC = 0.35     # 카메라가 따라가는 속도(작을수록 빠름)
BLUR_REGIONS = []         # 전 clip 공통 가림 영역 [(x0, y0, x1, y1), ...] 화면 비율 0~1
NOTE_SEC = 3.5            # "AI 판독" 패널/강조 박스를 띄우는 시간
NOTE_FOCUS = 1            # 판독 영역으로 카메라 확대(영역이 확대 화면에 들어갈 때만)
NOTE_BOX_CHANGE = 18.0    # 판독 영역 화면이 이만큼(평균 밝기차) 바뀌면 강조 박스를 거둔다
NOTE_WRAP = 24            # 패널 내용 한 줄 최대 글자 수
NOTE_IMAGE_WIDTH = 0.24   # 근거 정지화면(보정 매칭 overlay) 폭, 출력 폭 대비

ACCENT = (66, 133, 244)   # 클릭 원/카드 강조선 (RGB)
SUBTITLE_FADE_SEC = 0.3
RING_SEC = 0.6
_BOLD_FONTS = ("C:/Windows/Fonts/malgunbd.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc")


# ------------------------------------------------------------------
# 시간/카메라 (순수 함수 - test_polish_demo_video.py).
# ------------------------------------------------------------------


def ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * p)


def fade_level(t: float, start: float, end: float, fade: float) -> float:
    """[start, end] 구간 양끝 fade 초 동안 0->1->0."""
    if fade <= 0:
        return 1.0
    return max(0.0, min(1.0, (t - start) / fade, (end - t) / fade))


def stage_range(stages: list, names, detail: str = "") -> tuple[float, float] | None:
    """stages.json 에서 names 단계(들)의 첫 시작~마지막 끝. 없으면 None."""
    names = [names] if isinstance(names, str) else list(names)
    hits = [s for s in stages if s["stage"] in names and (not detail or s.get("detail") == detail)]
    if not hits:
        return None
    return min(s["start"] for s in hits), max(s["end"] for s in hits)


def clip_range(item: dict, stages: list, duration: float, pad: float) -> tuple[float, float]:
    """clip 항목의 [start, end] (원본 초). start/end > stage > 전체 순."""
    start, end = 0.0, duration
    if "stage" in item and not ("start" in item and "end" in item):
        found = stage_range(stages, item["stage"], item.get("detail", ""))
        if found is None:
            raise ValueError(f"stages.json 에 {item['stage']!r} 단계가 없습니다: {item}")
        start, end = found[0] - pad, found[1] + pad
    if "start" in item:
        start = item["start"] if item["start"] >= 0 else duration + item["start"]
    if "end" in item:
        end = item["end"] if item["end"] > 0 else duration + item["end"]
    return max(0.0, start), min(duration, end)


def plan_shots(clicks: list, base: tuple, zoom: float, *, gap: float, margin: float,
               lead: float, hold: float) -> list[tuple]:
    """클릭들을 확대 샷 (시작, 끝, 중심 x, 중심 y) 으로 묶는다.

    다음 클릭이 직전 클릭에서 gap 초 안이고, 샷의 클릭 전부가 확대 화면(base / zoom)에
    여백 margin 을 두고 들어가면 같은 샷이다. 샷 중심 = 클릭들 bbox 중심.
    """
    if zoom <= 1.0 or not clicks:
        return []
    view_w, view_h = base[2] / zoom * (1 - 2 * margin), base[3] / zoom * (1 - 2 * margin)

    def fits(group):
        xs, ys = [c[1] for c in group], [c[2] for c in group]
        return max(xs) - min(xs) <= view_w and max(ys) - min(ys) <= view_h

    groups = []
    for click in sorted(clicks):
        if groups and click[0] - groups[-1][-1][0] <= gap and fits(groups[-1] + [click]):
            groups[-1].append(click)
        else:
            groups.append([click])
    shots = []
    for group in groups:
        xs, ys = [c[1] for c in group], [c[2] for c in group]
        shots.append((group[0][0] - lead, group[-1][0] + hold,
                      (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
    return shots


def camera_target(t: float, shots: list, home: tuple, zoom: float) -> tuple[float, float, float]:
    """(배율, 중심 x, 중심 y). t 에 걸린 샷(겹치면 나중 샷)으로, 아니면 home."""
    active = [shot for shot in shots if shot[0] <= t <= shot[1]]
    if not active:
        return (1.0, home[0], home[1])
    _, _, x, y = max(active)
    return (zoom, x, y)


def note_focus_points(notes: list, base: tuple, zoom: float, margin: float) -> list[tuple]:
    """note 의 박스 합집합 중심을 샷 계획용 (t, x, y) 로. 확대 화면에 안 들어가면 뺀다.

    점유 판독은 MC ID(왼쪽 끝)와 Connection User(오른쪽 끝) 두 셀이라 합집합이 화면 폭에
    가깝다 - 그 중심으로 확대하면 정작 읽은 두 셀이 잘린다.
    """
    if zoom <= 1.0:
        return []
    view_w, view_h = base[2] / zoom * (1 - 2 * margin), base[3] / zoom * (1 - 2 * margin)
    points = []
    for note in notes:
        rects = note["rects"]
        if not rects:
            continue
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
        if x1 - x0 <= view_w and y1 - y0 <= view_h:
            points.append((note["t"], (x0 + x1) / 2, (y0 + y1) / 2))
    return points


def camera_goal(t: float, priority: list, shots: list, home: tuple, zoom: float) -> tuple:
    """판독 초점 샷(priority)이 걸려 있으면 그것, 아니면 클릭 샷. 둘을 한 샷으로 묶으면
    판독 박스가 클릭 쪽으로 끌려가 잘린다(Codex 리뷰)."""
    live = [shot for shot in priority if shot[0] <= t <= shot[1]]
    return camera_target(t, live if live else shots, home, zoom)


def active_note(t: float, notes: list, sec: float):
    """t 에 떠 있는 note(겹치면 나중 것)와 그 fade level. 없으면 (None, 0)."""
    live = [n for n in notes if n["t"] <= t < n["t"] + sec]
    if not live:
        return None, 0.0
    note = max(live, key=lambda n: n["t"])
    return note, fade_level(t, note["t"], note["t"] + sec, SUBTITLE_FADE_SEC)


def step_camera(cam: tuple, target: tuple, dt: float, tau: float) -> tuple:
    alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
    return tuple(c + (g - c) * alpha for c, g in zip(cam, target))


def base_rect(crop, frame_w: int, frame_h: int, aspect: float) -> tuple:
    """crop(비율) 을 출력 종횡비로 넓힌 픽셀 rect (x0, y0, w, h). 프레임 밖으로는 안 나간다."""
    x0, y0, x1, y1 = crop or (0.0, 0.0, 1.0, 1.0)
    x0, x1, y0, y1 = x0 * frame_w, x1 * frame_w, y0 * frame_h, y1 * frame_h
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = min(frame_w, h * aspect)
    else:
        h = min(frame_h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(cx - w / 2, 0.0), frame_w - w)
    y0 = min(max(cy - h / 2, 0.0), frame_h - h)
    return x0, y0, w, h


def view_rect(cam: tuple, base: tuple) -> tuple:
    """카메라(배율, 중심)가 보는 rect. base 안에서만 움직인다."""
    z, cx, cy = cam
    bx, by, bw, bh = base
    w, h = bw / z, bh / z
    x0 = min(max(cx - w / 2, bx), bx + bw - w)
    y0 = min(max(cy - h / 2, by), by + bh - h)
    return x0, y0, w, h


# ------------------------------------------------------------------
# 그리기.
# ------------------------------------------------------------------


@lru_cache(maxsize=16)
def _font(size: int, bold: bool):
    for path in (_BOLD_FONTS if bold else ()):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    return resolve_font(size)[0]


def _cursor_polygon(x: float, y: float, unit: float) -> np.ndarray:
    shape = [(0, 0), (0, 17), (4.5, 13), (7.5, 20), (10.5, 18.7), (7.5, 12), (13, 12)]
    return np.array([(x + px * unit, y + py * unit) for px, py in shape], np.int32)


def draw_cursor(frame: np.ndarray, x: float, y: float, unit: float) -> None:
    poly = _cursor_polygon(x, y, unit)
    cv2.fillPoly(frame, [poly], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(frame, [poly], True, (20, 20, 20), max(1, round(unit)), cv2.LINE_AA)


def draw_click_rings(frame: np.ndarray, t: float, clicks: list, unit: float) -> None:
    for tc, x, y in clicks:
        p = (t - tc) / RING_SEC
        if not 0.0 <= p <= 1.0:
            continue
        radius = int((6 + 26 * ease(p)) * unit)
        pad = radius + 4
        x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
        roi = frame[y0:int(y) + pad, x0:int(x) + pad]
        if roi.size == 0:
            continue
        ring = roi.copy()
        cv2.circle(ring, (int(x) - x0, int(y) - y0), radius, ACCENT, max(2, round(3 * unit)), cv2.LINE_AA)
        cv2.addWeighted(ring, 1.0 - p, roi, p, 0, dst=roi)


def blur_regions(frame: np.ndarray, regions) -> None:
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in regions:
        roi = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if roi.size:
            roi[:] = cv2.GaussianBlur(roi, (0, 0), max(4.0, h / 60))


@lru_cache(maxsize=32)
def subtitle_patch(text: str, width: int) -> np.ndarray:
    """자막 한 장(RGBA): 반투명 둥근 바 + 흰 글자. 줄바꿈(\\n) 지원."""
    font = _font(max(14, width // 44), True)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.multiline_textbbox((0, 0), text, font=font, spacing=font.size // 3, align="center")
    pad_x, pad_y = font.size, font.size // 2
    # 여러 줄이면 Pillow 가 실수 bbox 를 준다 - 크기는 올림한 정수로.
    tw, th = math.ceil(box[2] - box[0]), math.ceil(box[3] - box[1])
    image = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=pad_y,
                           fill=(10, 12, 16, 175))
    draw.multiline_text((pad_x - box[0], pad_y - box[1]), text, font=font,
                        fill=(255, 255, 255, 255), spacing=font.size // 3, align="center")
    return np.asarray(image)


def overlay_rgba(frame: np.ndarray, patch: np.ndarray, x: int, y: int, level: float) -> None:
    h, w = patch.shape[:2]
    x, y = max(0, x), max(0, y)
    roi = frame[y:y + h, x:x + w]
    patch = patch[:roi.shape[0], :roi.shape[1]]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0 * level
    roi[:] = (roi * (1.0 - alpha) + patch[..., :3] * alpha).astype(np.uint8)


def draw_subtitle(frame: np.ndarray, text: str, level: float) -> None:
    h, w = frame.shape[:2]
    patch = subtitle_patch(text, w)
    overlay_rgba(frame, patch, (w - patch.shape[1]) // 2, int(h * 0.9) - patch.shape[0], level)


def _rect_pixels(frame: np.ndarray, rect) -> np.ndarray:
    x0, y0, x1, y1 = (max(0, int(v)) for v in rect)
    return frame[y0:y1, x0:x1]


def capture_note_refs(frame: np.ndarray, t: float, notes: list, state: dict) -> None:
    """note 의 기준 화면(t_ref, 없으면 t 시각의 그 영역)을 잡아 둔다. 매 프레임 부른다(잘린 프레임도)."""
    for note in notes:
        if note["rects"] and note["t"] not in state and t >= note.get("t_ref", note["t"]):
            state[note["t"]] = [_rect_pixels(frame, rect).copy() for rect in note["rects"]]
            state[(note["t"], "gone")] = set()


def visible_note_rects(frame: np.ndarray, note: dict, state: dict) -> list:
    """판독한 화면이 아직 그대로인 rect 만. 한 번 바뀌면(창이 덮으면) 그 note 동안 다시 안 그린다.

    note 는 판독이 끝난 뒤에 도착한다 - 점유 판독 note 는 더블클릭 뒤라, 그 사이 tool 창이
    List 를 덮으면 박스가 엉뚱한 화면 위에 뜬다(Codex 리뷰). 그래서 기준은 t_ref(판독 대상이
    확실히 보이던 시각)의 그 영역이다. t_ref 가 없으면 note 첫 프레임.
    """
    key = note["t"]
    if key not in state:
        capture_note_refs(frame, key, [note], state)
    gone = state[(key, "gone")]
    kept = []
    for index, (rect, ref) in enumerate(zip(note["rects"], state[key])):
        now = _rect_pixels(frame, rect)
        if index in gone or now.shape != ref.shape or ref.size == 0:
            continue
        if float(cv2.absdiff(now, ref).mean()) > NOTE_BOX_CHANGE:
            gone.add(index)
            continue
        kept.append(rect)
    return kept


def draw_note_boxes(frame: np.ndarray, rects: list, level: float, unit: float) -> None:
    """판독한 영역 강조 사각형(원본 좌표, 확대 전에 그린다)."""
    thick = max(2, round(3 * unit))
    for x0, y0, x1, y1 in rects:
        pad = thick * 2
        ax, ay = max(0, int(x0) - pad), max(0, int(y0) - pad)
        roi = frame[ay:int(y1) + pad, ax:int(x1) + pad]
        if roi.size == 0:
            continue
        box = roi.copy()
        cv2.rectangle(box, (int(x0) - ax - thick, int(y0) - ay - thick),
                      (int(x1) - ax + thick, int(y1) - ay + thick), ACCENT, thick, cv2.LINE_AA)
        cv2.addWeighted(box, level, roi, 1.0 - level, 0, dst=roi)


@lru_cache(maxsize=32)
def note_patch(title: str, lines: tuple, width: int, image_path: str = "") -> np.ndarray:
    """"AI 판독" 패널(RGBA): 반투명 카드 + 강조 막대 + 머리말/제목/내용 (+ 근거 정지화면)."""
    head_font, title_font = _font(max(11, width // 110), True), _font(max(14, width // 64), True)
    body_font = _font(max(12, width // 76), False)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    rows = [("AI 판독", head_font, (*ACCENT, 255)), (title, title_font, (255, 255, 255, 255))]
    rows += [(part, body_font, (214, 220, 230, 255))
             for line in lines for part in (textwrap.wrap(line, NOTE_WRAP) or [""])]
    sizes = [probe.textbbox((0, 0), text, font=font) for text, font, _ in rows]
    gap = body_font.size // 2
    pad, bar = title_font.size, max(4, width // 320)
    thumb = None
    if image_path:
        try:
            with Image.open(image_path) as source:
                thumb = source.convert("RGBA")
            thumb.thumbnail((int(width * NOTE_IMAGE_WIDTH), int(width * NOTE_IMAGE_WIDTH)))
        except OSError as exc:
            print(f"[WARNING] 판독 근거 이미지 열기 실패(패널은 글자만): {exc}")
    inner_w = max([b[2] - b[0] for b in sizes] + ([thumb.width] if thumb else []))
    inner_h = sum(b[3] - b[1] for b in sizes) + gap * (len(rows) - 1)
    inner_h += thumb.height + pad // 2 if thumb else 0
    image = Image.new("RGBA", (inner_w + pad * 2 + bar, inner_h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=pad // 2,
                           fill=(10, 12, 16, 200))
    draw.rectangle((0, pad // 2, bar - 1, image.height - pad // 2), fill=(*ACCENT, 255))
    y = pad
    for (text, font, color), box in zip(rows, sizes):
        draw.text((bar + pad - box[0], y - box[1]), text, font=font, fill=color)
        y += box[3] - box[1] + gap
    if thumb:
        image.alpha_composite(thumb, (bar + pad, y - gap + pad // 2))
    return np.asarray(image)


def draw_note_panel(frame: np.ndarray, note: dict, level: float) -> None:
    h, w = frame.shape[:2]
    patch = note_patch(note["title"], tuple(note["lines"]), w, note.get("image_path", ""))
    margin = w // 40
    overlay_rgba(frame, patch, w - patch.shape[1] - margin, margin, level)


def render_card(title: str, body: str, size: tuple) -> np.ndarray:
    """설명 카드 한 장(RGB)."""
    w, h = size
    image = Image.new("RGB", size, (16, 20, 28))
    draw = ImageDraw.Draw(image)
    title_font, body_font = _font(max(20, h // 13), True), _font(max(14, h // 26), False)
    tb = draw.textbbox((0, 0), title, font=title_font)
    bb = draw.multiline_textbbox((0, 0), body, font=body_font, spacing=body_font.size // 2,
                                 align="center") if body else (0, 0, 0, 0)
    gap = h // 18
    total = (tb[3] - tb[1]) + gap * 2 + (bb[3] - bb[1])
    y = (h - total) // 2
    draw.text(((w - (tb[2] - tb[0])) // 2 - tb[0], y - tb[1]), title, font=title_font,
              fill=(255, 255, 255))
    y += tb[3] - tb[1] + gap
    bar = max(40, w // 24)
    draw.rectangle(((w - bar) // 2, y - 2, (w + bar) // 2, y + 2), fill=ACCENT)
    if body:
        draw.multiline_text(((w - (bb[2] - bb[0])) // 2 - bb[0], y + gap - bb[1]), body,
                            font=body_font, fill=(200, 206, 216), spacing=body_font.size // 2,
                            align="center")
    return np.asarray(image)


# ------------------------------------------------------------------
# 조립.
# ------------------------------------------------------------------


def resolve_clip_dir(name: str) -> Path:
    path = Path(name).expanduser()
    if (path / VIDEO_NAME).is_file():
        return path
    matches = [d for d in DEMO_ROOT.glob(f"{name}*") if (d / VIDEO_NAME).is_file()]
    if not matches:
        raise FileNotFoundError(f"{VIDEO_NAME} 가 있는 clip 폴더가 없습니다: {name!r} ({DEMO_ROOT})")
    return max(matches, key=lambda d: d.stat().st_mtime)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_card(writer, item: dict, size: tuple, fps: int) -> int:
    image = render_card(item["card"], item.get("body", ""), size)
    sec = float(item.get("sec", 4.0))
    count = max(1, round(sec * fps))
    for i in range(count):
        level = fade_level(i / fps, 0.0, sec, FADE_SEC)
        writer.send(image if level >= 1.0 else (image * level).astype(np.uint8))
    print(f"[INFO] 카드 {item['card']!r}: {sec:.1f}s")
    return count


def write_clip(writer, item: dict, size: tuple, fps: int) -> int:
    import imageio_ffmpeg

    clip_dir = resolve_clip_dir(item["clip"])
    info = _read_json(clip_dir / EVENTS_NAME, {})
    stages = _read_json(clip_dir / STAGES_NAME, [])
    reader = imageio_ffmpeg.read_frames(str(clip_dir / VIDEO_NAME))
    meta = next(reader)
    fw, fh = meta["size"]
    src_fps = meta["fps"] or fps
    if abs(src_fps - fps) > 0.5:
        print(f"[WARNING] {clip_dir.name}: 원본 {src_fps}fps != 출력 {fps}fps - 속도가 달라집니다")
    start, end = clip_range(item, stages, float(meta["duration"]), STAGE_PAD_SEC)

    mon = info.get("monitor") or {"left": 0, "top": 0, "width": fw, "height": fh}

    def to_px(x, y):
        px = (x - mon["left"]) * fw / mon["width"]
        py = (y - mon["top"]) * fh / mon["height"]
        return (px, py) if 0 <= px < fw and 0 <= py < fh else None

    clicks = []
    for event in info.get("events", []):
        point = to_px(event["x"], event["y"]) if event.get("kind") == "click" else None
        if point is not None:
            clicks.append((event["t"], *point))
    # 화면이 안 바뀌는 클릭(같은 자리 반복)과 휠도 '움직임' 이다 - 정지 구간으로 자르지 않는다.
    activity = sorted(event["t"] for event in info.get("events", []))
    notes = []
    for note in _read_json(clip_dir / NOTES_NAME, []):
        if note["t"] + NOTE_SEC < start or note["t"] > end:  # 컷 시작에 걸친 note 는 남긴다
            continue
        rects = []
        for box in note.get("boxes", []):
            a, b = to_px(box["left"], box["top"]), to_px(box["right"], box["bottom"])
            if a and b:
                rects.append((a[0], a[1], b[0], b[1]))
        image = note.get("image", "")
        notes.append({**note, "rects": rects,
                      "image_path": str(clip_dir / image) if image else ""})
    if notes and "end" not in item:
        # 구간 끝에 뜬 판독 패널(보정 결과 등)이 읽히기 전에 잘리지 않게 끝을 늘린다.
        end = min(float(meta["duration"]),
                  max(end, max(n["t"] for n in notes) + NOTE_SEC + FADE_SEC))
    cursor = info.get("cursor", [])
    subtitles = item.get("subtitles")
    if subtitles is None:
        subtitles = [(s["start"], s["end"], s["text"])
                     for s in _read_json(clip_dir / SUBTITLES_NAME, [])]

    out_w, out_h = size
    base = base_rect(item.get("crop"), fw, fh, out_w / out_h)
    home = (base[0] + base[2] / 2, base[1] + base[3] / 2)
    zoom = float(item.get("zoom", ZOOM))
    blur = list(BLUR_REGIONS) + list(item.get("blur", []))
    focus = note_focus_points(notes, base, zoom, SHOT_MARGIN) if NOTE_FOCUS else []
    note_shots = plan_shots(focus, base, zoom, gap=0.0, margin=SHOT_MARGIN,
                            lead=ZOOM_LEAD_SEC, hold=NOTE_SEC)
    shots = plan_shots(clicks, base, zoom, gap=SHOT_GAP_SEC, margin=SHOT_MARGIN,
                       lead=ZOOM_LEAD_SEC, hold=ZOOM_HOLD_SEC)
    unit = fh / 1080 * 1.3
    cam = (1.0, *home)
    box_state = {}
    prev_small, prev_cur, last_change = None, None, start
    written = cut = 0
    try:
        for i, raw in enumerate(reader):
            t = i / src_fps
            if t > end:
                break
            cur = to_px(*cursor[min(i, len(cursor) - 1)]) if cursor else None
            target = camera_goal(t, note_shots, shots, home, zoom)
            new_cam = step_camera(cam, target, 1.0 / src_fps, CAMERA_TAU_SEC)
            moving = abs(new_cam[0] - cam[0]) > 1e-3 or math.dist(new_cam[1:], cam[1:]) > 0.5
            cam = new_cam
            # 판독 기준 화면은 컷 시작 전에 있을 수 있다 - 그 프레임은 기준을 잡으려고만 푼다.
            if any(n["rects"] and n["t"] not in box_state and t >= n.get("t_ref", n["t"])
                   for n in notes):
                capture_note_refs(np.frombuffer(raw, np.uint8).reshape(fh, fw, 3), t, notes,
                                  box_state)
            if t < start:
                continue

            frame = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3).copy()
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (fw // 8, fh // 8),
                               interpolation=cv2.INTER_AREA)
            changed = (prev_small is None or cur != prev_cur
                       or any(0.0 <= t - ta <= RING_SEC for ta in activity)
                       or np.count_nonzero(cv2.absdiff(small, prev_small) > 10) >= 3)
            prev_small, prev_cur = small, cur
            if changed:
                last_change = t
            text = next((s[2] for s in subtitles if s[0] <= t < s[1]), "")
            note, note_level = active_note(t, notes, NOTE_SEC)
            # 가림(blur) 전에 비교한다 - 기준 화면도 가림 전이다.
            note_rects = visible_note_rects(frame, note, box_state) if note and note["rects"] else []
            edge = t - start < FADE_SEC or end - t < FADE_SEC
            if (IDLE_MAX_SEC > 0 and t - last_change > IDLE_MAX_SEC
                    and not text and note is None and not moving and not edge):
                cut += 1
                continue

            blur_regions(frame, blur)
            if note_rects:
                draw_note_boxes(frame, note_rects, note_level, unit)
            if CLICK_RING:
                draw_click_rings(frame, t, clicks, unit)
            if CURSOR and cur is not None:
                draw_cursor(frame, cur[0], cur[1], unit)

            x0, y0, w, h = view_rect(cam, base)
            scale = min(out_w / w, out_h / h)
            shrink = min(1.0, min(out_w / base[2], out_h / base[3]))
            if shrink < 1.0:  # 크게 줄일 때 글자가 깨지지 않게 먼저 면적 보간으로 줄인다.
                frame = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            s = scale / shrink
            tx = (out_w - w * scale) / 2 - x0 * scale
            ty = (out_h - h * scale) / 2 - y0 * scale
            out = cv2.warpAffine(frame, np.float32([[s, 0, tx], [0, s, ty]]), (out_w, out_h),
                                 flags=cv2.INTER_CUBIC)
            # 종횡비가 달라 생긴 여백(레터박스)에 뷰 밖 화면이 비치지 않게 검게 칠한다.
            px0, py0 = int(round((out_w - w * scale) / 2)), int(round((out_h - h * scale) / 2))
            if px0 > 0:
                out[:, :px0] = 0
                out[:, out_w - px0:] = 0
            if py0 > 0:
                out[:py0] = 0
                out[out_h - py0:] = 0

            if note is not None:
                draw_note_panel(out, note, note_level)
            if text:
                sub = next(sub for sub in subtitles if sub[0] <= t < sub[1])
                draw_subtitle(out, text, fade_level(t, sub[0], sub[1], SUBTITLE_FADE_SEC))
            level = fade_level(t, start, end, FADE_SEC)
            if level < 1.0:
                out = (out * level).astype(np.uint8)
            writer.send(out)
            written += 1
    finally:
        reader.close()
    print(f"[INFO] clip {clip_dir.name} {item.get('stage', '')}: 원본 {start:.1f}~{end:.1f}s -> "
          f"{written / fps:.1f}s (멈춘 구간 {cut / src_fps:.1f}s 잘라냄, 클릭 {len(clicks)}, "
          f"확대 {len(shots) + len(note_shots)}회, 판독 {len(notes)})")
    return written


def main(sequence=None, output: str = "") -> str:
    import imageio_ffmpeg

    sequence = SEQUENCE if sequence is None else sequence
    size = OUT_SIZE
    if PREVIEW_WIDTH:
        size = (PREVIEW_WIDTH - PREVIEW_WIDTH % 2,
                round(PREVIEW_WIDTH * OUT_SIZE[1] / OUT_SIZE[0] / 2) * 2)
    out_path = Path(output or OUTPUT or DEMO_ROOT / f"final_{time.strftime('%y%m%d_%H%M%S')}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out_path), size, fps=FPS, codec="libx264", quality=None, macro_block_size=1,
        pix_fmt_out="yuv420p", ffmpeg_log_level="error",
        output_params=["-preset", "veryfast" if PREVIEW_WIDTH else "slow", "-crf", str(CRF),
                       "-profile:v", "high", "-movflags", "+faststart"],
    )
    writer.send(None)
    frames = 0
    try:
        for item in sequence:
            if "card" in item:
                frames += write_card(writer, item, size, FPS)
            elif "clip" in item:
                frames += write_clip(writer, item, size, FPS)
            else:
                print(f"[WARNING] card/clip 이 아닌 항목은 건너뜀: {item}")
    finally:
        writer.close()
    print(f"[INFO] 완료 -> {out_path} ({frames / FPS:.1f}s, {size[0]}x{size[1]}, "
          f"{out_path.stat().st_size / 1e6:.1f} MB)")
    return str(out_path)


if __name__ == "__main__":
    main()
