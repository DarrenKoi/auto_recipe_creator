"""Recipe Monitor Assist Window 의 score 색상을 읽어 측정 성부를 판정한다.

Assist Window 는 tool 창 내부 패널이다(별도 최상위 창이 아님). 3열(Addressing1 /
Addressing2 / Measurement) x 7행으로 최신 7회 측정의 썸네일과 score 가 실시간으로 쌓이고,
**score 숫자의 색이 곧 성부다** - 검정이면 정상 측정, 빨강이면 측정 실패. 측정이 진행
중인 행은 빈칸이다. 최신 행은 맨 아래에 쌓인다.

설계 경계(프로젝트 규칙): VLM 은 패널 *영역*만 1회 식별하고, 색 판정은 전부 CV 가 한다.
우리가 필요한 건 값이 아니라 색이므로 폴링마다 OCR 을 돌리지 않는다 - 읽지 않는 정보는
틀릴 수 없다.
"""

import numpy as np
from dataclasses import dataclass, field
from PIL import ImageDraw

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.util import crop_image
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
from poc.workflow_3.vlm.prompts import build_spotting_prompt
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

LOG_NAME = "assist_score"
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "assist_score"
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"

# 패널 crop 여유 - 로케이터가 준 점 주변을 넉넉히 잘라 표 전체를 담는다.
PANEL_LEFT_RATIO = 0.22
PANEL_RIGHT_RATIO = 0.22
PANEL_TOP_RATIO = 0.14
PANEL_BOTTOM_RATIO = 0.22

# 표 형태.
ASSIST_ROWS = 7
ASSIST_NEWEST_ROW_AT = "bottom"  # tool 버전이 다르면 "top".
ASSIST_COLUMNS = ("Addressing1", "Addressing2", "Measurement")

# 색 분류 임계. 배경은 밝고 잉크(글자)는 어둡다는 전제.
INK_MEAN_MAX = 200      # 채널 평균이 이보다 어두우면 잉크로 본다.
INK_MIN_PIXELS = 6      # 잉크가 이보다 적으면 빈칸(안티에일리어싱 무시).
RED_CHROMA_MIN = 60     # max-min 이 이 이상이면 유채색.
RED_DOMINANCE_MIN = 40  # R - max(G,B) 가 이 이상이면 붉은 계열.
RED_RATIO_MIN = 0.30    # 잉크 중 빨강 비율이 이 이상이면 red.
BLACK_RATIO_MAX = 0.10  # 이 이하면 black. 사이는 unknown.

# SEM 썸네일처럼 셀 대부분이 어두운 픽셀로 덮이면 글자가 아니라 이미지다 - 이 비율을
# 넘으면 무조건 unknown 으로 돌려 streak 을 끊는다(검정으로 오판정하는 것보다 안전한
# 방향).
#
# 값의 근거(0.85):
#   - 실제 숫자를 자기 tight bbox 안에서 재면 잉크 비율이 0.44~0.74(중앙값 0.58)다
#     (Arial/Arial Bold/Helvetica/Verdana Bold, 11/13/16px 렌더 측정).
#   - 썸네일로 덮인 셀은 0.95~1.0 이다.
#   - 게다가 `build_score_grid` 가 셀을 글자 bbox 보다 넉넉히(축당 CELL_PAD_*_RATIO)
#     키우므로 실제 숫자는 배경 위에 놓여 위 tight bbox 값보다 훨씬 더 성글어진다.
# 즉 0.85 는 정상 숫자 분포 전체보다 위, 썸네일 분포보다 아래에 있다. 이전 값 0.55 는
# 숫자 분포 한가운데(중앙값 0.58 바로 아래)라, 셀이 글자에 딱 붙던 시절 정상 숫자까지
# unknown 으로 끊어 기능이 통째로 무력화됐다.
INK_DENSE_MAX_RATIO = 0.85


def classify_ink(cell_rgb: np.ndarray) -> str:
    """셀 하나의 잉크 색을 판정한다. "black"|"red"|"blank"|"unknown".

    입력은 RGB numpy 배열이다(PIL Image 를 np.array 로 바꾼 형태). 흑/적 비율이 어느
    쪽으로도 확실하지 않으면 "unknown" 을 돌려준다 - 호출부가 streak 을 끊게 해서
    애매함이 done 판정으로 새지 않게 한다.
    """
    if cell_rgb is None or cell_rgb.size == 0:
        return "blank"
    arr = cell_rgb.astype(np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return "blank"

    red_c = arr[:, :, 0]
    green_c = arr[:, :, 1]
    blue_c = arr[:, :, 2]
    mean = arr[:, :, :3].mean(axis=2)
    ink = mean < INK_MEAN_MAX
    ink_count = int(ink.sum())
    if ink_count < INK_MIN_PIXELS:
        return "blank"

    total_px = int(ink.size)
    if total_px > 0 and (ink_count / total_px) > INK_DENSE_MAX_RATIO:
        # 잉크가 셀 대부분을 덮음 - 글자가 아니라 썸네일(그레이스케일 SEM 이미지) 등
        # 밀집 영역일 가능성이 크다. black 으로 잘못 읽어 streak 을 부풀리는 것보다
        # 여기서 끊는 게 안전하다.
        return "unknown"

    chroma = arr[:, :, :3].max(axis=2) - arr[:, :, :3].min(axis=2)
    dominance = red_c - np.maximum(green_c, blue_c)
    is_red = ink & (chroma >= RED_CHROMA_MIN) & (dominance >= RED_DOMINANCE_MIN)
    red_ratio = float(is_red.sum()) / float(ink_count)

    if red_ratio >= RED_RATIO_MIN:
        return "red"
    if red_ratio <= BLACK_RATIO_MAX:
        return "black"
    return "unknown"


# Addressing2 는 대개 비어 있어 판정에 쓰지 않는다.
ASSIST_CRITICAL_COLUMNS = ("Addressing1", "Measurement")


def row_verdict(cells: dict) -> str:
    """측정 1회(행 하나)의 성부. "ok"|"fail"|"pending"|"unknown".

    pending 을 **Measurement 기준으로만** 판정하는 이유: Addressing2 는 대개 비어 있고
    Addressing1 도 레시피에 따라 없을 수 있다. 없는 칸을 '진행 중' 으로 읽으면 그
    레시피는 영영 done 이 되지 않는다. Measurement 가 최종 결과이므로 그것으로 완료를
    판정하고, Addressing1 은 값이 있을 때만 실패 신호로 쓴다.
    """
    critical = [cells.get(name, "blank") for name in ASSIST_CRITICAL_COLUMNS]
    if any(state == "red" for state in critical):
        return "fail"
    if any(state == "unknown" for state in critical):
        return "unknown"
    if cells.get("Measurement", "blank") == "blank":
        return "pending"
    return "ok"


@dataclass
class RowState:
    """Assist Window 의 행 하나 - 열별 잉크 색과 그로부터 나온 성부."""

    cells: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return row_verdict(self.cells)


def ok_streak(rows: list) -> int:
    """최신 행부터 세어 연속 정상(ok) 개수. fail/unknown 을 만나면 멈춘다.

    최신 쪽의 pending(측정 진행 중)은 건너뛴다 - 아직 결과가 안 나온 행이 그 앞의 연속
    정상 기록을 지우면 안 된다. rows 는 물리적 위(index 0)에서 아래(index 끝) 순서다.
    `ASSIST_NEWEST_ROW_AT` 이 "bottom"(기본)이면 index 끝이 최신이라 뒤에서 앞으로
    걷고, "top" 이면 index 0 이 최신이라 앞에서 뒤로 걷는다 - `build_score_grid` 의
    행 anchoring 과 반드시 짝을 맞춰야 한다(하나만 뒤집으면 조용히 반대로 읽는다).
    """
    if ASSIST_NEWEST_ROW_AT == "top":
        idx = 0
        n = len(rows)
        while idx < n and rows[idx].verdict == "pending":
            idx += 1
        streak = 0
        while idx < n and rows[idx].verdict == "ok":
            streak += 1
            idx += 1
        return streak

    idx = len(rows) - 1
    while idx >= 0 and rows[idx].verdict == "pending":
        idx -= 1
    streak = 0
    while idx >= 0 and rows[idx].verdict == "ok":
        streak += 1
        idx -= 1
    return streak


# 정규화 후 패널 crop 안에 남은 항목이 이보다 적으면(헤더 최소 요구치 미달) 격자를
# 만들 근거가 없다고 보고 즉시 포기한다. build_score_grid 도 자체적으로 같은 상황을
# 걸러내지만, 좌표계 오판정으로 항목 대부분이 통째로 버려지는 것을 여기서 먼저
# 눈에 띄게 로그한다.
MIN_USABLE_SPOTTING_ITEMS = 4  # 헤더 3개 + 숫자 1개가 최소.

# 0-1000 정규화 좌표 판정용 띠. 정규화 응답은 가장 큰 좌표가 1000 부근에 붙는다.
NORM1000_BAND_LO = 950.0
NORM1000_BAND_HI = 1005.0


def _resolve_item_coord_space(items: list, panel_size: tuple) -> tuple:
    """OCR spotting 항목 bbox 가 어느 좌표계인지 판정해 (sx, sy, space) 를 돌려준다.

    PaddleOCR-VL 의 ``Spotting:`` 이 패널 crop 픽셀 좌표를 돌려준다고 가정하지만,
    모델이 대신 0-1 비율이나 0-1000 정규화 좌표를 줄 수 있다(pm_dropdown.py 의
    같은 서비스/같은 창에서 실제로 관찰됨). 텍스트 매칭은 좌표와 무관하므로 헤더는
    어느 좌표계든 그대로 잡히고, 격자도 만들어진다 - 좌표계를 안 맞추면 셀이 실제
    글자가 아니라 엉뚱한 자리(예: 회색조 SEM 썸네일)를 가리키는 조용한 오류가 된다.
    bbox 최대값으로 좌표계를 추정한다(패널 crop 자체가 리사이즈되지 않으므로
    crop_px 가 기본 가정).
    """
    pw, ph = panel_size
    max_coord = 0.0
    for item in items:
        bbox = item.get("bbox")
        if bbox:
            max_coord = max(max_coord, float(bbox.get("right", 0)), float(bbox.get("bottom", 0)))
    if items and max_coord <= 1.5:
        return float(pw), float(ph), "frac01"
    if NORM1000_BAND_LO <= max_coord <= NORM1000_BAND_HI:
        # 최대 좌표가 1000 바로 언저리면 패널 크기와 무관하게 0-1000 정규화로 본다.
        # crop 픽셀 좌표가 하필 이 좁은 띠 안에서 최대값을 찍을 확률은 사실상 없지만,
        # 큰 모니터(패널 crop 이 1000px 을 넘음)에서는 아래의 크기 의존 임계가 진짜
        # 0-1000 좌표를 crop 픽셀로 오판정한다 - 그러면 셀이 썸네일 위에 놓여 모든 행이
        # black 으로 읽히고 streak 이 7 에 고정되는(이 검출을 도입한 이유였던) 버그가
        # 조용히 되살아난다.
        return pw / 1000.0, ph / 1000.0, "norm1000"
    if max_coord <= max(pw, ph) * 1.05:
        return 1.0, 1.0, "crop_px"
    return pw / 1000.0, ph / 1000.0, "norm1000"


def normalize_spotting_items_to_panel(items: list, panel_size: tuple) -> list:
    """OCR spotting 항목의 bbox 를 패널 crop 픽셀 좌표계로 맞추고, 범위를 벗어난 항목은
    버린다.

    `locate_assist_layout` 이 `build_score_grid` 를 부르기 **전에** 반드시 거쳐야 한다.
    안 거치면 0-1000/0-1 좌표가 crop 픽셀인 것처럼 그대로 들어가고, 텍스트 매칭은
    좌표 무관이라 헤더는 여전히 잡히므로 격자가 "성공적으로" 만들어진다 - 다만 셀이
    실제 숫자가 아니라 회색조 썸네일 위에 놓여, classify_ink 가 어두운 무채색을
    black 으로 읽어 streak 이 허위로 쌓인다.

    패널 밖으로 매핑된 항목은 좌표계 오판정이거나 OCR 오검출이므로 버린다(그런 항목이
    셀 계산에 섞이면 잘못된 곳을 가리키는 셀이 나온다).
    """
    if not items:
        return []
    pw, ph = panel_size
    sx, sy, space = _resolve_item_coord_space(items, panel_size)
    if space != "crop_px":
        print(f"[INFO] Assist 좌표계 보정: {space} (panel={pw}x{ph})")

    out = []
    for item in items:
        bbox = item.get("bbox")
        if not bbox:
            continue
        try:
            left = float(bbox.get("left", 0)) * sx
            top = float(bbox.get("top", 0)) * sy
            right = float(bbox.get("right", 0)) * sx
            bottom = float(bbox.get("bottom", 0)) * sy
        except (TypeError, ValueError):
            continue
        if right <= left or bottom <= top:
            continue
        # 패널 crop 밖으로 나간 항목은 버린다(썸네일/다른 패널을 셀로 오인 방지).
        if left < 0 or top < 0 or right > pw or bottom > ph:
            continue
        out.append({
            **item,
            "bbox": {
                "left": left, "top": top, "right": right, "bottom": bottom,
            },
        })
    return out


# 셀 여유(padding). 셀 x 범위를 숫자 bbox 합집합으로 잡으면 글자가 셀을 꽉 채워
# classify_ink 의 밀집 가드(INK_DENSE_MAX_RATIO)를 건드린다 - 글자가 배경 위에 놓이도록
# 축당 이만큼 키운다(각 변에 폭/높이의 35%).
CELL_PAD_X_RATIO = 0.35
CELL_PAD_Y_RATIO = 0.35

# 세로 여유는 행 pitch 를 넘으면 안 된다 - 넘으면 셀이 위/아래 이웃 행의 숫자를 함께
# 담아 색 판정이 섞인다(글자 높이가 pitch 의 큰 부분을 차지하는 표에서 실제로 발생).
# 패딩 후 높이를 pitch 의 이 비율로 잘라 행 사이에 눈에 보이는 틈을 남긴다.
CELL_MAX_PITCH_RATIO = 0.9


def _pad_span(low: float, high: float, ratio: float) -> tuple:
    """[low, high] 구간을 폭의 ratio 만큼 양쪽으로 넓힌다."""
    pad = (high - low) * ratio
    return low - pad, high + pad


# 헤더 텍스트 매칭용 - 영숫자만 남기고 소문자 비교(OCR 공백/기호 흔들림 흡수).
def _normalize(text: str) -> str:
    """열 이름 비교용 정규화."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _is_score_text(text: str) -> bool:
    """score 로 볼 텍스트인지(숫자만)."""
    stripped = (text or "").strip()
    return bool(stripped) and all(ch.isdigit() for ch in stripped)


@dataclass
class AssistLayout:
    """Assist 패널의 score 셀 격자. 1회 만들어 캐시한다."""

    grid: list        # grid[row][col] = {"left","top","right","bottom"} (패널 crop 좌표계)
    columns: tuple


def _assign_number_column(cx: float, header_boxes: dict):
    """숫자 항목의 x 중심이 속하는(또는 가장 가까운) 헤더 열을 고른다.

    헤더 x 범위 안에 들어오면 그 열로 확정한다. 어느 범위에도 안 들어오면(칼럼 사이
    간격 등) 가장 가까운 헤더로 배정한다 - 배정을 포기하면 그 숫자는 어느 열의 x 범위
    계산에도 기여하지 못해 헤더 폴백으로 새어버린다.
    """
    best_column = None
    best_dist = None
    for column, box in header_boxes.items():
        left = float(box.get("left", 0))
        right = float(box.get("right", 0))
        if left <= cx <= right:
            return column
        dist = min(abs(cx - left), abs(cx - right))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_column = column
    return best_column


def build_score_grid(items: list, panel_size: tuple, *, rows: int = ASSIST_ROWS):
    """OCR spotting 항목에서 score 셀 격자를 만든다. 실패 시 None.

    열은 헤더 텍스트로 식별한다(순서로 추정하지 않는다 - Addressing2 가 비어 있으면 숫자
    덩어리가 2개뿐이라 어느 것이 Measurement 인지 알 수 없다). 단, 각 열의 **x 범위**는
    헤더 bbox 가 아니라 그 열에 배정된 **숫자 bbox 들의 합집합**으로 잡는다 - 헤더가
    썸네일까지 덮는 넓은 칼럼 위에 놓이면 헤더 x 범위를 그대로 쓴 셀이 썸네일과 겹쳐
    회색조 이미지를 글자로 오판정할 수 있다. 그 열에 숫자가 하나도 안 잡히면(예:
    Addressing2 가 대개 비어 있음) 헤더 x 범위로 폴백한다.

    행은 숫자 항목의 y 중심을 모아 띠(band) 간 간격의 중앙값으로 pitch 를 구한 뒤 rows
    개로 외삽한다 - 중간 띠 하나가 누락돼도 다수결로 버틴다.

    셀은 글자 bbox 에 딱 붙이지 않고 축당 `CELL_PAD_*_RATIO` 만큼 키운다 - 글자가 셀을
    꽉 채우면 `classify_ink` 의 밀집 가드가 정상 숫자를 썸네일로 오인해 기능이 통째로
    죽는다. 세로는 `CELL_MAX_PITCH_RATIO` 로 pitch 안에 가둬 이웃 행을 물지 않게 한다.

    items 는 패널 crop 좌표계여야 한다(호출부는 `normalize_spotting_items_to_panel` 로
    먼저 좌표계를 맞춰야 한다). panel_size 는 (width, height).
    """
    if not items:
        return None

    # --- 열: 헤더 텍스트로 식별 ---
    header_boxes = {}
    for item in items:
        name = _normalize(item.get("text", ""))
        for column in ASSIST_COLUMNS:
            if name == _normalize(column) and column not in header_boxes:
                header_boxes[column] = item.get("bbox") or {}
    if len(header_boxes) != len(ASSIST_COLUMNS):
        print(f"[WARNING] Assist 헤더 인식 부족({sorted(header_boxes)}) - 격자 생성 실패")
        return None

    # --- 행: 숫자 항목의 y 중심 -> pitch -> 외삽. 동시에 숫자를 열에 배정해 x 범위를 모은다 ---
    header_bottom = max(int(box.get("bottom", 0)) for box in header_boxes.values())
    number_x_ranges = {column: None for column in ASSIST_COLUMNS}
    centers = []
    heights = []
    for item in items:
        if not _is_score_text(item.get("text", "")):
            continue
        box = item.get("bbox") or {}
        top = float(box.get("top", 0))
        bottom = float(box.get("bottom", 0))
        if bottom <= header_bottom:
            continue
        left = float(box.get("left", 0))
        right = float(box.get("right", 0))
        centers.append((top + bottom) / 2.0)
        heights.append(max(1.0, bottom - top))

        column = _assign_number_column((left + right) / 2.0, header_boxes)
        if column is not None:
            current = number_x_ranges[column]
            if current is None:
                number_x_ranges[column] = [left, right]
            else:
                current[0] = min(current[0], left)
                current[1] = max(current[1], right)
    if not centers:
        print("[WARNING] Assist 숫자 항목 없음 - 격자 생성 실패")
        return None

    cell_h = int(round(sorted(heights)[len(heights) // 2]))
    band_centers = _cluster_1d(sorted(centers), tolerance=cell_h)
    if len(band_centers) < 2:
        print("[WARNING] Assist 행이 2개 미만 - pitch 를 알 수 없어 격자 생성 실패")
        return None

    gaps = sorted(band_centers[i + 1] - band_centers[i] for i in range(len(band_centers) - 1))
    pitch = gaps[len(gaps) // 2]
    if pitch <= 0:
        return None

    # 열별 x 범위 확정: 숫자가 배정됐으면 그 합집합, 아니면 헤더 폴백. 어느 쪽이든
    # 좌우로 여유를 줘 글자가 셀을 꽉 채우지 않게 한다(밀집 가드 오작동 방지).
    column_x_ranges = {}
    for column in ASSIST_COLUMNS:
        span = number_x_ranges[column]
        if span is not None:
            raw = (float(span[0]), float(span[1]))
        else:
            box = header_boxes[column]
            raw = (float(box.get("left", 0)), float(box.get("right", 0)))
        left, right = _pad_span(raw[0], raw[1], CELL_PAD_X_RATIO)
        column_x_ranges[column] = (int(round(left)), int(round(right)))

    # 세로 여유: 글자 높이의 CELL_PAD_Y_RATIO 를 위아래로 더하되, pitch 안에 확실히
    # 머무르도록 자른다. cell_h 가 pitch 의 큰 부분을 차지하면 패딩만 믿었다가 이웃 행을
    # 물어 들인다.
    padded_h = cell_h * (1.0 + 2.0 * CELL_PAD_Y_RATIO)
    max_h = pitch * CELL_MAX_PITCH_RATIO
    if padded_h > max_h:
        padded_h = max_h
    cell_h_padded = max(1, int(round(padded_h)))
    if cell_h_padded >= pitch:
        # 반올림이 pitch 를 건드리는 극단(pitch 가 아주 작은 표)에서도 틈을 보장한다.
        cell_h_padded = max(1, int(pitch) - 1)

    newest_at = ASSIST_NEWEST_ROW_AT
    if newest_at not in ("top", "bottom"):
        print(f"[WARNING] ASSIST_NEWEST_ROW_AT 값 이상({newest_at!r}) - bottom 으로 처리")
        newest_at = "bottom"

    if newest_at == "top":
        # 최신 행이 맨 위이므로 가장 위 띠를 첫 행(index 0)에 맞추고 아래로 채운다.
        anchor_center = band_centers[0]

        def _center_for(row_idx):
            return anchor_center + pitch * row_idx
    else:
        # 최신 행이 맨 아래이므로 가장 아래 띠를 마지막 행에 맞추고 위로 채운다.
        anchor_center = band_centers[-1]

        def _center_for(row_idx):
            return anchor_center - pitch * (rows - 1 - row_idx)

    grid = []
    for row_idx in range(rows):
        center = _center_for(row_idx)
        top = int(round(center - cell_h_padded / 2.0))
        bottom = top + cell_h_padded
        row_boxes = []
        for column in ASSIST_COLUMNS:
            left, right = column_x_ranges[column]
            row_boxes.append({
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
            })
        grid.append(row_boxes)

    return AssistLayout(
        grid=grid,
        columns=tuple(ASSIST_COLUMNS),
    )


def _cluster_1d(values: list, *, tolerance: int) -> list:
    """정렬된 1D 값들을 tolerance 이내로 묶어 각 묶음의 평균을 돌려준다."""
    if not values:
        return []
    clusters = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / float(len(group)) for group in clusters]


def read_row_states(image, layout) -> list:
    """패널 이미지에서 행별 상태를 읽는다. layout 이 없으면 빈 목록.

    image 는 패널 crop 된 PIL Image 다(격자 좌표계와 같아야 한다). 폴링마다 호출되며
    VLM/OCR 을 쓰지 않는다 - 셀 박스는 이미 격자에 있고 필요한 건 색뿐이다.
    """
    if layout is None or image is None:
        return []
    frame = np.array(image.convert("RGB"))
    height, width = frame.shape[:2]

    rows = []
    for row_boxes in layout.grid:
        cells = {}
        for column, box in zip(layout.columns, row_boxes):
            left = max(0, min(width, int(box["left"])))
            right = max(left, min(width, int(box["right"])))
            top = max(0, min(height, int(box["top"])))
            bottom = max(top, min(height, int(box["bottom"])))
            cells[column] = classify_ink(frame[top:bottom, left:right])
        rows.append(RowState(cells=cells))
    return rows


def assist_panel_target() -> TargetConfig:
    """Assist 패널 grounding 타겟.

    기하는 bench_tool_window_reader._button_target 와 같은 계열이다 - 같은 tool 창에서
    오피스 acc=1.000 이 나온 설정이라 근거 없는 값을 새로 만들지 않는다.
    """
    return TargetConfig(
        key="assist_panel",
        description=(
            "the Recipe Monitor Assist panel inside this CD-SEM tool window - the table "
            "that lists recent measurements with Addressing1 / Addressing2 / Measurement "
            "thumbnails and their score numbers stacked vertically. Return a point at the "
            "centre of that table, not on the live SEM image and not on the button row."
        ),
        left_pad_ratio=0.8,
        right_pad_ratio=0.8,
        vertical_pad_ratio=0.8,
        min_crop_width=320,
        min_crop_height=96,
        vertical_pad_min_px=16,
    )


def locate_assist_layout(window, window_title: str, backend: str, image):
    """Assist 패널을 찾아 score 격자를 만든다. 실패 시 None.

    watch 당 1회만 돈다(이후 폴링은 read_row_states 가 캐시된 격자를 쓴다). 반환은
    (panel_box, AssistLayout) 이며 panel_box 는 창-이미지 좌표계다.
    """
    try:
        result = analyze_window_target(
            window, window_title, backend, assist_panel_target(),
            debug_image_dir=DEBUG_ARTIFACT_DIR,
            log_name=LOG_NAME,
            component_name=LOG_NAME,
            artifact_prefix="assist_panel",
            image=image,
            timeout_sec=15.0,
        )
    except Exception as exc:
        print(f"[WARNING] Assist 패널 grounding 실패: {exc}")
        return None

    point = getattr(result, "point", None)
    if not point:
        print("[WARNING] Assist 패널을 찾지 못함 - 감지 비활성(cap 대기)")
        return None

    width, height = image.size
    panel_box = {
        "left": max(0, int(point["x"] - width * PANEL_LEFT_RATIO)),
        "right": min(width, int(point["x"] + width * PANEL_RIGHT_RATIO)),
        "top": max(0, int(point["y"] - height * PANEL_TOP_RATIO)),
        "bottom": min(height, int(point["y"] + height * PANEL_BOTTOM_RATIO)),
    }
    if panel_box["right"] <= panel_box["left"] or panel_box["bottom"] <= panel_box["top"]:
        print(f"[WARNING] Assist 패널 crop 영역이 비정상(grounding 점이 창 밖) - point={point}")
        return None
    try:
        panel = crop_image(image, panel_box)
    except Exception as exc:
        print(f"[WARNING] Assist 패널 crop 실패: {exc}")
        return None

    try:
        client = Workflow1VLMClient(OCR_SERVICE_SLUG, timeout_sec=30.0, log_name=LOG_NAME)
        image_b64, _w, _h = encode_image_webp(panel.convert("RGB"), quality=90)
        system_message, user_text = build_spotting_prompt()
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
        )
        items = parse_spotting_items(response.text)
    except Exception as exc:
        print(f"[WARNING] Assist 패널 OCR 실패: {exc}")
        return None

    items = normalize_spotting_items_to_panel(items, panel.size)
    if len(items) < MIN_USABLE_SPOTTING_ITEMS:
        print(
            f"[WARNING] Assist 좌표 정규화 후 사용 가능 항목 부족({len(items)}) - "
            "격자 생성 포기(확신 없는 격자보다 안전)"
        )
        return None

    layout = build_score_grid(items, panel.size)
    if layout is None:
        return None
    print(
        f"[INFO] Assist 격자 확보: panel={panel_box} rows={len(layout.grid)} "
        f"columns={layout.columns}"
    )
    return panel_box, layout


_VERDICT_COLORS = {
    "ok": (0, 200, 0),
    "fail": (255, 0, 0),
    "pending": (128, 128, 128),
    "unknown": (255, 160, 0),
}


def save_assist_overlay(image, layout, rows: list, out_path) -> None:
    """판독 결과를 패널 이미지 위에 그려 저장한다 (실패 무시).

    오피스가 행 방향(최신이 아래인지) / 열 매핑 / 색 임계를 한 장으로 검증할 수 있게
    한다. 폴링마다가 아니라 판정이 바뀔 때만 부른다.
    """
    try:
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for row_idx, row_boxes in enumerate(layout.grid):
            verdict = rows[row_idx].verdict if row_idx < len(rows) else "unknown"
            color = _VERDICT_COLORS.get(verdict, (255, 160, 0))
            for box in row_boxes:
                draw.rectangle(
                    [box["left"], box["top"], box["right"], box["bottom"]],
                    outline=color, width=2,
                )
            label = f"{row_idx}:{verdict}"
            draw.text((row_boxes[0]["left"], max(0, row_boxes[0]["top"] - 12)), label, fill=color)
        save_debug_jpeg(canvas, out_path)
    except Exception as exc:
        print(f"[WARNING] Assist 오버레이 저장 실패: {exc}")


__all__ = [
    "ASSIST_COLUMNS",
    "ASSIST_CRITICAL_COLUMNS",
    "ASSIST_NEWEST_ROW_AT",
    "ASSIST_ROWS",
    "AssistLayout",
    "RowState",
    "assist_panel_target",
    "build_score_grid",
    "classify_ink",
    "locate_assist_layout",
    "normalize_spotting_items_to_panel",
    "ok_streak",
    "read_row_states",
    "row_verdict",
    "save_assist_overlay",
]
