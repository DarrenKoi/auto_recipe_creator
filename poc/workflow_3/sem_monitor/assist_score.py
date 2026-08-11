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
    정상 기록을 지우면 안 된다. 목록은 index 0 이 가장 오래된 행이다.
    """
    idx = len(rows) - 1
    while idx >= 0 and rows[idx].verdict == "pending":
        idx -= 1
    streak = 0
    while idx >= 0 and rows[idx].verdict == "ok":
        streak += 1
        idx -= 1
    return streak


__all__ = [
    "ASSIST_COLUMNS",
    "ASSIST_CRITICAL_COLUMNS",
    "ASSIST_NEWEST_ROW_AT",
    "ASSIST_ROWS",
    "RowState",
    "classify_ink",
    "ok_streak",
    "row_verdict",
]
