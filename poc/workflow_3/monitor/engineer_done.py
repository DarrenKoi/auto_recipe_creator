"""엔지니어 수동 align 보정 완료 감지 — Recipe Monitor 측정 카운터 기반.

미보정 engineer watch 동안 "측정이 시작됐다"(= align 완료)를 감지해 녹화를 조기
종료한다. 신호는 tool 창 Recipe Monitor 의 측정 점 카운터 분자(N/M 의 N)가
증가하는 것 (1/350 -> 2/350 -> ...).

hybrid 파이프라인 (사이클당 VLM 1회):
  1. grounding(1회 캐시): VLM(ui-venus)으로 분자 위치를 찾아 tool-window
     상대비율 ROI 로 캐시한다 (tool 마다/드래그로 위치가 달라 고정 ROI 불가).
  2. CV gate(매 호출): ROI crop 변화감지 — align-fix 중엔 카운터가 정적이라
     OCR 호출이 0회로 유지된다 (recording.py 의 다운샘플+delta 로직 재사용).
  3. OCR confirm(변화 시에만): paddleocr 로 분자 N 을 읽어
     N >= min_count 이고 직전 읽기 대비 비감소(연속 2회 확인)면 done.
     연속 2회 확인은 단일 프레임 OCR 오독(분모 등)으로 끊기는 것을 막는다.

실패는 전부 graceful: grounding 거부/OCR 실패/예외 -> False -> watch 의
engineer_watch_sec cap 이 안전망. (CLAUDE.md 규칙: VLM 은 위치만, 전이 판정의
정량 근거는 CV 변화 + OCR 숫자.)

오피스 캘리브레이션 (단독 실행):
  지금 측정 중인 tool 의 Remote Monitoring 창을 열어 두고 실행하면 — 측정
  중에는 분자가 실제로 증가하므로 — grounding/gate/OCR 전 체인을 align fail
  없이 즉시 검증할 수 있다.

  uv run python poc/workflow_3/monitor/engineer_done.py
"""

import re

_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_INT_RE = re.compile(r"\d+")


def parse_point_1000(text: str) -> tuple[int, int] | None:
    """ui-venus 응답에서 첫 [x,y](0-1000)를 파싱한다.

    거부([-1,-1])·범위 밖·미검출은 None.
    """
    match = _POINT_RE.search(text or "")
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def point_to_roi_ratios(
    x_1000: int, y_1000: int, pad_x: float, pad_y: float
) -> tuple[float, float, float, float] | None:
    """grounding 점(0-1000)을 분자 crop 용 상대비율 ROI (l,t,r,b) 로 확장한다."""
    cx, cy = x_1000 / 1000.0, y_1000 / 1000.0
    left = min(max(cx - pad_x, 0.0), 1.0)
    right = min(max(cx + pad_x, 0.0), 1.0)
    top = min(max(cy - pad_y, 0.0), 1.0)
    bottom = min(max(cy + pad_y, 0.0), 1.0)
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None
    return left, top, right, bottom


def extract_numerator(text: str) -> int | None:
    """OCR 텍스트에서 분자 정수를 뽑는다 ('2/350' -> 2, 첫 연속 숫자열)."""
    match = _INT_RE.search(text or "")
    if match is None:
        return None
    return int(match.group(0))
