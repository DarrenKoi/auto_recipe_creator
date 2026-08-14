"""Recipe Monitor Assist Window 를 읽어 "측정이 정상 진행 중인가" 를 판정한다.

Assist Window 는 tool 창 내부 패널이다(별도 최상위 창이 아님). 3열(Addressing1 /
Addressing2 / Measurement)로 최근 측정의 썸네일과 score 가 쌓이고, **score 숫자의 색이
곧 성부다** - 검정이면 정상 측정, 빨강이면 측정 실패. 측정이 진행 중인 행은 빈칸이다.

판정 규칙 (2026-08-13 재설계):
  * 표는 **새 측정이 시작될 때 초기화된다**(스크롤 아님). 따라서 지금 표에 보이는 붉은
    숫자는 이번 사이클의 실패이며, 하나라도 있으면 완료가 아니다(보수적 정책 - 놓쳐도
    watch cap 이 안전망이고, 잘못 닫는 쪽이 훨씬 비싸다).
  * 검정 숫자 띠의 **개수**가 정상적으로 끝난 측정 행 수다. 열은 구분하지 않는다.

설계 경계(프로젝트 규칙): VLM 은 패널 *영역*만 1회 식별하고, 판정은 전부 CV 가 한다.
값이 아니라 색과 존재만 보므로 **OCR 이 필요 없다** - 읽지 않는 정보는 틀릴 수 없다.
(구 구현은 OCR 로 열 헤더를 매칭해 격자를 세웠고, 헤더 한 글자만 잘려도 통째로 실패했다.)
"""

import numpy as np
from dataclasses import dataclass

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.sem_monitor.sem_box_detect import true_runs
from poc.workflow_3.util import crop_image
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target

LOG_NAME = "assist_score"
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "assist_score"

# 패널 crop 여유 - 로케이터가 준 점 주변을 넉넉히 잘라 표 전체를 담는다.
PANEL_LEFT_RATIO = 0.22
PANEL_RIGHT_RATIO = 0.22
PANEL_TOP_RATIO = 0.14
PANEL_BOTTOM_RATIO = 0.22

# 색 분류 임계. 배경은 밝고 잉크(글자)는 어둡다는 전제.
INK_MEAN_MAX = 200      # 채널 평균이 이보다 어두우면 잉크로 본다.
INK_MIN_PIXELS = 6      # 잉크가 이보다 적으면 없는 것으로 본다(안티에일리어싱 무시).
RED_CHROMA_MIN = 60     # max-min 이 이 이상이면 유채색.
RED_DOMINANCE_MIN = 40  # R - max(G,B) 가 이 이상이면 붉은 계열.

# 한 스캔라인에 검정 잉크가 이만큼 있어야 그 줄을 글자 줄로 본다. 표 테두리 1px 선이나
# 안티에일리어싱 부스러기가 행으로 세어지지 않게 하는 최소치다.
ROW_MIN_INK_PX = 6

# 숫자 띠보다 위에 있는 열 헤더("Addressing1" 등) 줄 수. 헤더도 검정 잉크라 띠로 잡히므로
# 위에서부터 이만큼은 행으로 세지 않는다. 패널 crop 이 헤더를 잘라내 버린 경우에는 실제
# 데이터 행 하나를 잃지만, 그 방향(과소 계수 -> done 지연)이 안전한 쪽이다.
HEADER_BAND_COUNT = 1

# 글자 줄 높이의 이 배를 넘는 띠는 글자가 아니라 덩어리(그레이스케일 SEM 썸네일 등)로
# 본다. 절대 픽셀이 아니라 **같은 패널 안 띠 높이의 중앙값 대비**로 재는 이유는 오피스
# DPI 배율(125/150%)에 따라 글자 크기가 달라지기 때문이다.
BAND_MAX_HEIGHT_RATIO = 3.0


@dataclass(frozen=True)
class AssistObservation:
    """한 캡처 프레임에서 읽은 Assist 패널 상태.

    status="usable" 이 아니면 ok_row_count/has_red 는 의미가 없다 - 판독 자체가 안 된
    회차와 "읽었더니 0행" 은 완전히 다른 상태이므로 호출부가 섞으면 안 된다.
    """

    status: str
    ok_row_count: int = 0
    has_red: bool = False
    reason: str = ""


def _drop_tall_bands(bands: list) -> list:
    """글자 줄로 보기엔 지나치게 두꺼운 띠를 버린다.

    한계: 띠가 전부 덩어리뿐이면 중앙값 자체가 덩어리 높이라 아무것도 못 거른다.
    표에 글자 줄이 다수라는 전제에 기대는 판정이다.
    """
    if len(bands) < 2:
        return bands
    heights = sorted(end - start for start, end in bands)
    median = heights[len(heights) // 2]
    if median <= 0:
        return bands
    limit = median * BAND_MAX_HEIGHT_RATIO
    return [band for band in bands if (band[1] - band[0]) <= limit]


def read_assist_state(panel_image) -> AssistObservation:
    """패널 crop 픽셀만으로 Assist 상태를 읽는다 (OCR/VLM 없이 매 폴링 호출).

    붉은 잉크가 있으면 has_red, 검정 잉크 띠 개수가 ok_row_count 다. 열은 구분하지
    않으므로 한 행에 검정 숫자가 하나라도 있으면 측정 1회로 센다.
    """
    if panel_image is None:
        return AssistObservation(status="unusable", reason="no_image")
    # convert("RGB") 가 3채널을 보장하므로 남은 이상 상태는 빈 이미지뿐이다.
    frame = np.array(panel_image.convert("RGB")).astype(np.int16)
    if frame.size == 0:
        return AssistObservation(status="unusable", reason="bad_image")

    ink = frame.mean(axis=2) < INK_MEAN_MAX
    chroma = frame.max(axis=2) - frame.min(axis=2)
    dominance = frame[:, :, 0] - np.maximum(frame[:, :, 1], frame[:, :, 2])
    red = ink & (chroma >= RED_CHROMA_MIN) & (dominance >= RED_DOMINANCE_MIN)
    black = ink & ~red

    has_red = bool(np.count_nonzero(red) >= INK_MIN_PIXELS)
    bands = true_runs(black.sum(axis=1) >= ROW_MIN_INK_PX)
    row_bands = _drop_tall_bands(bands[HEADER_BAND_COUNT:])
    return AssistObservation(
        status="usable",
        ok_row_count=len(row_bands),
        has_red=has_red,
        reason="ok",
    )


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


def _save_locate_evidence(image, reason: str, debug_dir) -> None:
    """locate 시도가 '무엇을 보고 실패했는지' 를 디스크에 남긴다 (실패 무시).

    실패가 조용하면 오피스에 남는 게 콘솔 한 줄뿐이라 원인을 좁힐 수 없다(2026-08-12
    실측). VLM 이 거부했을 때는 crop 이 없으므로 입력 프레임 전체를 남긴다 - 그 프레임에
    패널이 실제로 보였는지(가려짐/스크롤/탭 전환)를 사람이 바로 판별할 수 있다.
    """
    target = debug_dir if debug_dir is not None else DEBUG_ARTIFACT_DIR
    try:
        stamp = f"assist_locate_fail_{reason}"
        if image is not None:
            image_path = target / f"{stamp}.jpg"
            save_debug_jpeg(image, image_path)
            print(f"[INFO] Assist locate 실패 프레임: {image_path}")
        save_debug_json(
            target / f"{stamp}.json",
            {
                "reason": reason,
                "image_size": list(image.size) if image is not None else None,
            },
        )
    except Exception as exc:
        print(f"[WARNING] Assist 실패 증거 저장 실패(무시): {exc}")


def locate_assist_panel(window, window_title: str, backend: str, image, *, debug_dir=None):
    """Assist 패널 위치를 찾아 panel_box(창-이미지 좌표계)를 돌려준다. 실패 시 None.

    watch 당 1회만 돈다 - 이후 폴링은 `read_assist_state` 가 이 박스로 crop 한 픽셀만
    본다(OCR 없음). VLM 은 영역만 답하고 정량 판정에는 관여하지 않는다(CLAUDE.md 규칙).
    """
    artifact_dir = debug_dir if debug_dir is not None else DEBUG_ARTIFACT_DIR
    try:
        result = analyze_window_target(
            window, window_title, backend, assist_panel_target(),
            debug_image_dir=artifact_dir,
            log_name=LOG_NAME,
            component_name=LOG_NAME,
            artifact_prefix="assist_panel",
            artifact_naming="service",
            # 한 run 의 Assist 산출물을 debug_dir 한 곳에 평평하게 모은다 - 모델 하위
            # 폴더로 갈리면 대조가 안 된다.
            artifact_model_subdir=False,
            image=image,
            timeout_sec=15.0,
        )
    except Exception as exc:
        print(f"[WARNING] Assist 패널 grounding 실패: {exc}")
        return None

    point = getattr(result, "point", None)
    if not point:
        print("[WARNING] Assist 패널을 찾지 못함 - 감지 비활성(cap 대기)")
        _save_locate_evidence(image, "no_panel_point", artifact_dir)
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
        save_debug_jpeg(panel, artifact_dir / "assist_panel_crop_region.jpg")
    except Exception as exc:
        print(f"[WARNING] Assist 패널 crop 실패: {exc}")
        return None
    print(f"[INFO] Assist 패널 확보: panel={panel_box}")
    return panel_box


__all__ = [
    "AssistObservation",
    "assist_panel_target",
    "locate_assist_panel",
    "read_assist_state",
]
