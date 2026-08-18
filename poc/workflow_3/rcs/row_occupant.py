"""List 행의 점유자 컬럼을 읽어 점유를 3-상태로 판별한다.

점유는 참/거짓이 아니라 3-상태다. "모른다" 를 "비어 있다" 로 접으면, 화면 공유(view-only)
세션에서 먹지 않는 클릭을 하고도 '보정 완료' 로 보고하는 조용한 오보가 된다.
`correct_align_fail_auto` 는 open-loop 라 클릭이 반영됐는지 화면으로 되읽지 않으므로,
반영 여부를 아는 유일한 방법이 사전에 세션 성격을 아는 것뿐이다.

`read_occupancy` 가 crop -> OCR -> 분류의 전체 흐름을 소유하고, OCR 호출만 주입받는다
(`share_request` 와 같은 형태 - 모듈은 순수하게 두고 배선은 orchestrator 가 한다).

**이 판독은 자기 crop 을 쓴다.** `tool_row_verify` 의 행 strip 을 넓혀
한 번에 읽으면 안 된다: 그쪽 `_looks_like_tool_id` 가 점유자 ID(KIM0234 등)를 장비 ID 로
오인해 `classify_tokens` 가 `mismatch` 를 내고, `accepts()` 는 lenient 에서도 mismatch 를
거부하므로 **정상 행의 클릭이 거부된다**. 지금은 무해한 `unreadable`(lenient 통과)이
파괴적인 `mismatch`(무조건 거부)로 승격되는 셈이고, 하필 점유자 컬럼이 채워진 행에서만,
즉 이 기능이 겨냥한 바로 그 케이스에서 터진다. 게다가 `PointTextRead.tokens` 는 좌표를
버리므로 한 crop 으로 읽으면 사후에 컬럼별로 나눌 수도 없다.
"""

from poc.workflow_3.vlm.label_verify import crop_box_around_point, looks_like_id_token

OCCUPIED_BY_OTHER = "occupied_by_other"
FREE = "free"
UNKNOWN = "unknown"

# 점유자 ID 길이 범위. 사번/계정 형태를 받되 한두 글자 잡음은 배제한다.
OCCUPANT_MIN_LEN = 4
OCCUPANT_MAX_LEN = 16

# 점유자 컬럼 crop 기하 - 장비 ID 점 기준 오른쪽으로 뻗는다.
# 오피스 캘리브레이션 대상: 첫 실행에서 crop 산출물을 보고 조정한다. 어긋나면 대부분의
# 사이클이 UNKNOWN 으로 떨어진다(안전하지만 알림이 시끄러워진다).
OCCUPANT_LEFT_RATIO = 0.02
OCCUPANT_RIGHT_RATIO = 0.30
OCCUPANT_HALF_HEIGHT_RATIO = 0.010
OCCUPANT_MIN_HALF_HEIGHT_PX = 8


def looks_like_occupant(token: str) -> bool:
    """토큰이 사람 ID 모양인지 (판정은 `label_verify.looks_like_id_token` 공유).

    장비 ID 와 사번은 모양이 같고 자릿수만 다르므로 길이 범위만 다르게 준다.
    """
    return looks_like_id_token(
        token, min_len=OCCUPANT_MIN_LEN, max_len=OCCUPANT_MAX_LEN
    )


def classify_occupancy(read_ok: bool, tokens) -> str:
    """OCR 성공 여부와 토큰으로 점유 3-상태를 정한다.

    read_ok=False 는 UNKNOWN 이다. 읽지 못한 것을 '비어 있음' 으로 접으면 view-only
    세션이 보정 가능 세션으로 오인된다 - 이 구분이 이 모듈의 존재 이유다.
    """
    if not read_ok:
        return UNKNOWN
    if any(looks_like_occupant(token) for token in (tokens or [])):
        return OCCUPIED_BY_OTHER
    return FREE


def build_occupant_box(row_point: dict, image_width: int, image_height: int) -> dict:
    """장비 ID 점 기준으로 점유자 컬럼 crop box 를 만든다 (경계 clamp 포함).

    좌표를 새로 만들지 않는다 - 이미 확정된 행 좌표에서 오른쪽으로 뻗을 뿐이다
    (프로젝트 규칙: 좌표는 VLM 이 정하고 OCR 은 확인만 한다).
    """
    return crop_box_around_point(
        row_point,
        image_width,
        image_height,
        left_ratio=OCCUPANT_LEFT_RATIO,
        right_ratio=OCCUPANT_RIGHT_RATIO,
        half_height_ratio=OCCUPANT_HALF_HEIGHT_RATIO,
        min_half_height_px=OCCUPANT_MIN_HALF_HEIGHT_PX,
    )


def read_occupancy(list_image, row_point, *, read_tokens_fn) -> str:
    """List 캡처와 행 좌표로 점유 3-상태를 판정한다 (crop → OCR → 분류).

    OCR 호출은 `read_tokens_fn(image, box) -> list[str] | None` 으로 주입받는다.
    None 은 판독 실패를 뜻하며 UNKNOWN 이 된다(빈 리스트 = 읽었는데 비어 있음 = FREE
    와 구분해야 한다 - 그 구분이 이 모듈의 존재 이유다).

    입력이 없거나 어떤 예외가 나도 UNKNOWN 으로 흡수한다. 판독 불가를 '비어 있음'으로
    접으면 view-only 세션이 보정 가능 세션으로 오인된다.
    """
    if list_image is None or not row_point:
        return UNKNOWN
    try:
        box = build_occupant_box(row_point, list_image.width, list_image.height)
        tokens = read_tokens_fn(list_image, box)
    except Exception as exc:
        print(f"[WARNING] 점유자 컬럼 판독 실패(점유 미상으로 진행): {exc}")
        return UNKNOWN
    if tokens is None:
        return UNKNOWN
    return classify_occupancy(True, tokens)


__all__ = [
    "FREE",
    "OCCUPIED_BY_OTHER",
    "UNKNOWN",
    "build_occupant_box",
    "classify_occupancy",
    "looks_like_occupant",
    "read_occupancy",
]
