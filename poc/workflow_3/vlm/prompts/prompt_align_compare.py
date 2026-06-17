"""Align-point 2-이미지 비교(fallback) VLM 프롬프트 빌더 - workflow_2 A/B 실험용.

CV 매처의 점수면이 평평해 1차 매칭 신뢰도가 낮을 때(=fallback)만 호출하는 보조 단계.
역할 분담 규칙(2026-05-25)을 지킨다: VLM 은 좌표를 만들지 않는다 - 이미 CV 가 찾아
번호를 붙인 corner 후보들 중 '등록 align key 와 같은 지점'을 고르기만 한다.

  image 1 = 등록된 align key(from_rcp), align point 가 마킹된 상태.
  image 2 = 라이브 SEM box, CV 가 검출한 corner 후보들에 번호(1..N)가 그려진 상태.

VLM 은 image 1 의 마킹 지점 '모양'을 먼저 서술(reasoning scaffold)한 뒤, image 2 의
번호 후보 중 같은 feature 의 index 를 반환한다. 확신이 없으면 -1(거부) - 순수 CV 폴백.

도메인 prior(엔지니어 관측): align point 는 두 edge 가 교차하는 box 의 corner 이며,
4분면 중 아래쪽(Q3 좌하 / Q4 우하)에 위치하는 경향이 있다. 단, hard filter 가 아니라
동점일 때만 쓰는 soft tiebreaker 로 준다(Q1 에 있는 recipe 를 silent fail 시키지 않기 위해).

2-이미지 입력이라 ui-venus(단일 요소 grounding) 가 아니라 직접 게이트웨이의
Qwen3-VL-30B-Instruct(native 2-image) 를 대상으로 한다.
"""


def build_align_compare_prompt(
    n_candidates: int,
    box_width: int,
    box_height: int,
) -> tuple[str, str]:
    """align key vs 라이브 SEM box corner 후보 비교 프롬프트를 구성한다.

    Args:
        n_candidates: image 2 에 그려진 번호 후보 corner 개수(유효 index 는 1..N).
        box_width:    image 2(라이브 SEM box crop)의 픽셀 폭 - 4분면 추론 보조용.
        box_height:   image 2 의 픽셀 높이.

    Returns:
        (system_message, user_message) 튜플.
    """
    system_message = (
        "You compare two grayscale CD-SEM (scanning electron microscope) metrology "
        "images and select a matching point. Return STRICT JSON only, no markdown.\n"
        "\n"
        "ROLE: You do NOT produce pixel coordinates. Classical computer vision has "
        "already found candidate corners in the live image and drawn a NUMBERED marker "
        "on each. Your job is only to choose WHICH numbered candidate is the same "
        "physical feature as the marked align point in the reference image.\n"
        "\n"
        "WHAT AN ALIGN POINT IS: the alignment point sits where TWO edge lines cross - "
        "typically a CORNER of one of the large device boxes (an L-, T-, or +-shaped "
        "junction of two bright/dark edges), not a flat area and not the middle of an edge.\n"
        "\n"
        "QUADRANT PRIOR (soft, tiebreaker ONLY): treat the live SEM box as four "
        "quadrants - Q1 top-right, Q2 top-left, Q3 bottom-left, Q4 bottom-right. Align "
        "points are most often in the lower half (Q3 or Q4). Use this ONLY to break a "
        "tie between otherwise equally good candidates; never reject a clearly correct "
        "upper-quadrant candidate because of it."
    )

    user_message = (
        "IMAGE 1 (reference): the registered align key. Its align point is marked.\n"
        "IMAGE 2 (live): the current SEM box, "
        f"{box_width}x{box_height} pixels, with {n_candidates} candidate corners "
        f"labelled 1 to {n_candidates}. Each numbered marker sits on a corner where two "
        "edges cross.\n"
        "\n"
        "Steps:\n"
        "1. Describe the marked feature in IMAGE 1: the junction shape (L / T / +), which "
        "two edges meet, and the local box it belongs to.\n"
        "2. Find the candidate in IMAGE 2 that is the SAME junction (same shape and same "
        "position relative to its box). Compare structure, not brightness - the live "
        "image may differ in contrast and process appearance.\n"
        "3. If two candidates are equally plausible, prefer the one in Q3/Q4 (lower half).\n"
        "4. If NONE of the numbered candidates is a confident structural match, return "
        "match_index = -1.\n"
        "\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "key_description": "shape of the marked junction in image 1 (e.g. \'L-corner, '
        "top-right of lower-left box')\",\n"
        '  "match_index": 0,\n'
        '  "quadrant": "Q3",\n'
        '  "confidence": "high",\n'
        '  "evidence": "why this candidate matches the reference junction"\n'
        "}\n"
        "Rules:\n"
        f"- match_index is an integer from 1 to {n_candidates}, or -1 if no confident match.\n"
        '- quadrant is one of "Q1","Q2","Q3","Q4" for the chosen candidate (use "" if -1).\n'
        '- confidence is one of "high","medium","low".\n'
        "- Output ONLY the JSON object. No extra keys, no explanation outside the JSON."
    )

    return system_message, user_message


__all__ = ["build_align_compare_prompt"]
