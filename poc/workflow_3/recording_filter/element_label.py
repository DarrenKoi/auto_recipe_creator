"""STAGE 2c - 클릭 지점의 UI 요소 라벨을 읽는다(OCR 우선, VLM 폴백).

timeline 의 element 필드는 지금까지 예약만 되어 있었다. workflow 로 변환하려면
"언제 어디를" 다음에 "무엇을" 이 필요하다. 라벨이 있어야 다른 장비에서도 그 요소를
다시 찾을 수 있기 때문이다(좌표는 창 위치가 달라 이식되지 않는다).

순서가 곧 비용 설계다 - 텍스트 버튼은 PaddleOCR 한 번으로 끝나고, 아이콘/라이브 영상
처럼 읽을 텍스트가 없을 때만 VLM 이 나선다. 전체 스크린샷 OCR 은 환각이 심하므로
반드시 작은 crop 에만 적용한다.
"""

from dataclasses import dataclass

from poc.workflow_3.util import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_spotting_prompt


@dataclass
class ElementLabel:
    """클릭 지점의 요소 라벨 1건."""

    text: str
    source: str        # "ocr" | "vlm" | "none"
    confidence: float


def crop_box_around(x, y, side, width, height) -> dict:
    """클릭 지점을 중심으로 한 정사각 crop 박스를 이미지 안으로 클램프해 만든다.

    업스트림 커서 좌표(Task 6 의 VLM 검출 등)는 프레임을 살짝 벗어나거나 아주
    멀리 벗어날 수 있다. 그런 입력에서도 항상 이미지 안의 유효한 박스
    (0 <= left < right <= width, 0 <= top < bottom <= height) 를 돌려준다 -
    상한과 하한을 모두 클램프해야 하며, 하한만 클램프하면 x/y 가 프레임보다
    한참 클 때 left 가 width 를 넘어선 채로 남아 right<=left 오류가 난다.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    half = max(1, int(side) // 2)

    def _clamp_range(center, size):
        left = int(center) - half
        right = int(center) + half
        left = max(0, min(left, size - 1))
        right = max(left + 1, min(right, size))
        return left, right

    left, right = _clamp_range(x, width)
    top, bottom = _clamp_range(y, height)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def pick_nearest_item(items, click_xy, crop_origin):
    """OCR 항목 중 클릭 지점에 가장 가까운 것을 고른다(crop 좌표계로 환산해 비교).

    box 키는 `bbox` 하나만 받는다 - `parse_spotting_items` (poc/workflow_3/vlm/
    ocr_spotting.py) 가 실제로 내보내는 키가 `bbox` 뿐이기 때문이다. 존재하지
    않는 `box` 키까지 받아주면 실제로는 없는 계약을 코드에 남기게 된다.
    """
    if not items:
        return None
    cx = int(click_xy[0]) - int(crop_origin[0])
    cy = int(click_xy[1]) - int(crop_origin[1])

    def _distance(item):
        box = item.get("bbox") or {}
        try:
            mx = (int(box["left"]) + int(box["right"])) / 2.0
            my = (int(box["top"]) + int(box["bottom"])) / 2.0
        except Exception:
            return float("inf")
        return ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5

    best = min(items, key=_distance)
    return best if _distance(best) != float("inf") else None


def element_label_prompt():
    """아이콘/비텍스트 요소용 VLM 프롬프트 (system, user) 를 만든다."""
    system_message = (
        "You are a GUI analyst. Look at the cropped screenshot region and identify "
        "the single UI element at its center. Answer with JSON only."
    )
    user_text = (
        "The center of this crop is where the user clicked. "
        "Identify that UI element concisely (e.g. \"OK button\", \"zoom in icon\", "
        "\"recipe list row\", \"live SEM image\"). "
        "Respond with JSON: {\"element\": \"<short name>\", \"confidence\": <0-1>}"
    )
    return system_message, user_text


def _read_with_ocr(crop_image, click_xy, crop_box, ocr_client):
    """crop 을 PaddleOCR Spotting 으로 읽어 클릭 지점 최근접 텍스트를 돌려준다."""
    crop_b64, _w, _h = encode_image_webp(crop_image, quality=90)
    system_msg, user_text = build_spotting_prompt()
    response = ocr_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    items = parse_spotting_items((response.text or "").strip())
    picked = pick_nearest_item(items, click_xy, (crop_box["left"], crop_box["top"]))
    if picked is None:
        return ""
    return str(picked.get("text") or "").strip()


def _describe_with_vlm(crop_image, vlm_client):
    """crop 중앙의 요소를 VLM 에 서술시킨다. (text, confidence) 반환."""
    crop_b64, _w, _h = encode_image_webp(crop_image, quality=90)
    system_msg, user_text = element_label_prompt()
    response = vlm_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    parsed = extract_json(response.text)
    text = str(parsed.get("element") or "").strip()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return text, confidence


def label_element(image, click_xy, settings, *, ocr_client, vlm_client) -> ElementLabel:
    """클릭 지점의 요소 라벨을 읽는다. 실패해도 던지지 않고 source="none" 을 준다."""
    width, height = image.size
    crop_box = crop_box_around(
        click_xy[0], click_xy[1], settings.element_crop_px, width, height
    )
    crop_image = image.crop(
        (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
    )

    if ocr_client is not None:
        try:
            text = _read_with_ocr(crop_image, click_xy, crop_box, ocr_client)
            if text:
                return ElementLabel(text=text, source="ocr", confidence=1.0)
        except Exception as exc:
            print(f"[WARNING] 요소 OCR 실패(VLM 폴백): {exc}")

    if vlm_client is not None:
        try:
            text, confidence = _describe_with_vlm(crop_image, vlm_client)
            if text:
                return ElementLabel(text=text, source="vlm", confidence=confidence)
        except Exception as exc:
            print(f"[WARNING] 요소 VLM 서술 실패: {exc}")

    return ElementLabel(text="", source="none", confidence=0.0)
