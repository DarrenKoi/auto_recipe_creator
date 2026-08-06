"""label_verify 순수 로직 스모크 테스트 (VLM 없이 Mac 에서 돈다).

  uv run python poc/workflow_3/vlm/test_label_verify.py
"""

import sys

from PIL import Image

from poc.workflow_3.vlm.label_verify import (
    crop_box_around_point,
    label_matches,
    tokens_from_text,
    upscale_for_ocr,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """단언 결과를 기록한다."""
    if condition:
        PASSED.append(name)
        print(f"[INFO] PASS {name}")
    else:
        FAILED.append(name)
        print(f"[ERROR] FAIL {name} {detail}")


def test_exact_label() -> None:
    """정확히 일치하는 라벨."""
    for label in ("Stop", "Queue", "PM"):
        check(f"exact_{label}", label_matches([label], label))


def test_case_and_punctuation_insensitive() -> None:
    """대소문자/기호는 무시한다 (OCR 이 붙여 오는 경우)."""
    check("lowercase", label_matches(["stop"], "Stop"))
    check("punctuation", label_matches(["Stop:"], "Stop"))
    check("spaced_token", label_matches(["[Queue]"], "Queue"))


def test_substring_only_for_long_labels() -> None:
    """3자 이상 라벨은 포함 매칭 허용, 2자 이하는 정확 일치만.

    'PM' 에 포함 매칭을 허용하면 'RPM'/'PMX' 같은 이웃 텍스트가 통과해버린다.
    """
    check("long_label_substring", label_matches(["Queuexx"], "Queue"))
    check("short_label_no_substring", not label_matches(["RPM"], "PM"))
    check("short_label_no_substring2", not label_matches(["PMX"], "PM"))
    check("short_label_exact_ok", label_matches(["PM"], "PM"))


def test_different_label_rejected() -> None:
    """다른 버튼 라벨은 매칭되지 않는다."""
    check("start_not_stop", not label_matches(["Start"], "Stop"))
    check("queue_not_stop", not label_matches(["Queue"], "Stop"))


def test_empty_inputs() -> None:
    """빈 입력은 안전하게 False."""
    check("empty_tokens", not label_matches([], "Stop"))
    check("empty_label", not label_matches(["Stop"], ""))
    check("blank_token", not label_matches([""], "Stop"))


def test_tokens_from_text() -> None:
    """줄바꿈/공백 기준 토큰화."""
    tokens = tokens_from_text("Stop  Queue\n PM \n\n")
    check("tokenize", tokens == ["Stop", "Queue", "PM"], f"tokens={tokens}")
    check("tokenize_empty", tokens_from_text("") == [])


def test_crop_box_geometry() -> None:
    """crop box 는 점을 감싸고 이미지 밖으로 나가지 않는다."""
    box = crop_box_around_point(
        {"x": 500, "y": 400}, 1920, 1080,
        left_ratio=0.035, right_ratio=0.035, half_height_ratio=0.018,
    )
    check(
        "box_contains_point",
        box["left"] < 500 < box["right"] and box["top"] < 400 < box["bottom"],
        f"box={box}",
    )
    edge = crop_box_around_point(
        {"x": 2, "y": 1}, 1920, 1080,
        left_ratio=0.035, right_ratio=0.035, half_height_ratio=0.018,
    )
    check("box_clamped", edge["left"] == 0 and edge["top"] == 0, f"box={edge}")

    far = crop_box_around_point(
        {"x": 1919, "y": 1079}, 1920, 1080,
        left_ratio=0.035, right_ratio=0.035, half_height_ratio=0.018,
    )
    check("box_clamped_far", far["right"] == 1920 and far["bottom"] == 1080, f"box={far}")


def test_upscale() -> None:
    """작은 crop 만 확대하고, max_upscale 을 넘지 않는다."""
    small = Image.new("RGB", (100, 20))
    out, scale = upscale_for_ocr(small, 72, 6.0)
    check("upscaled", out.size[1] >= 72 and scale > 1.0, f"size={out.size} scale={scale}")

    tiny = Image.new("RGB", (10, 4))
    out2, scale2 = upscale_for_ocr(tiny, 72, 6.0)
    check("upscale_capped", scale2 <= 6.0, f"scale={scale2}")

    big = Image.new("RGB", (400, 200))
    out3, scale3 = upscale_for_ocr(big, 72, 6.0)
    check("no_upscale_when_big", scale3 == 1.0 and out3.size == (400, 200), f"scale={scale3}")


def main() -> int:
    """모든 테스트를 돌리고 실패 수를 반환한다."""
    test_exact_label()
    test_case_and_punctuation_insensitive()
    test_substring_only_for_long_labels()
    test_different_label_rejected()
    test_empty_inputs()
    test_tokens_from_text()
    test_crop_box_geometry()
    test_upscale()

    print("")
    print(f"[INFO] 통과 {len(PASSED)} / 실패 {len(FAILED)}")
    if FAILED:
        print(f"[ERROR] 실패 목록: {FAILED}")
    return len(FAILED)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
