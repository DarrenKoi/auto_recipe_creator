"""PPT 슬라이드 이미지 → 레이아웃 박스 추출 샘플 코드.

이 스크립트는 텍스트/차트 등의 위치 박스만 추출한다.
원본 이미지 위에 overlay 박스를 그린 새 이미지를 저장하고,
레이아웃 좌표를 JSON 으로 함께 저장한다.

사용법:
    uv run python poc/work2/sample_code/ppt_slide_box_locator.py

같은 폴더에 있는 .jpg 또는 .png 이미지를 자동으로 읽는다.
"""

import sys

from ppt_slide_reader import (
    IMAGE_EXTENSIONS,
    OCR_SERVICE,
    OCR_URL,
    SAMPLE_DIR,
    VLM_SERVICE,
    VLM_URL,
    analyze_slide_layout,
    save_layout_result,
    save_overlay_image,
)


def print_layout_summary(image_name: str, result: dict) -> None:
    """레이아웃 박스 결과를 간단히 출력한다."""
    layout_elements = result.get("_layout_elements") or []

    print(f"\n{'─' * 60}")
    print(f"  레이아웃 결과: {image_name}")
    print(f"{'─' * 60}")

    if not layout_elements:
        print("[WARNING] 추출된 레이아웃 박스가 없습니다.")
        return

    print(f"[INFO] 레이아웃 박스: {len(layout_elements)}개")
    for item in layout_elements:
        bbox = item.get("bbox_pixels") or item.get("bbox_1000") or {}
        label = item.get("label") or "-"
        print(
            f"  [{item.get('index', '?')}] {item.get('type', 'other'):<10} "
            f"({bbox.get('left', '?')}, {bbox.get('top', '?')}) - "
            f"({bbox.get('right', '?')}, {bbox.get('bottom', '?')})  "
            f"label={label}"
        )


def main() -> None:
    """메인 실행."""
    image_paths = sorted(
        path for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("[INFO] PPT 슬라이드 박스 로케이터 시작")
    print(f"[INFO] OCR: {OCR_SERVICE} ({OCR_URL})")
    print(f"[INFO] VLM: {VLM_SERVICE} ({VLM_URL})")
    print(f"[INFO] 대상 이미지: {len(image_paths)}개")

    if not image_paths:
        print("[ERROR] 분석할 이미지가 없습니다.")
        print("[INFO] 이 폴더에 .jpg 또는 .png 이미지를 넣어주세요.")
        print(f"[INFO] 경로: {SAMPLE_DIR}")
        sys.exit(1)

    success_count = 0
    for image_path in image_paths:
        result = analyze_slide_layout(image_path)
        if result is None:
            continue

        print_layout_summary(image_path.name, result)
        save_layout_result(image_path, result)
        save_overlay_image(image_path, result)
        success_count += 1

    print(f"\n{'=' * 60}")
    print(f"[INFO] 레이아웃 분석 완료: {success_count}/{len(image_paths)}개 성공")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
