"""cond.txt 좌표로 white box 테두리 / crosshair 선을 inpaint 해 지운다.

왜 지우나
--------
- rcp 의 **white box** 는 웨이퍼 위에 그려 넣은 주석(unique-area 표시)일 뿐
  실제 영상 내용이 아니다 ([[project_rcp_white_box_unique_area]]).
- msr 의 **crosshair** 는 CV matcher 의 distractor 라 매칭 전에 제거해야 한다
  ([[project_msr_crosshair_cv_distractor]]).

핵심 원칙
--------
**그려진 '선'만 마스크한다. white box 안쪽은 실제 내용이므로 절대 채우지 않는다.**
박스는 테두리 strip 만, crosshair 는 FOV 전체를 가로지르는 두 선만 마스크하고
cv2.inpaint(TELEA) 로 주변 픽셀로 메운다.

좌표계
------
cond.txt 의 cursor 좌표는 Pixel 의 **10배 oversample** 프레임이다 →
이미지 px = cursor / 10 (OVERSAMPLE). 1024-px 이미지 샘플이 아직 없어
이 비율은 오피스 실데이터에서 검증 필요 ([[project_align_cond_files_and_coords]]).
"""

import cv2
import numpy as np

from poc.workflow_2.cond_file import CondInfo

# cursor 좌표 → 이미지 px 축소 비율 (cursor frame = Pixel × OVERSAMPLE).
OVERSAMPLE = 10
# 그려진 주석 선의 반폭(px). 실제 선 두께에 맞춰 오피스에서 조정.
DEFAULT_THICKNESS = 3
# inpaint 반경(px).
DEFAULT_INPAINT_RADIUS = 3


def cursor_to_image(xy, oversample=OVERSAMPLE):
    """cursor 프레임 좌표를 이미지 px 로 변환한다 (x/oversample, y/oversample)."""
    x, y = xy
    return x / float(oversample), y / float(oversample)


def build_removal_mask(
    shape_hw,
    *,
    box_ltrb=None,
    crosshair_xy=None,
    oversample=OVERSAMPLE,
    thickness=DEFAULT_THICKNESS,
):
    """지울 선(박스 테두리 + crosshair)만 255 로 칠한 uint8 마스크를 만든다.

    ``shape_hw`` 는 (height, width). box/crosshair 는 cursor 프레임 raw 좌표이며
    내부에서 oversample 로 나눠 이미지 px 로 변환한다.
    """
    h, w = shape_hw[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    t = max(1, int(thickness))

    if box_ltrb is not None:
        l, top = cursor_to_image(box_ltrb[:2], oversample)
        r, b = cursor_to_image(box_ltrb[2:], oversample)
        # 테두리만(채우지 않음): thickness 두께의 사각형 outline.
        cv2.rectangle(mask, (round(l), round(top)), (round(r), round(b)), 255, t)

    if crosshair_xy is not None:
        cx, cy = cursor_to_image(crosshair_xy, oversample)
        cxi, cyi = round(cx), round(cy)
        cv2.line(mask, (cxi, 0), (cxi, h - 1), 255, t)     # 세로선 (높이 전체)
        cv2.line(mask, (0, cyi), (w - 1, cyi), 255, t)     # 가로선 (폭 전체)

    return mask


def clean_image(
    image,
    cond: CondInfo,
    *,
    oversample=OVERSAMPLE,
    thickness=DEFAULT_THICKNESS,
    inpaint_radius=DEFAULT_INPAINT_RADIUS,
):
    """CondInfo 의 box/crosshair 선을 inpaint 로 지운 이미지를 돌려준다.

    지울 게 없으면(box·crosshair 모두 None) 원본을 그대로 반환한다.
    """
    mask = build_removal_mask(
        image.shape[:2],
        box_ltrb=cond.box_ltrb,
        crosshair_xy=cond.crosshair_xy,
        oversample=oversample,
        thickness=thickness,
    )
    if not mask.any():
        return image
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def main():
    """합성 데모: crosshair·box 를 그린 뒤 cond 좌표로 지워 before/after 저장."""
    import os

    out_dir = os.path.dirname(__file__)
    img = np.full((512, 512), 110, dtype=np.uint8)
    rng = np.random.default_rng(0)
    img = cv2.add(img, rng.integers(-15, 15, img.shape, dtype=np.int16).astype(np.int16).clip(0).astype(np.uint8))
    # 합성 주석: box 테두리 + crosshair 를 흰색으로 그려 둔다.
    cv2.rectangle(img, (160, 160), (352, 352), 255, 3)
    cv2.line(img, (210, 0), (210, 511), 255, 3)
    cv2.line(img, (0, 256), (511, 256), 255, 3)

    cond = CondInfo(scope="OM", pixel=(512, 512),
                    box_ltrb=(1600, 1600, 3520, 3520), crosshair_xy=(2100, 2560))
    cleaned = clean_image(img, cond)
    before = os.path.join(out_dir, "clean_align_before.jpg")
    after = os.path.join(out_dir, "clean_align_after.jpg")
    cv2.imwrite(before, img)
    cv2.imwrite(after, cleaned)
    print(f"[INFO] before: {before}")
    print(f"[INFO] after : {after}")


if __name__ == "__main__":
    main()
