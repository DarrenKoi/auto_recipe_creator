"""ensemble proposer(C2 Scharr·C3 directional·RRF) 합성 단위테스트 — Mac 실행 가능."""
import numpy as np
import cv2

from poc.workflow_2 import ensemble_proposer as ep


def _square_img(size=200, box=(70, 70, 60, 60), bg=110, edge=230):
    """배경 위 사각 윤곽 하나 — gradient/edge 채널이 윤곽을 잡는지 본다."""
    img = np.full((size, size), bg, np.uint8)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 2)
    return img


def test_scharr_edges_density_matched_to_canny():
    img = _square_img()
    canny = cv2.Canny(cv2.GaussianBlur(img, (0, 0), 1.0), 60, 160)
    r_c1 = float((canny > 0).mean())
    edges = ep._scharr_edges(img, r_c1)
    assert edges.dtype == np.uint8 and edges.shape == img.shape
    r_c2 = float((edges > 0).mean())
    # 밀도가 C1 근처(±5%p 절대)로 맞춰져야 동등 비교 가능.
    assert abs(r_c2 - r_c1) <= 0.05, (r_c1, r_c2)


def test_scharr_edges_clamp_low_density():
    # 거의 균일한 이미지 → r_c1 매우 작아도 하한 3% clamp 로 edge 가 생긴다.
    img = np.full((200, 200), 110, np.uint8)
    edges = ep._scharr_edges(img, 0.0)
    assert float((edges > 0).mean()) > 0.0


def test_orientation_bin_edges_shape_and_count():
    img = _square_img()
    bins = ep._orientation_bin_edges(img, n_bins=8)
    assert len(bins) == 8
    assert all(b.shape == img.shape and b.dtype == np.uint8 for b in bins)
    # 사각 윤곽 → 가로/세로 방향 bin 에 edge 가 몰린다(전체 edge 합 > 0).
    assert sum(int((b > 0).sum()) for b in bins) > 0


def test_directional_chamfer_peak_at_true_location():
    # template = 사각 윤곽 crop, frame 에 같은 윤곽을 (dx,dy) 평행이동 배치 →
    # directional chamfer score map 의 peak 가 그 위치 근처여야 한다.
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    smap, (tw, th) = ep._directional_chamfer_score_map(tpl, frame, scale=1.0, n_bins=8)
    assert smap is not None
    y, x = np.unravel_index(int(np.argmax(smap)), smap.shape)
    cx, cy = x + tw // 2, y + th // 2
    # true 중심 ≈ (120+30, 90+30) = (150,120). 허용 8px.
    assert abs(cx - 150) <= 8 and abs(cy - 120) <= 8, (cx, cy)
