"""ensemble proposer(C2 Scharr·C3 directional·RRF) 합성 단위테스트 — Mac 실행 가능."""
import numpy as np
import cv2

from poc.workflow_3.vision import ensemble_proposer as ep


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


def test_rrf_fuse_is_scale_free_rank_based():
    # 채널 A 점수 스케일이 100배 커도 RRF 는 순위만 보므로 결과가 스케일에 불변.
    A = [ep._Cand(xy=(10, 10), score=900.0), ep._Cand(xy=(50, 50), score=100.0)]
    B = [ep._Cand(xy=(10, 10), score=9.0), ep._Cand(xy=(80, 80), score=1.0)]
    fused = ep._rrf_fuse([A, B], k0=10, match_radius=5, top_n=3)
    # (10,10) 은 두 채널 모두 rank1 → 최상위.
    assert fused[0].xy == (10, 10)
    A2 = [ep._Cand(xy=(10, 10), score=9.0), ep._Cand(xy=(50, 50), score=1.0)]
    fused2 = ep._rrf_fuse([A2, B], k0=10, match_radius=5, top_n=3)
    assert fused2[0].xy == (10, 10)   # 점수 스케일 바뀌어도 순위 동일.


def test_rrf_fuse_preserves_source_scale():
    # 융합 대표 후보는 source 후보의 scale 을 보존해야 — production rescore/ORB 가 scale 사용.
    A = [ep._Cand(xy=(10, 10), score=900.0, scale=0.6),
         ep._Cand(xy=(50, 50), score=100.0, scale=0.85)]
    B = [ep._Cand(xy=(80, 80), score=9.0, scale=1.2)]
    fused = ep._rrf_fuse([A, B], k0=10, match_radius=5, top_n=3)
    by_xy = {c.xy: c.scale for c in fused}
    assert by_xy[(10, 10)] == 0.6
    assert by_xy[(50, 50)] == 0.85
    assert by_xy[(80, 80)] == 1.2


def test_rrf_fuse_representative_scale_follows_best_member():
    # 같은 위치를 두 채널이 다른 scale·chamfer 로 제안 → 대표는 더 높은 chamfer 의 scale·xy.
    A = [ep._Cand(xy=(10, 10), score=5.0, scale=0.6)]
    B = [ep._Cand(xy=(12, 12), score=9.0, scale=1.2)]   # match_radius 안, chamfer 더 높음
    fused = ep._rrf_fuse([A, B], k0=10, match_radius=5, top_n=1)
    assert fused[0].xy == (12, 12)
    assert fused[0].scale == 1.2


def test_ensemble_candidates_returns_topn_and_shadow():
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    res = ep.compute_ensemble_candidates(tpl, frame, top_n=8, shadow_n=24)
    assert len(res.fused) <= 24 and len(res.fused) >= 1
    assert res.top_n_count == 8
    assert set(res.solo.keys()) == {"canny", "scharr", "orient"}
    # 진짜 위치(≈150,120)가 fused 후보 중에 있다.
    assert any(abs(c.xy[0] - 150) <= 8 and abs(c.xy[1] - 120) <= 8 for c in res.fused)


def _reference_directional(template_gray, frame_gray, scale, n_bins=8):
    """원본(중복 연산판) directional chamfer 의 인라인 재현 — 리팩터 동치(점수 불변) 가드."""
    import cv2 as _cv2
    import numpy as _np
    from poc.workflow_3.vision.align_key_matcher import _scaled_edges, DT_TAU_PX
    t_bins = ep._orientation_bin_edges(template_gray, n_bins)
    f_bins = ep._orientation_bin_edges(frame_gray, n_bins)
    num, den, out_size = None, 0.0, None
    for tb, fb in zip(t_bins, f_bins):
        tb_s = _scaled_edges(tb, scale)
        th, tw = tb_s.shape[:2]
        fh, fw = fb.shape[:2]
        if th >= fh or tw >= fw:
            return None, (tw, th)
        mask = (tb_s > 0).astype(_np.float32)
        cnt = float(mask.sum())
        if cnt <= 0:
            continue
        f_dt = _cv2.distanceTransform(_cv2.bitwise_not(fb), _cv2.DIST_L2, 5).astype(_np.float32)
        mean_dt = _cv2.matchTemplate(f_dt, mask, _cv2.TM_CCORR) / cnt
        num = mean_dt * cnt if num is None else num + mean_dt * cnt
        den += cnt
        out_size = (tw, th)
    if num is None or den <= 0:
        return None, (out_size or (0, 0))
    return _np.exp(-(num / den) / DT_TAU_PX).astype(_np.float32), out_size


def test_directional_refactor_preserves_score_map():
    # 리팩터(중복 DT 제거)가 원본과 *수치 동일* 한 score map 을 내는지 — 여러 scale 에서.
    import numpy as np
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    for scale in (0.6, 0.85, 1.0):
        ref, ref_sz = _reference_directional(tpl, frame, scale)
        new, new_sz = ep._directional_chamfer_score_map(tpl, frame, scale=scale)
        assert (ref is None) == (new is None)
        if ref is not None:
            assert ref_sz == new_sz
            assert np.allclose(ref, new, atol=1e-6), scale


def test_directional_context_shapes():
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    t_bins, f_dts = ep._directional_context(tpl, frame, n_bins=8)
    assert len(t_bins) == 8 and len(f_dts) == 8
    # frame DT 들은 frame 크기, float32.
    assert all(d.shape == frame.shape and d.dtype == np.dtype("float32") for d in f_dts)
