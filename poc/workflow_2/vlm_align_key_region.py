"""Perception probe — large VLM 이 *두 이미지를 함께 보고* "등록 align key 가 현재 SEM
화면의 어디에 있는지" region 을 짚을 수 있는지, 그리고 그 region 이 CV matcher 를
실제로 도와주는지를 사람이 overlay 로 평가하기 위한 throwaway 프로브.

`vlm_align_key_box.py` 와의 차이
--------------------------------
`vlm_align_key_box.py` 는 *단일 이미지*를 small grounding 모델(ui-venus)에 하나씩 보내
"이 그림 안의 align key 박스" 를 묻는다. 본 프로브는 *네이티브 멀티이미지*로 reference
(등록 key) + scene(현재 SEM)을 **함께** large VLM(Qwen3-VL-30B-A3B)에 보내, scene 안에서
reference 와 같은 구조가 있는 region 을 cross-image 로 짚게 한다. 이는 CV 단독 reranking 이
죽은 뒤의 escalation = "VLM 이 영역을 좁히고 CV 가 좌표" 가설을 직접 검증한다
([[project_matcher_flat_chamfer_distinctiveness]], [[project_vlm_multi_image_capability]]).

검증 방식
---------
  modality 짝짓기 — reference 와 scene 의 modality 를 *반드시 일치*시킨다(cross-modality 금지):
    from_msr 이미지를 scale-bar OCR 로 OM/SEM 분류 → SEM reference(IMAP0002)는 SEM scene 과만,
    OM reference(IMAP0001)는 OM scene 과만 짝짓는다. ([[project_align_fail_modality_om_vs_sem]])
    REFERENCE_MODALITY=both 면 두 짝을 모두 돌려 비교한다. scale-bar OCR 불가 시 current_sem 을
    SEM 으로 가정하는 폴백만 허용(OM 은 분류 없이 시험하지 않음).
  1) VLM: scene-relative region bbox(relative_1000) + confidence + evidence (좌표는 *비권위*).
  2) CV: reference 로 template 을 만들어 compute_align_key_score 를 두 번 호출 —
         (a) full-frame, (b) roi_hint=VLM-region — 점수/판정 delta 를 비교.
VLM 좌표를 매칭에 직접 쓰지 않는다. region 이 matcher 의 distractor 를 줄여 점수를
끌어올리는지(= escalation 이 먹히는지)만 본다.

실행
----
    uv run python poc/workflow_2/vlm_align_key_region.py

오피스(회사망)에서 실행해야 게이트웨이에 도달하고 실제 align_images 자산을 읽는다. Mac
에서는 네트워크 ERROR 가 정상이며, 자산 해석/인코딩/CV 경로는 그대로 점검할 수 있다.
게이트웨이 api_base/api_key 는 probe_multi_image_vlm.py 상단 상수를 단일 소스로 재사용한다.
"""

import os

# OpenBLAS/OMP 스레드 제한 — 반드시 numpy/cv2 import *이전*. Windows 다중코어에서
# "Memory allocation failed after 10 retries" 회피 ([[project_vlm_multi_image_capability]]).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import (
    iter_msr_images,
    iter_recipe_dirs,
    load_gray,
    resolve_assets,
    resolve_assets_auto,
)
from poc.workflow_3.vision.align_key_matcher import build_template, compute_align_key_score
from poc.workflow_3.vision.align_point_correction import SCALE_BAR_OCR_SERVICE, _ocr_scale_bar
from poc.workflow_2.probe_multi_image_vlm import (
    LARGE_VLM_API_BASE,
    LARGE_VLM_API_KEY,
    encode_under_limit,
)
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_3.util.time_utils import format_elapsed_ms, make_timestamp_tag

# ====================================================================
# 설정 (CLI 인자 없음 — 상수로만).
# ====================================================================

# 멀티이미지 native PASS 모델 (probe_multi_image_vlm 결과, 2026-06-02).
PERCEPTION_MODEL = "Qwen3-VL-30B-A3B-Instruct"
ENCODE_FMT = "webp"          # 게이트웨이 WebP PASS, payload 작음.
TIMEOUT_SEC = 90.0
MAX_TOKENS = 512             # JSON 한 덩어리 + 여유.
REQUEST_DELAY_SEC = 1.0
LOG_NAME = "vlm_align_key_region"
OUTPUT_ROOT = DEBUG_IMAGE_DIR / LOG_NAME

# recipe 폴더 선택: "random"(무작위 샘플링) | "latest"(가장 최근). 환경변수로 변경.
# full override(ALIGN_EQP_ID + CLASS + RECIPE)가 있으면 항상 그게 우선이며 모드를 무시한다
# (특정 recipe 를 고정해 재현하고 싶을 때 사용).
RECIPE_SELECT = os.getenv("PROBE_RECIPE_SELECT", "random").strip().lower()
# reference modality: "both"(OM·SEM 짝 모두) | "sem"(IMAP0002) | "om"(IMAP0001) | "auto"(scene modality).
# 각 reference 는 same-modality scene 과만 짝지어진다. modality: [[project_align_fail_modality_om_vs_sem]].
REFERENCE_MODALITY = os.getenv("PROBE_REFERENCE", "both").strip().lower()
# scale-bar OCR on/off. PaddleOCR-VL(Flask proxy) 필요. *scene modality 분류에 사용* — 끄면 OM scene 을
# 분류할 수 없어 SEM(current_sem 폴백)만 시험된다. Mac/오프라인이면 자동으로 SEM 폴백.
ENABLE_SCALE_BAR_HINT = os.getenv("PROBE_SCALE_BAR_OCR", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

SYSTEM_MESSAGE = (
    "You are a semiconductor metrology vision assistant. You are given TWO grayscale images.\n"
    "IMAGE-1 is a REFERENCE crop of a registered alignment key (a deliberately fabricated, "
    "high-contrast fiducial: nested/concentric boxes (box-in-box), a cross/plus, an L/corner "
    "mark, or an asymmetric dot cluster).\n"
    "IMAGE-2 is the CURRENT SEM field-of-view captured at align-fail time. In IMAGE-2 the same "
    "key may be drifted, shifted, partially missing, or lower-contrast than the reference.\n"
    "Your task: find WHERE IN IMAGE-2 the same alignment-key structure appears. Match by "
    "geometric structure, NOT by absolute position and NOT by random wafer texture or charging "
    "gradients. Return strict JSON only; if the structure is not clearly present, say so rather "
    "than guessing on texture."
)

USER_MESSAGE = (
    "Return JSON with this exact schema:\n"
    "{\n"
    '  "found": true,\n'
    '  "coord_system": "relative_1000_of_IMAGE2",\n'
    '  "region_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
    '  "confidence": 0.0,\n'
    '  "evidence": "short string: which structure in IMAGE-1 you matched, and where in IMAGE-2"\n'
    "}\n"
    "region_bbox are coordinates in IMAGE-2 on a 0-1000 scale, tightly enclosing the matched "
    "alignment-key region (leave a little margin). If not clearly found, set found=false and "
    "region_bbox=null."
)


# ====================================================================
# 자산 선택.
# ====================================================================


def _full_override_present() -> bool:
    """ALIGN_EQP_ID + (class+recipe ≥ 2단계) 완전 override 가 잡혀 있는지.

    resolve_assets_auto 의 판정과 동일하게 본다 — 완전 override 면 폴더 선택 모드를
    무시하고 그 recipe 를 고정한다.
    """
    eqp = os.getenv("ALIGN_EQP_ID", "").strip()
    cls = os.getenv("ALIGN_CLASS_NAME", "").strip()
    rcp = os.getenv("ALIGN_RECIPE_NAME", "").strip()
    rel = [p for seg in (cls, rcp)
           for p in seg.replace("\\", "/").strip("/").split("/") if p]
    return bool(eqp and len(rel) >= 2)


def _select_assets():
    """RECIPE_SELECT 에 따라 recipe 폴더를 고른다(override 우선)."""
    if not _full_override_present() and RECIPE_SELECT == "random":
        dirs = iter_recipe_dirs(ALIGN_IMAGES_ROOT)
        if dirs:
            chosen = random.choice(dirs)
            print(f"[INFO] 무작위 recipe 선택 ({len(dirs)}개 중): {chosen}")
            return resolve_assets(chosen)
        print("[WARNING] recipe 폴더가 없어 latest 경로로 폴백합니다.")
    return resolve_assets_auto()


def _build_ocr_client():
    """scale-bar OCR(PaddleOCR-VL) 클라이언트를 만든다(best-effort). 반환 (client|None, note)."""
    if not ENABLE_SCALE_BAR_HINT:
        return None, "disabled"
    try:
        return Workflow1VLMClient(
            service_slug=SCALE_BAR_OCR_SERVICE, log_name="scale_bar_ocr"), ""
    except Exception as exc:
        return None, f"client_failed: {str(exc)[:80]}"


def _scene_scale_bar(scene_gray: np.ndarray, client) -> dict:
    """한 scene 의 하단 scale bar 를 OCR 해 {raw_text, um, modality_hint, note} 반환.

    `_ocr_scale_bar` 를 재사용해 OCR→파싱→100µm 임계 로직을 단일 소스로 유지한다.
    client=None(=OCR 불가)이면 modality_hint=None.
    """
    if client is None:
        return {"raw_text": "", "um": None, "modality_hint": None, "note": "no_ocr"}
    try:
        raw_text, um, hint = _ocr_scale_bar(scene_gray, ocr_client=client)
    except Exception as exc:
        return {"raw_text": "", "um": None, "modality_hint": None,
                "note": f"ocr_failed: {str(exc)[:80]}"}
    return {"raw_text": raw_text, "um": um, "modality_hint": hint, "note": ""}


def _classify_scenes(assets, client) -> tuple[dict, bool]:
    """from_msr 를 scale-bar modality 로 분류한다 → {modality: {path, scale_bar}}.

    최신(visit-order 큰) 이미지부터 OCR 해 각 modality 의 *첫(=최신)* 이미지를 채택하고,
    om·sem 둘 다 찾으면 멈춘다(OCR 호출 수 제한). 반환 (scene_map, ocr_ok).
    ocr_ok=False 면 OCR 자체가 불가(client 없음) — 호출자가 폴백한다.
    """
    if client is None:
        return {}, False
    scene_map: dict[str, dict] = {}
    for path in reversed(iter_msr_images(assets)):  # 최신 visit-order 부터.
        try:
            gray = load_gray(path)
        except Exception as exc:
            print(f"[WARNING] msr 로드 실패({path.name}): {exc}")
            continue
        sb = _scene_scale_bar(gray, client)
        hint = sb["modality_hint"]
        if hint and hint not in scene_map:
            scene_map[hint] = {"path": path, "scale_bar": sb}
            print(f"[INFO] scene modality 분류: {hint} ← {path.name} (um={sb['um']})")
        if "om" in scene_map and "sem" in scene_map:
            break
    return scene_map, True


def _pairs_to_test(assets, scene_map: dict, ocr_ok: bool) -> list[dict]:
    """REFERENCE_MODALITY × 분류된 scene 으로 (reference, same-modality scene) 짝을 만든다.

    OM reference 는 OM scene 과만, SEM reference 는 SEM scene 과만 짝지어 cross-modality
    매칭을 막는다. OCR 자체가 불가하면(=분류 없음) current_sem 을 SEM scene 으로 가정하는
    폴백만 허용하고(이름 규약), OM 은 분류 없이 시험하지 않는다.
    """
    scene_map = dict(scene_map)
    # SEM 폴백: current_sem 이 *특정 modality 로 분류되지 않았을 때만* SEM 으로 가정한다
    # (이름 규약). OCR 이 current_sem 을 OM 으로 분류했다면 폴백하지 않아 오라벨을 막는다.
    classified_paths = {info["path"] for info in scene_map.values()}
    if ("sem" not in scene_map and assets.current_sem is not None
            and assets.current_sem not in classified_paths):
        note = "assumed_sem(no_ocr)" if not ocr_ok else "assumed_sem(scale_bar_unread)"
        scene_map["sem"] = {"path": assets.current_sem,
                            "scale_bar": {"raw_text": "", "um": None,
                                          "modality_hint": None, "note": note}}
    refs = {"sem": assets.recipe_sem, "om": assets.recipe_om}

    if REFERENCE_MODALITY in ("sem", "om"):
        wanted = [REFERENCE_MODALITY]
    elif REFERENCE_MODALITY == "both":
        wanted = ["sem", "om"]
    else:  # auto — current_sem 의 modality(분류 결과)만, 없으면 가용 첫.
        wanted = next(
            ([m] for m, info in scene_map.items() if info["path"] == assets.current_sem),
            list(scene_map.keys())[:1],
        )

    pairs: list[dict] = []
    for mod in wanted:
        if refs.get(mod) is None:
            print(f"[WARNING] {mod} reference(IMAP) 없음 — skip")
            continue
        if scene_map.get(mod) is None:
            print(f"[WARNING] {mod} modality scene 을 from_msr 에서 찾지 못함 — skip "
                  f"(OCR={'on' if ocr_ok else 'off'})")
            continue
        pairs.append({"modality": mod, "ref_label": f"recipe_{mod}",
                      "ref_path": refs[mod], "scene_path": scene_map[mod]["path"],
                      "scene_scale_bar": scene_map[mod]["scale_bar"]})
    return pairs


# ====================================================================
# VLM 호출 (네이티브 멀티이미지).
# ====================================================================


def _endpoint() -> str:
    base = LARGE_VLM_API_BASE.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    key = LARGE_VLM_API_KEY.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    else:
        print("[WARNING] LARGE_VLM_API_KEY 가 비어 있음 — 401 가능 (probe_multi_image_vlm 상수 미설정).")
    return headers


def _image_block(b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _ask_region(ref_bgr: np.ndarray, scene_bgr: np.ndarray) -> dict:
    """reference + scene 을 함께 보내 scene-relative region 을 묻는다.

    반환 dict: {ok, status, latency_s, payload_kb, raw_text, parsed, error}.
    """
    ref_enc = encode_under_limit(ref_bgr, ENCODE_FMT)
    scene_enc = encode_under_limit(scene_bgr, ENCODE_FMT)
    if ref_enc is None or scene_enc is None:
        return {"ok": False, "error": "encode_failed(1MB 초과/미지원)", "parsed": {}}
    ref_b64, mime, _ = ref_enc
    scene_b64, _, _ = scene_enc

    payload = {
        "model": PERCEPTION_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": [
                {"type": "text", "text": USER_MESSAGE},
                _image_block(ref_b64, mime),     # IMAGE-1 = reference.
                _image_block(scene_b64, mime),   # IMAGE-2 = scene.
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    out = {"ok": False, "status": None, "latency_s": None,
           "payload_kb": round(len(json.dumps(payload)) / 1024, 1),
           "raw_text": "", "parsed": {}, "error": ""}
    t0 = time.time()
    try:
        resp = requests.post(_endpoint(), headers=_headers(), json=payload, timeout=TIMEOUT_SEC)
        out["latency_s"] = round(time.time() - t0, 2)
        out["status"] = resp.status_code
        resp.raise_for_status()
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        except (ValueError, KeyError, IndexError, TypeError):
            text = resp.text
        out["raw_text"] = (text or "").strip()[:1000]
        out["parsed"] = extract_json(out["raw_text"])
        out["ok"] = True
    except requests.RequestException as exc:
        out["latency_s"] = round(time.time() - t0, 2)
        out["error"] = str(exc)[:300]
        out["status"] = getattr(getattr(exc, "response", None), "status_code", None)
    return out


def _region_to_pixels(parsed: dict, scene_w: int, scene_h: int) -> dict | None:
    """VLM 응답에서 scene-relative region 을 원본 scene 픽셀 bbox 로 변환한다.

    relative_1000 은 해상도 독립이라 인코딩 단계의 다운스케일과 무관하게 원본
    dims 로 바로 매핑된다.
    """
    if parsed.get("found") is not True:
        return None
    bbox_1000 = normalize_bbox_1000(parsed.get("region_bbox"))
    if bbox_1000 is None:
        return None
    return bbox_1000_to_pixels(bbox_1000, scene_w, scene_h)


# ====================================================================
# CV 핸드오프 — full-frame vs VLM-ROI.
# ====================================================================


def _score_summary(result) -> dict:
    """AlignKeyMatchResult 에서 비교에 필요한 필드만 추린다."""
    return {
        "score": round(float(result.score), 4),
        "chamfer_score": round(float(result.chamfer_score), 4),
        "orb_inlier_ratio": round(float(result.orb_inlier_ratio), 4),
        "decision": result.decision,
        "best_xy": list(result.best_xy),
        "best_scale": round(float(result.best_scale), 4),
        "distinctive": bool(result.distinctive),
        "score_gap": None if result.score_gap is None else round(float(result.score_gap), 4),
        "reject_reason": result.reject_reason,
    }


def _run_cv(ref_gray: np.ndarray, scene_gray: np.ndarray, recipe_id: str,
            roi: tuple[int, int, int, int] | None, out_dir: Path) -> dict:
    """reference template 으로 scene 을 매칭한다. roi=None 이면 full-frame.

    반환 dict: {ok, summary|error, overlay_path}.
    """
    tag = "roi" if roi is not None else "full"
    try:
        template = build_template(ref_gray, recipe_id=recipe_id, version="probe")
        result = compute_align_key_score(template, scene_gray, roi_hint=roi)
        overlay_path = out_dir / f"matcher_{tag}.jpg"
        cv2.imwrite(str(overlay_path), result.debug_overlay)
        return {"ok": True, "summary": _score_summary(result), "overlay_path": str(overlay_path)}
    except Exception as exc:  # 크롭이 template 보다 작거나 매칭 불가.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200], "overlay_path": ""}


def _save_region_overlay(scene_bgr: np.ndarray, region_px: dict | None,
                         out_path: Path) -> None:
    """scene 위에 VLM region(cyan) + 중심 마커를 그려 저장한다."""
    canvas = scene_bgr.copy()
    if region_px is not None:
        cv2.rectangle(canvas, (region_px["left"], region_px["top"]),
                      (region_px["right"], region_px["bottom"]), (255, 200, 0), 2)
        c = bbox_center(region_px)
        cv2.drawMarker(canvas, (c["x"], c["y"]), (255, 200, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(canvas, "VLM region", (region_px["left"], max(18, region_px["top"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "VLM: not found", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


# ====================================================================
# 엔트리.
# ====================================================================


def _probe_pair(pair: dict, recipe_id: str, out_dir: Path) -> dict:
    """한 (reference, same-modality scene) 짝에 대해 VLM region + CV 핸드오프를 수행한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = pair["modality"]
    ref_label, ref_path, scene_path = pair["ref_label"], pair["ref_path"], pair["scene_path"]
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    scene_bgr = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    base = {"modality": mod, "reference": {"label": ref_label, "path": str(ref_path)},
            "scene": {"path": str(scene_path), "scale_bar": pair["scene_scale_bar"]}}
    if ref_bgr is None or scene_bgr is None:
        print(f"[ERROR] [{mod}] 디코드 실패: ref={ref_path}, scene={scene_path}")
        return {**base, "error": "decode_failed"}
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    scene_h, scene_w = scene_gray.shape[:2]

    print(f"[INFO] [{mod}] reference={ref_path.name}  scene={scene_path.name} "
          f"{scene_w}x{scene_h}  model={PERCEPTION_MODEL}")

    # 1) VLM region (same-modality reference + scene).
    vlm = _ask_region(ref_bgr, scene_bgr)
    time.sleep(REQUEST_DELAY_SEC)
    region_px = _region_to_pixels(vlm.get("parsed", {}), scene_w, scene_h) if vlm["ok"] else None
    _save_region_overlay(scene_bgr, region_px, out_dir / "vlm_region_overlay.jpg")
    print(f"[INFO] [{mod}] VLM ok={vlm['ok']} status={vlm.get('status')} "
          f"found={vlm.get('parsed', {}).get('found')} conf={vlm.get('parsed', {}).get('confidence')} "
          f"region_px={region_px} err={vlm.get('error', '')[:60]}")

    # 2) CV 핸드오프 — full-frame, 그리고 region 이 있으면 ROI.
    cv_full = _run_cv(ref_gray, scene_gray, recipe_id, roi=None, out_dir=out_dir)
    roi = None
    if region_px is not None:
        roi = (region_px["left"], region_px["top"],
               region_px["right"] - region_px["left"],
               region_px["bottom"] - region_px["top"])
    cv_roi = _run_cv(ref_gray, scene_gray, recipe_id, roi=roi, out_dir=out_dir) if roi else None
    print(f"[INFO] [{mod}] CV full: {cv_full.get('summary') or cv_full.get('error')}")
    if cv_roi is not None:
        print(f"[INFO] [{mod}] CV roi : {cv_roi.get('summary') or cv_roi.get('error')}")

    base["scene"].update({"width": scene_w, "height": scene_h})
    return {
        **base,
        "vlm": {"ok": vlm["ok"], "status": vlm.get("status"),
                "latency_s": vlm.get("latency_s"), "payload_kb": vlm.get("payload_kb"),
                "found": vlm.get("parsed", {}).get("found"),
                "confidence": vlm.get("parsed", {}).get("confidence"),
                "evidence": vlm.get("parsed", {}).get("evidence"),
                "region_px": region_px, "raw_text": vlm.get("raw_text", ""),
                "error": vlm.get("error", "")},
        "cv_full": cv_full,
        "cv_roi": cv_roi,
        "output_dir": str(out_dir),
    }


def run() -> str:
    started = time.time()
    assets = _select_assets()
    if assets is None:
        print("[ERROR] align fail recipe 폴더를 찾지 못했습니다.")
        return "no_recipe"
    recipe_id = assets.recipe_id

    # from_msr 를 scale-bar modality 로 분류 → reference 를 same-modality scene 과만 짝짓는다.
    client, client_note = _build_ocr_client()
    if client is None:
        print(f"[WARNING] scale-bar OCR client 없음({client_note}) — SEM 폴백만 가능, OM 분류 불가.")
    scene_map, ocr_ok = _classify_scenes(assets, client)
    pairs = _pairs_to_test(assets, scene_map, ocr_ok)
    if not pairs:
        print(f"[ERROR] 시험할 (reference, same-modality scene) 짝이 없습니다: {assets.recipe_dir}")
        return "no_pairs"

    tag = make_timestamp_tag()
    out_dir = OUTPUT_ROOT / f"{tag}_{recipe_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] recipe_id={recipe_id}  reference_modality={REFERENCE_MODALITY}  "
          f"pairs={[(p['modality'], p['scene_path'].name) for p in pairs]}")

    pair_results = [
        _probe_pair(pair, recipe_id, out_dir / pair["ref_label"]) for pair in pairs
    ]

    summary = {
        "recipe_id": recipe_id,
        "recipe_dir": str(assets.recipe_dir),
        "recipe_select": RECIPE_SELECT,
        "reference_modality": REFERENCE_MODALITY,
        "model": PERCEPTION_MODEL,
        "ocr_ok": ocr_ok,
        "scene_classification": {
            mod: {"path": str(info["path"]), "scale_bar": info["scale_bar"]}
            for mod, info in scene_map.items()
        },
        "pairs": pair_results,
        "elapsed": format_elapsed_ms(started),
        "output_dir": str(out_dir),
        "note": "perception probe; reference paired ONLY with same-modality scene; "
                "VLM region NOT used as final coordinate (CV decides).",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    digest = _build_digest(summary)
    print(digest)
    (out_dir / "digest.txt").write_text(digest, encoding="utf-8")
    print(f"[INFO] 저장: {out_dir}")
    return "success"


def _decision_delta(cv_full: dict, cv_roi: dict | None) -> str:
    """full vs roi 점수/판정 변화를 한 줄로 요약 — escalation 이 먹혔는지."""
    if not cv_full.get("ok"):
        return f"CV full 실패({cv_full.get('error')}) — 비교 불가"
    full = cv_full["summary"]
    if cv_roi is None:
        return f"VLM region 없음 → ROI 매칭 미실행 (full score={full['score']}, {full['decision']})"
    if not cv_roi.get("ok"):
        return f"ROI 매칭 실패({cv_roi.get('error')}) — region 이 template 보다 작을 수 있음"
    roi = cv_roi["summary"]
    d_score = round(roi["score"] - full["score"], 4)
    arrow = "↑" if d_score > 0 else ("↓" if d_score < 0 else "=")
    return (f"full {full['decision']}/{full['score']} → roi {roi['decision']}/{roi['score']} "
            f"(Δscore {arrow}{d_score}; distinctive {full['distinctive']}→{roi['distinctive']})")


def _scale_bar_str(scene: dict) -> str:
    """scene scale-bar 힌트를 digest 한 토막으로 — audit 전용 라벨."""
    sb = scene.get("scale_bar") or {}
    if sb.get("modality_hint"):
        return f"scale_bar={sb.get('um')}µm→{sb['modality_hint']}(audit)"
    return f"scale_bar=?({sb.get('note') or 'n/a'})"


def _build_digest(summary: dict) -> str:
    lines = ["", "=" * 72, "align key region perception probe 결과", "=" * 72,
             f"recipe_id : {summary['recipe_id']}  (select={summary['recipe_select']})",
             f"recipe_dir: {summary['recipe_dir']}",
             f"model     : {summary['model']}  reference_modality={summary['reference_modality']}  "
             f"ocr={'on' if summary['ocr_ok'] else 'off'}"]
    cls = summary.get("scene_classification") or {}
    if cls:
        lines.append("scene 분류: " + "  ".join(
            f"{mod}={Path(info['path']).name}({_scale_bar_str(info)})"
            for mod, info in cls.items()))
    else:
        lines.append("scene 분류: (없음 — OCR 불가, SEM 폴백만)")
    for pr in summary["pairs"]:
        mod = pr["modality"]
        lines.append("-" * 72)
        ref_name = Path(pr["reference"]["path"]).name
        scene_name = Path(pr["scene"]["path"]).name
        if "error" in pr:
            lines.append(f"[{mod}] ref={ref_name} scene={scene_name} ERROR={pr['error']}")
            continue
        v = pr["vlm"]
        lines.append(f"[{mod}] ref={ref_name}  scene={scene_name}  {_scale_bar_str(pr['scene'])}")
        lines.append(f"  VLM   : ok={v['ok']} status={v['status']} found={v['found']} "
                     f"conf={v['confidence']} region_px={v['region_px']}")
        lines.append(f"          evidence={v['evidence']!r}" if v.get("evidence")
                     else f"          err={v['error']!r}")
        lines.append(f"  CV cmp: {_decision_delta(pr['cv_full'], pr['cv_roi'])}")
    lines.append("-" * 72)
    lines.append(f"elapsed={summary['elapsed']}  out_dir={summary['output_dir']}")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    run()
