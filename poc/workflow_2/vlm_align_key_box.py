"""Step 1 & 2 — VLM 이 align key 를 직접 bbox 로 잡을 수 있는지 확인하는 throwaway 프로브.

본 스크립트는 *production loop 에 들어가지 않는다*. 목적은 단 하나:
"외형이 (특히 align fail 상황에서) 달라진 align key 를, VLM 이 자연어 설명만으로
찾아 박스를 그릴 수 있는가?" 를 사람이 overlay 로 눈으로 평가하기 위한 증거 수집.

대상 이미지(= `align_fail_assets.resolve_assets_auto()` 가 해석한 recipe):
  - step 1: recipe_om(from_rcp/IMAP0001), recipe_sem(from_rcp/IMAP0002)  (등록 align key)
  - step 2: current_sem(from_msr 최신 E*)                                (현재 실패 SEM)

좌표를 신뢰해 매칭에 쓰지 않는다(그 일은 align_key_matcher 의 CV 가 한다). VLM 출력은
detect 여부 / confidence / overlay 로만 기록한다.

실행:
    uv run python poc/workflow_2/vlm_align_key_box.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.assets import resolve_assets_auto
from poc.workflow_3.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
from poc.workflow_3.vlm.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_3.util import format_elapsed_ms, make_timestamp_tag
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "vlm_align_key_box"

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정.
# ====================================================================
# recipe 폴더 선택은 align_fail_assets 가 담당(환경변수 override 또는 최신 자동).

DEFAULT_REQUEST_DELAY_SEC = float(os.getenv("TEST_VLM_REQUEST_DELAY_SEC", "1.0"))
DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME

# 각 자산의 의미를 프롬프트에 살짝 녹여 정확도를 높인다(소스별 힌트).
SOURCE_HINTS = {
    "recipe_om": "This is a stored OPTICAL-microscope (OM) reference image of the alignment key.",
    "recipe_sem": "This is a stored SEM reference image of the alignment key.",
    "current_sem": (
        "This is a LIVE SEM image captured at align-fail time; the key may look "
        "different (drifted, shifted, partially missing) from the stored reference."
    ),
}


def _align_key_system_prompt(source_hint: str) -> str:
    """Align key 탐지 시스템 프롬프트."""
    return (
        "You analyse a grayscale CD-SEM / optical alignment image from a "
        "semiconductor metrology tool. " + source_hint + "\n"
        "Locate the ALIGNMENT KEY (a.k.a. alignment mark / fiducial): a deliberately "
        "fabricated, high-contrast geometric pattern used to register the stage. "
        "Typical shapes are nested/concentric boxes (box-in-box), a cross/plus, an L "
        "or corner mark, or a small cluster of dots arranged asymmetrically. It is "
        "man-made and regular, NOT the random wafer texture, NOT charging gradients, "
        "NOT UI panels or text labels.\n"
        "Return strict JSON only. If the alignment key is not clearly present, say so "
        "rather than guessing on random texture."
    )


def _align_key_user_prompt() -> str:
    """Align key 탐지 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "key_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "key_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "key_shape": "box_in_box | cross | corner_L | dot_cluster | other",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string describing the pattern you used"\n'
        "}\n"
        "key_bbox must tightly enclose the whole alignment-key pattern. "
        "If no alignment key is clearly visible, set key_visible=false, key_bbox=null."
    )


def _detect_align_key(
    *, image_b64: str, width: int, height: int, source_hint: str, client: Workflow1VLMClient
) -> tuple[dict, dict | None]:
    """이미지에서 align key bbox 를 탐지한다. 반환 (payload, bbox_px|None)."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_align_key_system_prompt(source_hint),
        user_text=_align_key_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("key_visible") is not True:
        return parsed, None
    bbox_1000 = normalize_bbox_1000(parsed.get("key_bbox"))
    if bbox_1000 is None:
        return parsed, None
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


def _save_overlay(*, image_path: Path, key_bbox: dict, output_path: Path) -> str:
    """원본 이미지 위에 align key bbox 를 마킹한다."""
    with Image.open(image_path) as image:
        elements = {"align_key": {"bbox": key_bbox, "center": bbox_center(key_bbox)}}
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={"align_key": "lime"},
            out_path=output_path,
        )
    return str(output_path)


def run() -> str:
    started = time.time()
    assets = resolve_assets_auto()
    if assets is None:
        print("[ERROR] align fail recipe 폴더를 찾지 못했습니다.")
        return "no_recipe"
    recipe_id = assets.recipe_id

    targets = assets.available()
    if not targets:
        print(f"[ERROR] 분석할 이미지가 없습니다: {assets.recipe_dir}")
        return "no_assets"

    tag = make_timestamp_tag()
    out_dir = DEBUG_IMAGE_DIR / LOG_NAME / f"{tag}_{recipe_id}"
    overlays_dir = out_dir / "overlays"
    results_dir = out_dir / "results"
    for d in (overlays_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE, model_name=DEFAULT_MODEL, log_name=LOG_NAME
    )
    print(
        f"[INFO] align key 프로브 시작: recipe_id={recipe_id}, "
        f"service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, targets={len(targets)}"
    )

    results: list[dict] = []
    detected = 0
    for label, path in targets:
        step = "step1_recipe" if label.startswith("recipe") else "step2_current"
        print(f"[INFO] [{step}] {label}: {path.name}")
        try:
            with Image.open(path) as image:
                image_b64, w, h = encode_image_webp(image, quality=90)
        except Exception as exc:
            print(f"[ERROR] WebP 인코딩 실패: {label}, error={exc}")
            continue

        payload: dict = {}
        key_bbox: dict | None = None
        try:
            payload, key_bbox = _detect_align_key(
                image_b64=image_b64,
                width=w,
                height=h,
                source_hint=SOURCE_HINTS.get(label, ""),
                client=client,
            )
        except Exception as exc:
            print(f"[ERROR] VLM 호출 실패: {label}, error={exc}")
        finally:
            time.sleep(DEFAULT_REQUEST_DELAY_SEC)

        overlay_path = ""
        if key_bbox is not None:
            detected += 1
            try:
                overlay_path = _save_overlay(
                    image_path=path,
                    key_bbox=key_bbox,
                    output_path=overlays_dir / f"{step}_{label}_overlay.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패: {label}, error={exc}")
        else:
            print(f"[INFO] key_visible=false ({label})")

        result = {
            "step": step,
            "label": label,
            "image_path": str(path),
            "overlay_path": overlay_path,
            "key_bbox": key_bbox or {},
            "key_shape": payload.get("key_shape"),
            "confidence": payload.get("confidence"),
            "evidence": payload.get("evidence"),
            "raw_payload": payload,
        }
        save_debug_json(results_dir / f"{step}_{label}.json", result)
        results.append(result)

    summary = {
        "recipe_id": recipe_id,
        "recipe_dir": str(assets.recipe_dir),
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "targets": len(targets),
        "detected": detected,
        "elapsed": format_elapsed_ms(started),
        "output_dir": str(out_dir),
        "note": "throwaway feasibility probe; coordinates NOT used by the matcher",
    }
    save_debug_json(out_dir / "summary.json", summary)
    save_debug_text(
        out_dir / "timeline.txt",
        "\n".join(
            f"{r['step']:<13} {r['label']:<12} "
            f"detected={'Y' if r['key_bbox'] else 'N'} "
            f"shape={r['key_shape'] or '-':<12} conf={r['confidence']}"
            for r in results
        )
        + "\n",
    )
    print(
        f"[INFO] 완료: targets={len(targets)}, detected={detected}, "
        f"elapsed={format_elapsed_ms(started)}, out_dir={out_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
