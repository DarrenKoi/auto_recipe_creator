"""recipe 등록 align key (from_rcp) 를 template 으로, live SEM crop 을 scene 으로
한 번 매칭해 보는 probe.

`vlm_sem_monitor_box_realtime.py` 가 잘라낸 SEM-only crop (crops/iter_*.jpg) 안에
recipe 에 등록된 align key 패턴이 보이는지 `align_key_matcher` 로 점수화한다.
지금까지 matcher 는 합성 데이터로만 검증됐고, 본 스크립트가 **실제 SEM crop** 으로
처음 돌려보는 검증이다.

매칭 방향:
  - template = recipe 등록 SEM align key  = align_img_from_rcp/IMAP0002.jpeg
  - scene    = live SEM monitor crop       = realtime 실행의 crops/iter_*.jpg

office 이미지 레이아웃 (메모리 [[align-images-layout]]):
  poc/workflow_1/align_images/<eqp>/<class>/<recipe>/
    ├─ align_img_from_rcp/  IMAP0001.jpeg(OM)  IMAP0002.jpeg(SEM)
    └─ align_img_from_msr/  S*/E* (fail 궤적)

실행:
    uv run python poc/workflow_2/match_recipe_key_on_crop.py
"""

import os
import time
from pathlib import Path

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_2.align_fail_assets import load_gray
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text
from poc.workflow_1.util import format_elapsed_ms, make_timestamp_tag

# ====================================================================
# office align key 이미지 위치. 사용자가 알려준 실제 경로가 기본값.
# ====================================================================
ALIGN_IMAGES_ROOT = Path(
    os.getenv("ALIGN_IMAGES_ROOT", "")
).expanduser() if os.getenv("ALIGN_IMAGES_ROOT") else (
    Path(__file__).resolve().parent.parent / "workflow_1" / "align_images"
)
EQP_ID = os.getenv("ALIGN_EQP_ID", "MCD916").strip()
CLASS_NAME = os.getenv("ALIGN_CLASS_NAME", "RJ1BXXX").strip()
RECIPE_NAME = os.getenv("ALIGN_RECIPE_NAME", "Z_RJ1B_CBLHM2_FULL").strip()

# from_rcp 안의 파일명 (IMAP0001=OM, IMAP0002=SEM — 메모리 규약, 필요시 override).
RCP_SEM_FILE = os.getenv("ALIGN_RCP_SEM_FILE", "IMAP0002.jpeg").strip()

# ====================================================================
# scene(SEM crop) 후보. 비워두면 vlm_sem_monitor_box_realtime 의 가장 최신
# 실행 폴더에서 crops/*.jpg 를 자동으로 집는다.
# ====================================================================
QUERY_DIR_OVERRIDE = r""

WORKFLOW_2_RECORDING_DIR = WORKFLOW_2_DIR / "recordings"
REALTIME_ROOT = WORKFLOW_2_RECORDING_DIR / "vlm_sem_monitor_box_realtime"
DEFAULT_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / "match_recipe_key_on_crop"


def _recipe_dir() -> Path:
    """from_rcp/from_msr 가 들어 있는 recipe 폴더."""
    return ALIGN_IMAGES_ROOT / EQP_ID / CLASS_NAME / RECIPE_NAME


def _resolve_template_path() -> Path | None:
    """recipe 등록 SEM align key (template) 경로를 해석한다."""
    from_rcp = _recipe_dir() / "align_img_from_rcp"
    if not from_rcp.is_dir():
        print(f"[ERROR] align_img_from_rcp 폴더가 없습니다: {from_rcp}")
        return None
    path = from_rcp / RCP_SEM_FILE
    if path.is_file():
        return path
    print(f"[ERROR] recipe SEM align key 를 찾지 못했습니다: {path}")
    print(f"[INFO]  from_rcp 내용: {[p.name for p in sorted(from_rcp.iterdir())]}")
    return None


def _resolve_query_crops() -> list[Path]:
    """매칭 대상 SEM crop 목록을 해석한다."""
    override = (QUERY_DIR_OVERRIDE or "").strip()
    if override:
        query_dir = Path(override).expanduser()
    else:
        if not REALTIME_ROOT.is_dir():
            print(f"[ERROR] realtime 결과 루트가 없습니다: {REALTIME_ROOT}")
            return []
        runs = sorted(
            (p for p in REALTIME_ROOT.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            print(f"[ERROR] realtime 실행 폴더가 없습니다: {REALTIME_ROOT}")
            return []
        query_dir = runs[0] / "crops"
        print(f"[INFO] 최신 realtime crops 자동 선택: {query_dir}")

    if not query_dir.is_dir():
        print(f"[ERROR] crop 디렉터리가 없습니다: {query_dir}")
        return []
    crops = sorted(query_dir.glob("*.jpg"))
    if not crops:
        print(f"[ERROR] crop 이미지가 없습니다: {query_dir}")
    return crops


def run_probe() -> str:
    """recipe SEM align key 를 각 SEM crop 에 매칭한다."""
    started_at = time.time()

    template_path = _resolve_template_path()
    if template_path is None:
        return "template_not_found"

    crops = _resolve_query_crops()
    if not crops:
        return "no_query_crops"

    template_gray = load_gray(template_path)
    template = build_template(
        template_gray,
        recipe_id=RECIPE_NAME,
        version="from_rcp",
        nm_per_pixel=None,
        key_type="sem",
    )
    print(
        f"[INFO] template 로드: {template_path} "
        f"size={template_gray.shape[1]}x{template_gray.shape[0]}"
    )

    tag = make_timestamp_tag()
    output_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{RECIPE_NAME}"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    matched = 0

    for crop_path in crops:
        try:
            scene = load_gray(crop_path)
        except ValueError as exc:
            print(f"[WARNING] crop 디코드 실패: {crop_path.name} ({exc})")
            continue

        th, tw = template_gray.shape[:2]
        sh, sw = scene.shape[:2]
        if sh < th or sw < tw:
            print(
                f"[WARNING] crop 이 template 보다 작습니다 — 매칭 신뢰 낮음: "
                f"{crop_path.name} crop={sw}x{sh} template={tw}x{th}"
            )

        try:
            result = compute_align_key_score(template, scene, policy=STRUCTURE_POLICY)
        except Exception as exc:
            print(f"[ERROR] 매칭 실패: {crop_path.name} ({exc})")
            continue

        overlay_path = overlays_dir / f"{crop_path.stem}_match_{result.decision}.jpg"
        save_overlay_jpeg(result.debug_overlay, overlay_path)

        if result.decision == "match":
            matched += 1

        record = {
            "crop": str(crop_path),
            "overlay": str(overlay_path),
            "decision": result.decision,
            "score": round(float(result.score), 4),
            "chamfer": round(float(result.chamfer_score), 4),
            "orb": round(float(result.orb_inlier_ratio), 4),
            "best_xy": list(result.best_xy),
            "best_scale": round(float(result.best_scale), 4),
        }
        results.append(record)
        print(
            f"[INFO] {crop_path.name:<28s} decision={result.decision:<7s} "
            f"score={result.score:.3f} (chamfer={result.chamfer_score:.3f} "
            f"orb={result.orb_inlier_ratio:.3f}) scale={result.best_scale:.2f} "
            f"best_xy={result.best_xy}"
        )

    summary = {
        "template_path": str(template_path),
        "recipe_dir": str(_recipe_dir()),
        "policy": "STRUCTURE_POLICY",
        "crops_tested": len(results),
        "matched": matched,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "results": results,
    }
    save_debug_json(output_dir / "summary.json", summary)
    save_debug_text(
        output_dir / "scores.txt",
        "\n".join(
            f"{r['crop'].split('/')[-1]:<28s} {r['decision']:<7s} "
            f"score={r['score']:.3f} chamfer={r['chamfer']:.3f} orb={r['orb']:.3f}"
            for r in results
        )
        + "\n",
    )

    print(
        f"[INFO] 완료: crops={len(results)}, matched={matched}, "
        f"elapsed={format_elapsed_ms(started_at)}, output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run_probe() == "success" else 1)
