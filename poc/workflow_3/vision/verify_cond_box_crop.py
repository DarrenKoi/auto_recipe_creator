"""cond box-crop 발화 확인 — 오피스에서 실제 align_images 자산으로 box-crop 경로가
실제로 도는지(아니면 center-crop 으로 폴백하는지) 한눈에 확인하는 진단 스크립트.

알람을 기다릴 필요 없이, 운영 루프와 *동일한* 빌드 경로
(build_templates_from_assets -> _load_template)를 직접 돌려 그 로그/결과를 보여준다.

판정에 보는 두 신호:
  1) 플래그: load_workflow3_settings().cond_box_crop (env ALIGN_FAIL_COND_BOX_CROP).
     False 면 운영 루프는 무조건 whole-template — box-crop 자체가 비활성.
  2) 데이터: 각 modality(OM=IMAP0001 / SEM=IMAP0002)의 cond.box 상태(check_cond_box).
     box 가 있고 skip 이 아니면 -> 'cond box-crop' (발화), 아니면 -> 'center-area crop'(폴백, 사유 표시).

box-crop 은 둘 다 만족해야 발화한다(플래그 ON + 유효한 cond box). 데이터가 없으면 에러 없이
검증된 center-crop 으로 조용히 폴백하므로, 발화 여부는 이렇게 명시적으로 확인해야 한다.

대상 recipe: 기본은 최신 align fail 폴더. 특정 recipe 지정은 env override:
  ALIGN_EQP_ID / ALIGN_CLASS_NAME / ALIGN_RECIPE_NAME (resolve_assets_auto 가 읽는다).

실행(오피스 Windows, 인자 없음): uv run python poc/workflow_3/vision/verify_cond_box_crop.py
"""

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.vision.align_fail_assets import load_gray, resolve_assets_auto
from poc.workflow_3.vision.align_fail_correct import build_templates_from_assets
from poc.workflow_3.vision.cond_file import load_cond
from poc.workflow_3.vision.cond_template import check_cond_box


def _modality_verdict(path):
    """rcp 이미지 한 장의 cond box 상태 -> (verdict, detail).

    _load_template 의 분기 기준과 동일(cond/box 유무 + check_cond_box). 운영 루프가
    이 이미지를 box-crop 으로 쓸지 center-crop 으로 폴백할지를 미리 알려준다.
    """
    gray = load_gray(path)
    cond = load_cond(path)
    box_ltrb = cond.box_ltrb if cond is not None else None
    if box_ltrb is None:
        why = "cond 파일 없음" if cond is None else "box 없음"
        return "FALLBACK(center)", why
    status, reason, onorm = check_cond_box(box_ltrb, gray.shape)
    if status == "skip":
        return "FALLBACK(center)", f"{reason} (offset_norm={onorm:.3f})"
    return "BOX-CROP", f"{status}:{reason} (offset_norm={onorm:.3f})"


def run() -> int:
    settings = load_workflow3_settings()
    print(
        f"[INFO] flag cond_box_crop = {settings.cond_box_crop} "
        f"(env ALIGN_FAIL_COND_BOX_CROP; False 면 운영 루프는 무조건 whole-template)"
    )

    assets = resolve_assets_auto()
    if assets is None:
        print(
            "[ERROR] align fail recipe 폴더를 찾지 못했습니다 "
            "(ALIGN_EQP_ID/ALIGN_CLASS_NAME/ALIGN_RECIPE_NAME 로 특정 recipe 지정 가능)."
        )
        return 1
    print(f"[INFO] recipe: {assets.recipe_dir}")

    sources = [("OM", assets.recipe_om), ("SEM", assets.recipe_sem)]
    found = [(mod, path) for mod, path in sources if path is not None]
    if not found:
        print("[ERROR] 등록 OM/SEM(IMAP0001/IMAP0002) 이미지가 없습니다.")
        return 1

    print("\n[INFO] === modality 별 cond box 판정(빌드 전 미리보기) ===")
    any_box = False
    for mod, path in found:
        verdict, detail = _modality_verdict(path)
        any_box = any_box or (verdict == "BOX-CROP")
        print(f"  {mod}: {verdict:<16} {detail}   <- {path.name}")

    print("\n[INFO] === 실제 빌드 경로 로그(운영 루프와 동일) ===")
    templates = build_templates_from_assets(assets, cond_box_crop=settings.cond_box_crop)
    for mod, tpl in templates.items():
        print(f"  {mod}: offset={tpl.align_offset_xy} raw_shape={tpl.raw_image.shape}")

    print("\n[INFO] === 결론 ===")
    if not settings.cond_box_crop:
        print(
            "  플래그 OFF -> 운영 루프는 box-crop 을 쓰지 않는다(whole-template). "
            "켜려면 ALIGN_FAIL_COND_BOX_CROP 미설정 또는 1 로."
        )
    elif any_box:
        print(
            "  box-crop 발화 O -> 위 'cond box-crop' 로그/판정 참고. 운영 루프도 동일하게 동작한다."
        )
    else:
        print(
            "  box-crop 미발화(전부 center-crop 폴백) -> 사유는 위 detail 참고 "
            "(cond.txt 부재/박스 없음/경계밖/너무작음/offset 과도)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
