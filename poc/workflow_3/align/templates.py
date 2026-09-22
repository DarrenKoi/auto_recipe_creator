"""Recipe align image to AlignKeyTemplate materialization."""

from pathlib import Path

from poc.workflow_3.align.assets import AlignFailAssets, load_gray
from poc.workflow_3.align.cond_file import (
    OM_DARK_BRIGHTNESS_MAX,
    cond_for_image,
    load_cond,
)
from poc.workflow_3.align.cond_template import (
    CENTER_AREA_RATIO,
    centered_area_crop,
    check_cond_box,
    cond_align_offset,
    cond_align_point,
    cond_template_crop,
)
from poc.workflow_3.align.matching.engine import AlignKeyTemplate, build_template
from poc.workflow_3.util.env_utils import env_flag

# align point 를 cond crosshair 에서 읽는다(기본 on). 0 이면 '이미지 중심' 가정으로 롤백 —
# 오피스에서 crosshair 가 align point 가 아닌 recipe 가 나오면 Claude 없이 되돌릴 수 있게
# 남겨둔 스위치다. **호출 시점에 읽는다**: 모듈 import 는 seed_env() 보다 앞이라 import
# 시점에 읽으면 workflow_3_config.py 사본이 이 값을 못 건드린다(VLM_LOCATOR_COMBO 와 같은 이유).
_CROSSHAIR_ALIGN_POINT_ENV = "ALIGN_FAIL_COND_CROSSHAIR_ALIGN_POINT"


def load_template(
    path: Path, *, recipe_id: str, key_type: str, cond_box_crop: bool
) -> AlignKeyTemplate:
    """Load one registered recipe image as a cond-aware AlignKeyTemplate."""
    gray = load_gray(path)
    name = Path(path).name  # 호출부가 str 을 넘기기도 한다.
    cond = cond_for_image(load_cond(path), gray.shape)
    if not cond_box_crop:
        crop, offset = gray, (0, 0)
        (ax, ay), source = cond_align_point(None, gray.shape)
        crop_kind, reason = "whole", "cond_box_crop=0"
    else:
        # cond.Pixel 과 로드 크기가 다르면 cursor 좌표를 먼저 보정한다(멱등) —
        # 안 하면 box crop/offset 이 계통적으로 어긋난 채 confident 하게 내려간다.
        box_ltrb = cond.box_ltrb if cond is not None else None
        if box_ltrb is None:
            status = "skip"
            reason = "cond 파일 없음" if cond is None else "box 없음"
        else:
            status, reason, _onorm = check_cond_box(box_ltrb, gray.shape)
        if status != "skip":
            crop, _bbox = cond_template_crop(gray, cond)
            # align point = crosshair(있으면) / 이미지 중심(폴백). box 중심이 아니다.
            use_crosshair = env_flag(_CROSSHAIR_ALIGN_POINT_ENV, default=True)
            (ax, ay), source = cond_align_point(cond if use_crosshair else None, gray.shape)
            offset = cond_align_offset(box_ltrb, gray.shape, cond if use_crosshair else None)
            crop_kind = "box"
        else:
            crop = centered_area_crop(gray, CENTER_AREA_RATIO)
            offset = (0, 0)
            (ax, ay), source = cond_align_point(None, gray.shape)
            crop_kind = "center"
    # 한 줄로 합친다 - 콘솔이 붐비면 이 줄들이 서로를 가린다. 여기 없는 값은 없는 것.
    # crosshair 와 이미지 중심의 delta 는 **항상 있다**(사람이 박스 정중앙을 못 찍는다,
    # 사용자 2026-09-16). 그래서 경고가 아니라 값으로만 찍는다 - 경고로 두면 매 recipe 마다
    # 울려서 진짜 이상 신호(scale pinned / clamp / offset=0)를 덮는다.
    rotation = cond.image_rotation if cond is not None else None
    # key_type(=매칭 라우팅 키)은 파일명 규약(IMAP0001=om / IMAP0002=sem)이 정하고, Scope 는
    # 화면 모드 판정(required_image_mode)에만 쓴다. 둘이 갈려도 key_type 을 여기서 뒤집지
    # 않는다 - route_template 이 보는 dict 키는 build_templates_from_assets 가 따로 정하므로
    # 한쪽만 뒤집으면 'sem 라벨인데 OM 칸에 꽂힌 template' 이 되어 더 나쁘다. 알리기만 하고,
    # 오피스 콘솔에 이 경고가 뜨면 그때 양쪽을 함께 Scope 기준으로 옮긴다.
    scope = (cond.scope or "").upper() if cond is not None else ""
    expected_scope = "SEM" if key_type.lower() == "sem" else "OM"
    if scope and not scope.startswith(expected_scope):
        print(f"[WARNING] {name}: 파일명 규약은 {key_type} 인데 cond Scope={scope} "
              f"- template 라우팅이 틀릴 수 있다(regi 확인 필요)")
    # 등록 화면 모드(OM / OM-D / SEM). OM-D 는 OM 의 명암 반전이라 live 화면이 다른 모드면
    # 완벽한 key 라도 NCC 가 눌려 match 임계를 못 넘는다 - 보정 전에 맞춰야 할 상태다.
    required_mode = cond.required_image_mode if cond is not None else None
    brightness = cond.om_brightness if cond is not None else None
    if required_mode is None:
        print(f"[WARNING] {name}: Scope={scope or '-'} / !OM_Brightness="
              f"{'-' if brightness is None else format(brightness, 'g')} 로 화면 모드를 "
              f"가를 수 없다 - 모드 정합을 건너뛴다")
    elif not scope and brightness is not None:
        # Scope 가 없어 **밝기만으로** 정한 판정. OM_DARK_BRIGHTNESS_MAX 는 오피스에서
        # 확인된 적 없는 가정치라(사용자 2026-09-22, 본인도 "I assume") 확정처럼 다루면
        # 안 된다. 아래 교차검증 분기는 이 경로에서 **구조적으로 못 울린다** - required_mode
        # 자체가 같은 임계로 나온 값이라 by_brightness 와 항상 같기 때문이다. 그래서
        # 여기서 따로 알린다: 임계가 틀리면 모드가 반대로 뒤집히고, 그때 live 화면과
        # 극성이 어긋나 NCC 가 눌려 완벽한 key 라도 match 임계를 못 넘는다.
        print(f"[WARNING] {name}: Scope 가 없어 !OM_Brightness={brightness:g} 로 "
              f"{required_mode} 라고 추정한다 (임계 {OM_DARK_BRIGHTNESS_MAX:g}, 미검증) "
              f"- 모드가 반대면 극성 반전으로 매칭이 구조적으로 실패한다")
    elif required_mode in ("OM", "OM-D") and brightness is not None:
        # Scope 가 1순위고 밝기 임계는 가정이다. 둘이 갈리면 Scope 를 쓰되 **알린다** -
        # 이 줄이 OM_DARK_BRIGHTNESS_MAX 를 오피스 데이터로 확정하는 유일한 신호다
        # (Scope 가 있는 cond 에서만 - 없는 쪽은 위 분기가 맡는다).
        by_brightness = "OM-D" if brightness < OM_DARK_BRIGHTNESS_MAX else "OM"
        if by_brightness != required_mode:
            print(f"[WARNING] {name}: Scope={scope} 는 {required_mode} 인데 "
                  f"!OM_Brightness={brightness:g} 는 {by_brightness} 를 가리킨다 "
                  f"(임계 {OM_DARK_BRIGHTNESS_MAX:g}) - Scope 를 따른다")
    h, w = gray.shape[:2]
    print(f"[INFO] {key_type} template: crop={crop_kind} offset={offset} "
          f"ap=({ax:.0f},{ay:.0f})/{source} d_center=({ax - w / 2:+.0f},{ay - h / 2:+.0f}) "
          f"rot={'-' if rotation is None else format(rotation, 'g')} "
          f"mag={cond.magnification if cond is not None else '-'} "
          f"scope={scope or '-'} mode={required_mode or '-'} "
          f"bright={cond.om_brightness if cond is not None else '-'} ({reason})")
    return build_template(
        crop,
        recipe_id=recipe_id,
        version="v0",
        key_type=key_type,
        align_offset_xy=offset,
        source_wh=(gray.shape[1], gray.shape[0]),
        source_magnification=cond.magnification if cond is not None else None,
        source_rotation_deg=rotation,
        required_image_mode=required_mode,
    )


def build_templates_from_assets(
    assets: AlignFailAssets, *, cond_box_crop: bool = True
) -> dict[str, AlignKeyTemplate]:
    """Convert available recipe OM/SEM images to template map."""
    templates: dict[str, AlignKeyTemplate] = {}
    if assets.recipe_om is not None:
        templates["OM"] = load_template(
            assets.recipe_om,
            recipe_id=assets.recipe_id,
            key_type="om",
            cond_box_crop=cond_box_crop,
        )
    if assets.recipe_sem is not None:
        templates["SEM"] = load_template(
            assets.recipe_sem,
            recipe_id=assets.recipe_id,
            key_type="sem",
            cond_box_crop=cond_box_crop,
        )
    return templates


# Internal name kept for tests that explicitly exercise the branch behavior.
_load_template = load_template

__all__ = ["build_templates_from_assets", "load_template"]
