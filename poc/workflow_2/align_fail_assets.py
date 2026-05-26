"""Align fail 이미지 자산을 찾아 읽어주는 단일 창구.

오피스 MES 가 align fail 시 생성하는 실제 레이아웃 ([[align-images-layout]]):

```
poc/workflow_1/align_images/<eqp_id>/<class_name>/<recipe_name>/
  ├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe 등록 align key
  └─ align_img_from_msr/   S*/E*                             # 측정 궤적 (E 접두 = fail step)
```

본 모듈은 recipe leaf 폴더를 **최신 자동 선택**하거나 **(eqp/class/recipe) override**
로 지정해 `AlignFailAssets` 로 묶어준다. 모든 step 스크립트가 이 한 곳을 거쳐
경로를 해석하므로, 레이아웃이 바뀌면 여기만 고치면 된다.

공개 필드는 modality 기준으로 노출한다:
  - ``recipe_om``  = from_rcp/IMAP0001  (OM 등록 key)
  - ``recipe_sem`` = from_rcp/IMAP0002  (SEM 등록 key)
  - ``current_sem``= from_msr 의 최신 fail(E*) 이미지 (없으면 마지막 측정 이미지)
"""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import (
    ALIGN_IMAGES_ROOT,
    FROM_MSR_DIRNAME,
    FROM_RCP_DIRNAME,
    RCP_OM_STEM,
    RCP_SEM_STEM,
)

# 읽기 허용 확장자 (우선순위 순).
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class AlignFailAssets:
    """한 recipe(=한 align fail 이벤트)에 대한 이미지 경로 묶음.

    각 경로는 존재하지 않을 수 있다(부분 생성 허용). 호출자가 필요한 항목의
    존재 여부를 확인한다.
    """

    eqp_id: str
    class_name: str
    recipe_name: str
    recipe_dir: Path
    recipe_om: Path | None     # from_rcp/IMAP0001 (OM 등록 key).
    recipe_sem: Path | None    # from_rcp/IMAP0002 (SEM 등록 key).
    current_sem: Path | None   # from_msr 최신 fail 이미지.
    from_msr: tuple[Path, ...]  # from_msr 전체 궤적 (이름 순).

    @property
    def recipe_id(self) -> str:
        """출력 네이밍용 식별자 (recipe 폴더명)."""
        return self.recipe_name

    def available(self) -> list[tuple[str, Path]]:
        """존재하는 (label, path) 목록을 반환한다."""
        items = (
            ("recipe_om", self.recipe_om),
            ("recipe_sem", self.recipe_sem),
            ("current_sem", self.current_sem),
        )
        return [(label, path) for label, path in items if path is not None]


def _find_by_stem(directory: Path, stem: str) -> Path | None:
    """``directory`` 에서 주어진 stem 의 이미지 파일을 확장자 우선순위로 찾는다."""
    if not directory.is_dir():
        return None
    for ext in SUPPORTED_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # 대소문자/접미사 변형 fallback: stem 으로 시작하는 첫 이미지.
    for path in sorted(directory.glob(f"{stem}*")):
        if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
            return path
    return None


def _list_images(directory: Path) -> list[Path]:
    """``directory`` 의 지원 이미지들을 이름 순으로 모은다."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def _pick_current_sem(from_msr_images: list[Path]) -> Path | None:
    """from_msr 궤적에서 '현재 실패 SEM' 에 해당하는 이미지를 고른다.

    fail step 은 E 접두 파일이고 align 은 그 시점에서 멈춘다. 따라서 이름 순
    마지막 E* 파일을 우선하고, E* 가 없으면 마지막 측정 이미지로 fallback.
    """
    if not from_msr_images:
        return None
    e_files = [p for p in from_msr_images if p.name[:1].upper() == "E"]
    if e_files:
        return e_files[-1]
    return from_msr_images[-1]


def iter_recipe_dirs(root: Path = ALIGN_IMAGES_ROOT) -> list[Path]:
    """``root`` 아래에서 align_img_from_rcp 를 가진 recipe leaf 폴더를 모은다.

    레이아웃은 <eqp>/<class>/<recipe> 의 3단계라고 가정하되, from_rcp 폴더의
    부모를 recipe leaf 로 본다(중간 단계 수가 달라져도 동작).
    """
    if not root.is_dir():
        return []
    leaves: list[Path] = []
    for from_rcp in root.glob(f"*/*/*/{FROM_RCP_DIRNAME}"):
        if from_rcp.is_dir():
            leaves.append(from_rcp.parent)
    return sorted(leaves, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_assets(recipe_dir: Path) -> AlignFailAssets:
    """recipe leaf 폴더 하나를 AlignFailAssets 로 해석한다."""
    recipe_dir = recipe_dir.resolve()
    from_rcp = recipe_dir / FROM_RCP_DIRNAME
    from_msr = recipe_dir / FROM_MSR_DIRNAME

    # 경로 구조: .../<eqp_id>/<class_name>/<recipe_name>.
    parts = recipe_dir.parts
    recipe_name = parts[-1] if len(parts) >= 1 else ""
    class_name = parts[-2] if len(parts) >= 2 else ""
    eqp_id = parts[-3] if len(parts) >= 3 else ""

    from_msr_images = _list_images(from_msr)

    assets = AlignFailAssets(
        eqp_id=eqp_id,
        class_name=class_name,
        recipe_name=recipe_name,
        recipe_dir=recipe_dir,
        recipe_om=_find_by_stem(from_rcp, RCP_OM_STEM),
        recipe_sem=_find_by_stem(from_rcp, RCP_SEM_STEM),
        current_sem=_pick_current_sem(from_msr_images),
        from_msr=tuple(from_msr_images),
    )
    for label, path in (
        ("recipe_om", assets.recipe_om),
        ("recipe_sem", assets.recipe_sem),
        ("current_sem", assets.current_sem),
    ):
        if path is None:
            print(f"[WARNING] {label} 이미지를 찾지 못했습니다: {recipe_dir}")
        else:
            print(f"[INFO] {label} = {path.name}")
    return assets


def resolve_assets_auto(
    *,
    eqp_id: str = "",
    class_name: str = "",
    recipe_name: str = "",
    root: Path = ALIGN_IMAGES_ROOT,
) -> AlignFailAssets | None:
    """override(eqp/class/recipe) 가 모두 주어지면 그 경로를, 아니면 최신 폴더를 해석한다.

    override 우선순위: 인자 > 환경변수(ALIGN_EQP_ID/ALIGN_CLASS_NAME/ALIGN_RECIPE_NAME).
    셋 중 하나라도 비면 최신 align fail 폴더를 자동 선택한다.
    """
    eqp_id = (eqp_id or os.getenv("ALIGN_EQP_ID", "")).strip()
    class_name = (class_name or os.getenv("ALIGN_CLASS_NAME", "")).strip()
    recipe_name = (recipe_name or os.getenv("ALIGN_RECIPE_NAME", "")).strip()

    if eqp_id and class_name and recipe_name:
        recipe_dir = root / eqp_id / class_name / recipe_name
        if not recipe_dir.is_dir():
            print(f"[ERROR] 지정한 recipe 폴더가 없습니다: {recipe_dir}")
            return None
        print(f"[INFO] override recipe 폴더 사용: {recipe_dir}")
        return resolve_assets(recipe_dir)

    candidates = iter_recipe_dirs(root)
    if not candidates:
        print(f"[WARNING] align fail 폴더를 찾지 못했습니다: {root}")
        return None
    latest = candidates[0]
    print(f"[INFO] 최신 align fail 폴더 자동 선택: {latest}")
    return resolve_assets(latest)


def load_gray(path: Path) -> np.ndarray:
    """이미지를 grayscale uint8 numpy 로 읽는다. 실패 시 ValueError."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 디코드하지 못했습니다: {path}")
    return image
