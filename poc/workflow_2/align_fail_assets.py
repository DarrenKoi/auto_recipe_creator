"""Align fail 발생 시 내려받은 이미지 자산을 찾아 읽어주는 공용 로더.

`poc/workflow_1/` 의 align fail 핸들러가 recipe 등록 align key 와 현재 실패 SEM
이미지를 ``ALIGN_FAIL_DOWNLOAD_DIR/<recipe_id>/`` 로 내려받는다고 가정한다
(파일 stem 은 ``recipe_om`` / ``recipe_sem`` / ``current_sem``, 확장자는 자유).

step 1~3 의 세 스크립트가 동일한 경로 규약을 공유하도록 한 곳에 모았다.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import (
    ALIGN_FAIL_DOWNLOAD_DIR,
    CURRENT_SEM_STEM,
    RECIPE_OM_STEM,
    RECIPE_SEM_STEM,
)

# 내려받은 이미지에서 허용하는 확장자 (우선순위 순).
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class AlignFailAssets:
    """한 align fail 이벤트에 대해 내려받은 이미지 경로 묶음.

    각 경로는 존재할 수도, 없을 수도 있다(부분 다운로드 허용). 호출자가
    필요한 항목의 존재 여부를 확인한다.
    """

    recipe_id: str
    download_dir: Path
    recipe_om: Path | None
    recipe_sem: Path | None
    current_sem: Path | None

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
    for ext in SUPPORTED_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # 대소문자/접미사 변형 fallback: stem 으로 시작하는 첫 이미지.
    for path in sorted(directory.glob(f"{stem}*")):
        if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
            return path
    return None


def resolve_assets(
    recipe_id: str,
    *,
    download_root: Path = ALIGN_FAIL_DOWNLOAD_DIR,
) -> AlignFailAssets:
    """``download_root/<recipe_id>/`` 에서 세 이미지 경로를 해석한다.

    디렉터리가 없으면 모든 경로가 None 인 객체를 반환하고 경고를 출력한다.
    """
    directory = download_root / recipe_id
    if not directory.is_dir():
        print(f"[WARNING] align fail 다운로드 폴더가 없습니다: {directory}")
        return AlignFailAssets(recipe_id, directory, None, None, None)

    assets = AlignFailAssets(
        recipe_id=recipe_id,
        download_dir=directory,
        recipe_om=_find_by_stem(directory, RECIPE_OM_STEM),
        recipe_sem=_find_by_stem(directory, RECIPE_SEM_STEM),
        current_sem=_find_by_stem(directory, CURRENT_SEM_STEM),
    )
    for label, path in (
        ("recipe_om", assets.recipe_om),
        ("recipe_sem", assets.recipe_sem),
        ("current_sem", assets.current_sem),
    ):
        if path is None:
            print(f"[WARNING] {label} 이미지를 찾지 못했습니다: {directory}/{label}.*")
        else:
            print(f"[INFO] {label} = {path.name}")
    return assets


def latest_recipe_dir(download_root: Path = ALIGN_FAIL_DOWNLOAD_DIR) -> str | None:
    """download_root 아래에서 가장 최근에 갱신된 recipe 서브폴더 이름을 반환한다.

    recipe_id 를 명시하지 않은 스크립트가 "가장 최근 align fail" 을 집어들 때 사용.
    """
    if not download_root.is_dir():
        print(f"[WARNING] 다운로드 루트가 없습니다: {download_root}")
        return None
    subdirs = [p for p in download_root.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[WARNING] 다운로드된 recipe 폴더가 없습니다: {download_root}")
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return latest.name


def load_gray(path: Path) -> np.ndarray:
    """이미지를 grayscale uint8 numpy 로 읽는다. 실패 시 ValueError."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 디코드하지 못했습니다: {path}")
    return image
