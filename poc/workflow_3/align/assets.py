"""Align fail 이미지 자산을 찾아 읽어주는 단일 창구.

오피스 MES 가 align fail 시 생성하는 실제 레이아웃 ([[align-images-layout]]):

```
<ALIGN_IMAGES_DIR>/<eqp_id>/<class_name>/<recipe_name>/
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
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3.align import (
    ALIGN_IMAGES_ROOT,
    FROM_MSR_DIRNAME,
    FROM_RCP_DIRNAME,
    RCP_OM_STEM,
    RCP_SEM_STEM,
)
from poc.workflow_3.align.cond_file import CondInfo, load_cond

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

    def cond_for(self, path: Path | None) -> CondInfo | None:
        """이미지 경로에 딸린 cond.txt 를 읽어 CondInfo 로 돌려준다 (없으면 None)."""
        if path is None:
            return None
        return load_cond(path)


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
    """``directory`` 하위(서브폴더 포함)의 지원 이미지들을 경로 순으로 모은다.

    msr 궤적은 평면 파일(S*/E* 접두)일 수도, S*/E* 서브폴더 안일 수도 있어 recursive 로
    훑는다(코드리뷰 [1]: iterdir 만 쓰면 서브폴더 이미지를 통째로 놓쳤다). 단, cond.txt
    sidecar 가 든 *숨김* dot-folder(``.<파일명>/``) 내부는 제외한다 — 그 안의 파일은
    주석/조건이지 측정 이미지가 아니다([[project_align_cond_files_and_coords]]).
    """
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in directory.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS):
            continue
        if any(part.startswith(".") for part in p.relative_to(directory).parts):
            continue   # 숨김 dot-folder 내부(cond sidecar 등) 제외.
        out.append(p)
    return sorted(out)


def _safe_mtime(path: Path) -> float:
    """st_mtime 을 안전하게 읽는다(접근 실패 시 0.0)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


_VISIT_ORDER_RE = re.compile(r"A(\d+)", re.IGNORECASE)


def _visit_order(path: Path) -> int:
    """파일명에서 A000X 의 X 를 정수로 뽑는다. 없으면 사전식 정렬 뒤로 보낸다.

    S/E 접두 뒤 두 자리는 의미가 없고, 매칭 순서를 결정하는 것은 A000X 뿐이다.
    """
    m = _VISIT_ORDER_RE.search(path.name)
    return int(m.group(1)) if m else 10**9


def iter_msr_images(assets: "AlignFailAssets") -> list[Path]:
    """from_msr 의 모든 이미지를 A000X visit-order 오름차순으로 돌려준다.

    파일명 접두 (S/E) 는 도구가 self-reported 한 라벨이라 항상 신뢰할 수 없다.
    호출자가 라벨/순서를 직접 검사할 수 있도록 path 를 그대로 흘려보낸다.
    """
    return sorted(assets.from_msr, key=_visit_order)


def _pick_current_sem(from_msr_images: list[Path]) -> Path | None:
    """from_msr 궤적에서 '현재 실패 SEM' 에 해당하는 이미지를 고른다.

    fail step 은 E 접두 파일이고 align 은 그 시점에서 멈춘다. E* 파일 중 visit-order
    (A000X) 가 가장 큰 것을 고른다 (E* 가 없으면 전체 중 큰 것). 파일명에 측정 순서가
    명시되어 있으므로 mtime (archive restore 로 뒤바뀔 수 있음) 보다 visit-order 가
    더 신뢰 가능한 정렬 키다 — [[project_align_images_layout]] 의 시퀀스 약속.
    iter_msr_images 도 같은 키를 사용해 두 호출자가 같은 '최신 fail' 을 가리키게 한다.
    """
    if not from_msr_images:
        return None
    e_files = [p for p in from_msr_images if p.name[:1].upper() == "E"]
    pool = e_files or from_msr_images
    return max(pool, key=_visit_order)


def _subtree_latest_mtime(leaf: Path) -> float:
    """recipe leaf 하위 트리에서 가장 최근 파일 mtime 을 구한다.

    새 align fail 이미지는 from_msr/ 등 *하위* 폴더에 떨어져 leaf 자체의 mtime 을
    바꾸지 않는다(서브폴더가 이미 있으면). 따라서 leaf mtime 이 아니라 하위 트리의
    최신 파일 시각으로 '가장 최근 fail' recipe 를 판단해야 한다.
    """
    latest = _safe_mtime(leaf)
    for child in leaf.rglob("*"):
        if child.is_file():
            m = _safe_mtime(child)
            if m > latest:
                latest = m
    return latest


def iter_recipe_dirs(root: Path = ALIGN_IMAGES_ROOT) -> list[Path]:
    """``root`` 아래에서 align_img_from_rcp 를 가진 recipe leaf 폴더를 모은다.

    레이아웃은 <eqp>/<class>/<recipe> 의 3단계로 고정 가정한다. 하위 트리의 최신
    파일 시각 기준 내림차순으로 정렬해 '가장 최근 fail' recipe 가 앞에 오게 한다.
    """
    if not root.is_dir():
        return []
    leaves: list[Path] = []
    for from_rcp in root.glob(f"*/*/*/{FROM_RCP_DIRNAME}"):
        if from_rcp.is_dir():
            leaves.append(from_rcp.parent)
    return sorted(leaves, key=_subtree_latest_mtime, reverse=True)


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
    # current_sem(from_msr) 은 런타임 미소비라 부재해도 경고하지 않는다 - 오프라인
    # 진단에서만 쓰며, 거기서는 호출부가 None 여부를 직접 확인한다.
    for label, path in (
        ("recipe_om", assets.recipe_om),
        ("recipe_sem", assets.recipe_sem),
    ):
        if path is None:
            print(f"[WARNING] {label} 이미지를 찾지 못했습니다: {recipe_dir}")
        else:
            print(f"[INFO] {label} = {path.name}")
    if assets.current_sem is not None:
        print(f"[INFO] current_sem = {assets.current_sem.name}")
    return assets


def resolve_assets_auto(
    *,
    eqp_id: str = "",
    class_name: str = "",
    recipe_name: str = "",
    root: Path = ALIGN_IMAGES_ROOT,
) -> AlignFailAssets | None:
    """완전한 override(eqp + class + recipe)면 그 경로를, 아니면 최신 폴더를 해석한다.

    override 우선순위: 인자 > 환경변수(ALIGN_EQP_ID/ALIGN_CLASS_NAME/ALIGN_RECIPE_NAME).
    recipe_name 은 "<class>/<recipe>" 슬래시 형태(알람 RECIPE_ID 와 동일)로 줘도 되며,
    class_name 과 합쳐 슬래시 단위로 분해한 뒤 eqp 아래에 join 한다. eqp_id 가 있고
    class+recipe 2단계 이상이 모이면 완전 override 로 보고, 그 외 부분 지정은 무시하되
    경고를 남기고 최신 폴더로 폴백한다(조용히 엉뚱한 recipe 를 분석하지 않도록).
    """
    eqp_id = (eqp_id or os.getenv("ALIGN_EQP_ID", "")).strip()
    class_name = (class_name or os.getenv("ALIGN_CLASS_NAME", "")).strip()
    recipe_name = (recipe_name or os.getenv("ALIGN_RECIPE_NAME", "")).strip()

    # class_name + recipe_name 을 슬래시 단위로 분해 (recipe_name="class/recipe" 도 허용).
    rel_parts = [
        part
        for segment in (class_name, recipe_name)
        for part in segment.replace("\\", "/").strip("/").split("/")
        if part
    ]
    any_override = bool(eqp_id or rel_parts)
    full_override = bool(eqp_id and len(rel_parts) >= 2)

    if full_override:
        recipe_dir = root.joinpath(eqp_id, *rel_parts)
        if not recipe_dir.is_dir():
            print(f"[ERROR] 지정한 recipe 폴더가 없습니다: {recipe_dir}")
            return None
        print(f"[INFO] override recipe 폴더 사용: {recipe_dir}")
        return resolve_assets(recipe_dir)

    if any_override:
        print(
            f"[WARNING] override 가 불완전합니다(eqp_id={eqp_id!r}, class={class_name!r}, "
            f"recipe={recipe_name!r}) — 무시하고 최신 align fail 폴더를 자동 선택합니다. "
            f"고정하려면 eqp_id + class + recipe 를 모두 지정하세요."
        )

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
