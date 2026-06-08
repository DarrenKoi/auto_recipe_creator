"""cond.txt 경로/내용 디버그 — modality 미해결(missing_modality) 원인 규명용.

왜
--
consensus 에서 S 프레임 254장이 missing_modality 로 누락됐다(cond 는 로드됐으나 modality
미해결). 둘 중 무엇인지 눈으로 확인한다:
  (a) msr cond.txt 에 Scope 줄이 정말 없다 → 다른 modality 소스 필요(rcp Scope/Magnification).
  (b) 경로 문제로 *엉뚱한* cond.txt(=Scope 없는)를 읽는다 → cond_path_for 수정 필요.

무엇을 보여주나
--------------
recipe 별로 rcp(IMAP0001/0002) cond.txt 와 msr S 몇 장의 cond.txt 를, **실제 경로 +
존재 여부 + 원문 + 파싱결과(scope/crosshair/box)** 로 나란히 덤프한다. rcp 에는 Scope 가
있고 msr 에는 없는지(또는 경로가 어긋났는지) 한눈에 보이게.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/debug_cond_paths.py
  env: ALIGN_GOLDEN_ROOT(루트), DEBUG_RECIPES(덤프할 recipe 수, 기본 3),
       DEBUG_MSR_PER_RECIPE(recipe 당 msr 표본, 기본 4).
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from pathlib import Path

from poc.workflow_2 import ALIGN_IMAGES_ROOT
from poc.workflow_2.align_fail_assets import iter_msr_images, resolve_assets
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_2.cond_file import cond_path_for, load_cond, parse_cond
from poc.workflow_2 import golden_localization_eval as gle

GOLDEN_ROOT = ALIGN_IMAGES_ROOT.parent / "align_images_golden"
N_RECIPES = int(os.getenv("DEBUG_RECIPES", "3"))
N_MSR = int(os.getenv("DEBUG_MSR_PER_RECIPE", "4"))


def _rel(path, root):
    """root 기준 상대경로(가능하면). resolve 차이(/tmp vs /private/tmp) 대비 안전 폴백."""
    if path is None:
        return None
    try:
        return path.relative_to(root)
    except ValueError:
        try:
            return path.resolve().relative_to(root.resolve())
        except ValueError:
            return path


def _dump_cond(img_path, root):
    """이미지 한 장의 cond 경로/존재/원문/파싱을 출력한다."""
    cp = cond_path_for(img_path)
    print(f"  [img] {_rel(img_path, root)}")
    rel_cp = _rel(cp, root)
    print(f"    cond_path: {rel_cp}")
    print(f"    cond_exists: {cp.is_file()}")
    if cp.is_file():
        raw = cp.read_text(encoding="utf-8", errors="replace")
        cond = parse_cond(raw)
        print(f"    parsed: scope={cond.scope!r}  pixel={cond.pixel}  "
              f"crosshair={cond.crosshair_xy}  box={cond.box_ltrb}")
        print("    raw:")
        for line in raw.splitlines():
            print(f"      | {line}")
    else:
        # 경로 어긋남 진단: 같은 폴더에 어떤 dot-folder 들이 있나?
        parent = img_path.parent
        siblings = [d.name for d in parent.iterdir() if d.is_dir() and d.name.startswith(".")] \
            if parent.is_dir() else []
        print(f"    [!] cond 없음. 같은 폴더의 dot-folder 들: {siblings[:8]}")


def _dump_rcp(label, path, root):
    print(f"  --- rcp {label}: {_rel(path, root)}")
    if path is not None:
        cond = load_cond(path)
        print(f"      rcp cond scope={cond.scope if cond else None!r} "
              f"(cond_exists={cond_path_for(path).is_file()})")


def main():
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else GOLDEN_ROOT
    print(f"[INFO] golden root: {root}")
    if not root.is_dir():
        print(f"[ERROR] 루트 없음: {root}")
        raise SystemExit(1)

    recipes = gle._collect_recipes(root)
    print(f"[INFO] recipe {len(recipes)}개. 앞 {N_RECIPES}개 덤프(각 msr {N_MSR}장).\n")
    shown = 0
    for assets in recipes:
        if assets is None:
            continue
        if shown >= N_RECIPES:
            break
        shown += 1
        print("=" * 72)
        print(f"=== recipe: {assets.recipe_id}  (eqp={assets.eqp_id} class={assets.class_name})")
        _dump_rcp("om(IMAP0001)", assets.recipe_om, root)
        _dump_rcp("sem(IMAP0002)", assets.recipe_sem, root)
        msr = iter_msr_images(assets)
        s_imgs = [p for p in msr if _tool_label(p.name) == "S"]
        print(f"  --- msr: 전체 {len(msr)}장, S 라벨 {len(s_imgs)}장. 앞 {N_MSR}장 cond 덤프:")
        for p in s_imgs[:N_MSR]:
            _dump_cond(p, root)
    print("\n[INFO] rcp 에는 Scope 가 있는데 msr 에는 없으면 → modality 는 rcp/Magnification 에서 "
          "끌어와야 함. cond_path 가 엉뚱하거나 cond_exists=False 면 → 경로 버그.")


if __name__ == "__main__":
    main()
