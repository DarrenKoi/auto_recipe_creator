"""golden 트리 recipe 수집 진단 — glob/rglob/_collect_recipes 를 한 프로세스에서 대조.

왜
--
이전에 `recipes found = 1` 인데 깊이 probe 는 align_img_from_rcp 298개(전부 depth 4)를
찾았다. depth 4 면 현재 glob `*/*/*/align_img_from_rcp` 가 *정상이면* 298 을 잡아야 한다
(pathlib glob 시맨틱 검증 완료). 두 수가 다르면 (a) 측정 사이 데이터가 늘었거나
(b) 수집기 경로/정렬에 버그. 이 스크립트가 같은 프로세스·같은 root 로 세 수를 나란히
찍어 어느 쪽인지 확정한다.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/probe_golden_depth.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections import Counter

from poc.workflow_2 import FROM_RCP_DIRNAME
from poc.workflow_2.align_fail_assets import iter_recipe_dirs
from poc.workflow_2 import golden_localization_eval_cond as glec
from poc.workflow_2.golden_localization_eval import _collect_recipes


def main():
    root = glec.GOLDEN_ROOT
    print(f"[INFO] GOLDEN_ROOT = {root}")
    print(f"[INFO] ALIGN_GOLDEN_ROOT env = {os.getenv('ALIGN_GOLDEN_ROOT')!r}")
    if not root.is_dir():
        print(f"[ERROR] 루트 없음: {root}")
        raise SystemExit(1)

    g = [p for p in root.glob(f"*/*/*/{FROM_RCP_DIRNAME}") if p.is_dir()]
    rg = [p for p in root.rglob(FROM_RCP_DIRNAME) if p.is_dir()]
    leaves = iter_recipe_dirs(root)
    recipes = [a for a in _collect_recipes(root) if a is not None]

    print(f"[INFO] glob   '*/*/*/{FROM_RCP_DIRNAME}' = {len(g)}개")
    print(f"[INFO] rglob  '{FROM_RCP_DIRNAME}'       = {len(rg)}개")
    print(f"[INFO] iter_recipe_dirs(root)            = {len(leaves)}개")
    print(f"[INFO] _collect_recipes(root)            = {len(recipes)}개")

    depths = Counter(len(p.relative_to(root).parts) for p in rg)
    print(f"[INFO] rglob 깊이 분포(parts, from_rcp 포함): {dict(sorted(depths.items()))}")

    # glob 이 못 잡는(=depth!=4) 폴더가 있으면 rglob 로 바꿔야 한다 — 그 차집합을 보여준다.
    missed = sorted(set(rg) - set(g))
    if missed:
        print(f"[WARNING] glob 이 놓친 {len(missed)}개(depth≠4) 표본:")
        for p in missed[:8]:
            print(f"    {p.relative_to(root)}")
    print("--- rglob 표본 상대경로(앞 5개) ---")
    for p in rg[:5]:
        print(f"  {p.relative_to(root)}")


if __name__ == "__main__":
    main()
