"""golden 트리에서 align_img_from_rcp 폴더의 실제 깊이를 진단한다.

왜
--
iter_recipe_dirs 가 `*/*/*/align_img_from_rcp` (고정 3단계)로 glob 하는데 recipe 가
1개만 잡힌다. golden 트리의 실제 깊이가 다르면(2단계/4단계) 대부분이 안 잡힌다.
이 스크립트는 깊이 무관(rglob)으로 모든 from_rcp 를 찾아 깊이 분포를 보여줘
glob 패턴을 정확히 몇 단계로 고쳐야 하는지 알려준다.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/probe_golden_depth.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections import Counter

from poc.workflow_2 import FROM_RCP_DIRNAME
from poc.workflow_2 import golden_localization_eval_cond as glec


def main():
    root = glec.GOLDEN_ROOT
    print(f"[INFO] GOLDEN_ROOT = {root}")
    if not root.is_dir():
        print(f"[ERROR] 루트 없음: {root}")
        raise SystemExit(1)

    hits = [p for p in root.rglob(FROM_RCP_DIRNAME) if p.is_dir()]
    print(f"[INFO] align_img_from_rcp 폴더(깊이 무관) = {len(hits)}개")

    # depth = root 기준 상대 parts 수 (from_rcp 이름 포함). <eqp>/<class>/<recipe>/from_rcp = 4.
    depths = Counter(len(p.relative_to(root).parts) for p in hits)
    print(f"[INFO] 깊이 분포(parts, from_rcp 포함): {dict(sorted(depths.items()))}")
    print("[INFO] (현재 glob 은 4 parts = */*/*/align_img_from_rcp 만 매칭)")
    print("--- 표본 상대경로(앞 8개) ---")
    for p in hits[:8]:
        print(f"  {p.relative_to(root)}")


if __name__ == "__main__":
    main()
