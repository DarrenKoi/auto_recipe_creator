"""consensus recipes=1 진단 — collision vs sparse 를 가른다.

왜
--
mod_total 은 om=335 sem=427(누계 784장)인데 A/B 는 recipes=1 S_loo=6 로 붕괴한다.
두 가설이 같은 증상을 낸다:
  (a) collision: by_recipe[recipe_id] 에서 recipe_id = recipe_name(leaf)만이라
      eqp/class 가 달라도 leaf 이름이 같으면 덮어써져 dict 가 몇 키로 붕괴.
  (b) sparse: recipe 당 S 가 2~3장뿐이라 AB_MIN_S(=4, 같은 modality)를 못 넘김.

이 probe 가 이미지 로드 없이(cond 파싱 + 파일명 라벨만) 다음을 찍어 둘을 가른다:
  - recipe dir 수 vs *고유* recipe_id 수 (collision 폭)
  - recipe 당 '최다 modality S 장수' 히스토그램 (sparse 폭)
  - AB_MIN_S(4) 이상인 recipe 수 = consensus LOO 가능한 recipe 수의 상한

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/probe_recipe_s_counts.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections import Counter

from poc.workflow_3.align.assets import iter_msr_images
from poc.workflow_3.align.diagnostics.align_point_correction import _tool_label
from poc.workflow_3.align.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval_cond as glec
from poc.workflow_2.golden_localization_eval import _collect_recipes
from poc.workflow_2.golden_consensus_eval_cond import _resolve_mod
import poc.workflow_2.golden_localization_eval_cond as _glec_mod

AB_MIN_S = 4   # align_similarity.AB_MIN_S 와 동일(같은 modality 최소 장수).


def main():
    root = glec.GOLDEN_ROOT
    print(f"[INFO] GOLDEN_ROOT = {root}")
    recipes = [a for a in _collect_recipes(root) if a is not None]
    print(f"[INFO] recipe dir 수 = {len(recipes)}")

    recipe_ids = [a.recipe_id for a in recipes]
    uniq_ids = set(recipe_ids)
    print(f"[INFO] 고유 recipe_id(=recipe_name leaf) 수 = {len(uniq_ids)}  "
          f"(< recipe dir 수 면 COLLISION — by_recipe dict 가 덮어써짐)")
    # 가장 많이 충돌하는 leaf 이름 표본.
    dup = Counter(recipe_ids)
    top_dup = [(k, v) for k, v in dup.most_common(5) if v > 1]
    if top_dup:
        print(f"[WARNING] 충돌 leaf 이름 top: {top_dup}")

    # recipe 당 '최다 modality S 장수' 분포 (이미지 로드 없이 cond+파일명만).
    dom_counts = []          # 각 recipe 의 dominant-modality S 장수.
    total_s = 0
    for a in recipes:
        try:
            center_tpls, _ = _glec_mod._build_offset_templates_cond(a)
        except Exception:
            center_tpls = {}
        rcp_mods = [m for m, v in (center_tpls or {}).items() if v is not None]
        recipe_mod = rcp_mods[0] if len(rcp_mods) == 1 else None
        per_mod = Counter()
        for p in iter_msr_images(a):
            if _tool_label(p.name) != "S":
                continue
            total_s += 1
            mod = _resolve_mod(load_cond(p), recipe_mod)
            per_mod[mod or "unresolved"] += 1
        # unresolved 는 consensus 에서 못 쓰므로 om/sem 중 최다만.
        usable = {m: c for m, c in per_mod.items() if m in ("om", "sem")}
        dom_counts.append(max(usable.values()) if usable else 0)

    print(f"[INFO] 총 S 라벨 = {total_s}")
    hist = Counter(min(c, 10) for c in dom_counts)   # 10+ 는 한 통.
    print("[INFO] recipe 당 dominant-modality S 장수 히스토그램(0..9, 10=10+):")
    for k in range(0, 11):
        bar = "#" * hist.get(k, 0)
        label = f"{k}+" if k == 10 else str(k)
        print(f"    S={label:>3} : {hist.get(k, 0):4d} {bar}")
    n_ge = sum(1 for c in dom_counts if c >= AB_MIN_S)
    print(f"[INFO] dominant-modality S >= AB_MIN_S({AB_MIN_S}) 인 recipe = {n_ge}개 "
          f"(= consensus LOO 가능 recipe 상한; A/B recipes 가 이보다 작으면 collision 도 작용)")


if __name__ == "__main__":
    main()
