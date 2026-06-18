"""resolve_assets 가 msr 부재 시 current_sem 경고를 내지 않는지 검증하는 self-test.

프로덕션은 더 이상 align_img_from_msr 을 받지 않으므로(런타임 미소비), msr 트리가 비어도
'current_sem 못 찾음' 경고는 노이즈다. recipe_om/recipe_sem 경고는 그대로여야 한다.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/align/test_assets_no_msr.py
"""

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from poc.workflow_3.align import FROM_RCP_DIRNAME, RCP_OM_STEM, RCP_SEM_STEM
from poc.workflow_3.align.assets import resolve_assets


def _make_recipe_dir(base: Path, *, with_rcp: bool) -> Path:
    """<base>/EQP/CLS/RCP 레이아웃을 만들고 (옵션) rcp 더미 이미지 2장을 둔다."""
    recipe_dir = base / "EQP" / "CLS" / "RCP"
    if with_rcp:
        rcp = recipe_dir / FROM_RCP_DIRNAME
        rcp.mkdir(parents=True, exist_ok=True)
        (rcp / f"{RCP_OM_STEM}.jpg").write_bytes(b"x")
        (rcp / f"{RCP_SEM_STEM}.jpg").write_bytes(b"x")
    else:
        recipe_dir.mkdir(parents=True, exist_ok=True)
    return recipe_dir


def test_no_current_sem_warning_when_msr_absent():
    """rcp 만 있고 msr 이 없으면 current_sem 경고가 없어야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        recipe_dir = _make_recipe_dir(Path(tmp), with_rcp=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            assets = resolve_assets(recipe_dir)
        out = buf.getvalue()
    ok = (
        "current_sem" not in out.replace("current_sem =", "")  # 경고 라벨 부재
        and "[WARNING] current_sem" not in out
        and assets.current_sem is None
        and assets.recipe_om is not None
    )
    print(f"[{'PASS' if ok else 'FAIL'}] no_current_sem_warning_when_msr_absent: "
          f"current_sem_in_out={'[WARNING] current_sem' in out}")
    return ok


def test_recipe_warning_still_fires_when_rcp_absent():
    """rcp 도 없으면 recipe_om/recipe_sem 경고는 그대로 나와야 한다(회귀 가드)."""
    with tempfile.TemporaryDirectory() as tmp:
        recipe_dir = _make_recipe_dir(Path(tmp), with_rcp=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            resolve_assets(recipe_dir)
        out = buf.getvalue()
    ok = "[WARNING] recipe_om" in out and "[WARNING] recipe_sem" in out
    print(f"[{'PASS' if ok else 'FAIL'}] recipe_warning_still_fires_when_rcp_absent: "
          f"recipe_om_warn={'[WARNING] recipe_om' in out}")
    return ok


def main():
    print("[INFO] assets no-msr self-test 시작")
    results = [
        test_no_current_sem_warning_when_msr_absent(),
        test_recipe_warning_still_fires_when_rcp_absent(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
