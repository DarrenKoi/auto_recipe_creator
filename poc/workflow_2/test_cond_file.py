"""``cond_file`` 파서의 단위 테스트 (Mac 에서 실행 가능, 장비 불필요).

cond.txt 한 줄 한 줄은 `key  값,값,...` 형태이며, 우리가 쓰는 키는
  - Scope        : OM / SEM
  - Pixel        : 이미지 크기 (예: 512,512)
  - !Cursor_info : crosshair(elements[4],[5]) / white box(elements[6..9]) 좌표
            (cursor 좌표는 Pixel 의 10배 oversample 프레임)

실행:
    uv run python poc/workflow_2/test_cond_file.py
"""

import tempfile
from pathlib import Path

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_2.align_fail_assets import resolve_assets
from poc.workflow_2.cond_file import CondInfo, cond_path_for, load_cond, parse_cond

# 오피스에서 받아온 실제 cond.txt 샘플 (회귀 anchor).
REAL_SAMPLE = WORKFLOW_2_DIR / "docs" / "journals" / "260608" / "cond_sample.txt"

# --- 샘플 cond.txt 본문 (탭 구분, 값은 콤마 구분) --------------------------------

RCP_BOX = (
    "Scope\tSEM\n"
    "Magnification\t50000\n"
    "Pixel\t512,512\n"
    "!Cursor_info\t-1,-1,-1,-1,-1,-1,1770,1770,3380,3330,-1,-1\n"
)

MSR_CROSSHAIR = (
    "Scope\tOM\n"
    "Pixel\t1024,1024\n"
    "!Cursor_info\t-1,-1,-1,-1,2097,2561,-1,-1,-1,-1,-1\n"
)

BOTH = (
    "Scope\tSEM\n"
    "Pixel\t512,512\n"
    "!Cursor_info\t-1,-1,-1,-1,2097,2561,1770,1770,3380,3330\n"
)

NEITHER = (
    "Scope\tSEM\n"
    "Pixel\t512,512\n"
    "!Cursor_info\t-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n"
)


def test_parse_box_only():
    info = parse_cond(RCP_BOX)
    assert info.scope == "SEM", info.scope
    assert info.is_sem and not info.is_om
    assert info.pixel == (512, 512), info.pixel
    assert info.box_ltrb == (1770, 1770, 3380, 3330), info.box_ltrb
    assert info.crosshair_xy is None, info.crosshair_xy


def test_parse_crosshair_only():
    info = parse_cond(MSR_CROSSHAIR)
    assert info.scope == "OM" and info.is_om and not info.is_sem
    assert info.pixel == (1024, 1024), info.pixel
    assert info.crosshair_xy == (2097, 2561), info.crosshair_xy
    assert info.box_ltrb is None, info.box_ltrb


def test_parse_both_present():
    info = parse_cond(BOTH)
    assert info.crosshair_xy == (2097, 2561), info.crosshair_xy
    assert info.box_ltrb == (1770, 1770, 3380, 3330), info.box_ltrb


def test_parse_neither_present():
    info = parse_cond(NEITHER)
    assert info.box_ltrb is None and info.crosshair_xy is None


def test_cond_path_for_keeps_exact_filename():
    img = Path("align_images/EQP/CLS/RCP/align_img_from_rcp/IMAP0001AP.jpeg")
    expected = img.parent / ".IMAP0001AP.jpeg" / "cond.txt"
    assert cond_path_for(img) == expected, cond_path_for(img)

    msr = Path("a/align_img_from_msr/S14_A0001-01AP.jpeg")
    assert cond_path_for(msr) == msr.parent / ".S14_A0001-01AP.jpeg" / "cond.txt"


def test_condinfo_is_a_dataclass_instance():
    info = parse_cond(RCP_BOX)
    assert isinstance(info, CondInfo)


def test_parses_real_office_sample():
    """실제 오피스 cond.txt (주석/다중공백/꼬리 쓰레기 토큰 포함) 회귀 테스트."""
    if not REAL_SAMPLE.is_file():
        print(f"[WARNING] 실제 샘플 없음, skip: {REAL_SAMPLE}")
        return
    info = parse_cond(REAL_SAMPLE.read_text(encoding="utf-8"))
    assert info.is_om and info.scope == "OM", info.scope
    assert info.pixel == (512, 512), info.pixel
    assert info.box_ltrb == (1770, 1770, 3380, 3330), info.box_ltrb
    assert info.crosshair_xy is None, info.crosshair_xy


def _build_tree(root: Path, rel_image: str, cond_text: str) -> Path:
    """임시 align 트리에 이미지 + .<파일명>/cond.txt 를 만든다."""
    img = root / rel_image
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\xff\xd8\xff")            # 경로 해석용 최소 jpeg 헤더
    cond_dir = img.parent / f".{img.name}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "cond.txt").write_text(cond_text, encoding="utf-8")
    return img


def test_load_cond_reads_paired_file():
    with tempfile.TemporaryDirectory() as d:
        img = _build_tree(Path(d), "align_img_from_rcp/IMAP0002.jpeg", RCP_BOX)
        info = load_cond(img)
        assert info is not None and info.box_ltrb == (1770, 1770, 3380, 3330)


def test_load_cond_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "no_cond.jpeg"
        img.write_bytes(b"\xff\xd8\xff")
        assert load_cond(img) is None


def test_assets_cond_for_pairs_image_and_cond():
    with tempfile.TemporaryDirectory() as d:
        recipe = Path(d) / "EQP" / "CLS" / "RCP"
        _build_tree(recipe, "align_img_from_rcp/IMAP0002.jpeg", RCP_BOX)
        assets = resolve_assets(recipe)
        info = assets.cond_for(assets.recipe_sem)
        assert info is not None and info.is_sem
        assert info.box_ltrb == (1770, 1770, 3380, 3330), info.box_ltrb
        assert assets.cond_for(None) is None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[INFO] PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[ERROR] FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"[INFO] {len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
