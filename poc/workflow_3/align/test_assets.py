"""align_fail_assets 파일 열거(_list_images) 테스트.

msr 궤적 이미지는 평면(S*/E* 접두 파일)일 수도, S*/E* 서브폴더 안일 수도 있다.
또 각 이미지 옆에는 cond.txt 가 든 *숨김* dot-folder(.<파일명>/)가 있으므로, 열거는
서브폴더 이미지를 포함하되 숨김 dot-folder 내부는 제외해야 한다(코드 리뷰 [1] 대응).
"""

from pathlib import Path

from poc.workflow_3.align.assets import _list_images


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")


def test_lists_flat_images(tmp_path):
    _touch(tmp_path / "S14_A0001AP.jpg")
    _touch(tmp_path / "E20_A0002AP.png")
    out = _list_images(tmp_path)
    assert [p.name for p in out] == ["E20_A0002AP.png", "S14_A0001AP.jpg"]


def test_lists_images_in_subfolders(tmp_path):
    # S*/E* 서브폴더 안의 이미지도 찾아야 한다(iterdir 만으로는 누락됨).
    _touch(tmp_path / "S" / "S14_A0001AP.jpg")
    _touch(tmp_path / "E1" / "E20_A0002AP.png")
    names = {p.name for p in _list_images(tmp_path)}
    assert names == {"S14_A0001AP.jpg", "E20_A0002AP.png"}


def test_finds_flat_and_subfolder_together(tmp_path):
    _touch(tmp_path / "S00_A0000AP.jpg")          # flat
    _touch(tmp_path / "S" / "S14_A0001AP.jpg")    # subfolder
    assert len(_list_images(tmp_path)) == 2


def test_excludes_hidden_cond_dotfolders(tmp_path):
    # cond.txt sidecar 가 든 .<파일명>/ 숨김 폴더 내부는 절대 측정 이미지로 보지 않는다.
    _touch(tmp_path / "S14_A0001AP.jpg")
    (tmp_path / ".S14_A0001AP.jpg").mkdir()
    (tmp_path / ".S14_A0001AP.jpg" / "cond.txt").write_text("Scope OM\n")
    _touch(tmp_path / ".S14_A0001AP.jpg" / "stray.jpg")   # 숨김 폴더 내 이미지 → 제외돼야.
    out = _list_images(tmp_path)
    assert [p.name for p in out] == ["S14_A0001AP.jpg"]


def test_excludes_extensions_not_supported(tmp_path):
    _touch(tmp_path / "keep.jpg")
    _touch(tmp_path / "skip.txt")
    assert [p.name for p in _list_images(tmp_path)] == ["keep.jpg"]


def test_missing_directory_returns_empty(tmp_path):
    assert _list_images(tmp_path / "nope") == []
