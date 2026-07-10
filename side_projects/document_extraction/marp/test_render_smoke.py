"""Stage 6 렌더(marp-cli) 스모크 테스트.

순수 인자 빌더(build_render_args)는 marp-cli 없이 검증되고, render_deck 의
graceful degrade(바이너리 부재)도 네트워크/marp 없이 검증된다. 실제 marp 렌더는
office(또는 marp-cli 설치 시)에서 확인.

실행:
    uv run python -m side_projects.document_extraction.marp.test_render_smoke
"""

import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.marp.render import (
    build_render_args,
    render_deck,
    resolve_marp_command,
)


def test_build_render_args_png_per_slide() -> None:
    args = build_render_args(Path("deck.md"), Path("out"), "png")
    assert "deck.md" in args[0]
    assert "--images" in args and "png" in args
    # 출력 경로는 -o 다음, png 확장자.
    oi = args.index("-o")
    assert args[oi + 1].endswith(".png")
    print("[PASS] test_build_render_args_png_per_slide")


def test_build_render_args_pptx_pdf_html() -> None:
    for fmt, flag, ext in [("pptx", "--pptx", ".pptx"),
                           ("pdf", "--pdf", ".pdf"),
                           ("html", "--html", ".html")]:
        args = build_render_args(Path("deck.md"), Path("out"), fmt)
        assert flag in args, (fmt, args)
        oi = args.index("-o")
        assert args[oi + 1].endswith(ext), (fmt, args)
    print("[PASS] test_build_render_args_pptx_pdf_html")


def test_build_render_args_rejects_unknown_format() -> None:
    try:
        build_render_args(Path("deck.md"), Path("out"), "gif")
    except ValueError:
        print("[PASS] test_build_render_args_rejects_unknown_format")
        return
    raise AssertionError("unknown format 는 ValueError 여야 한다")


def test_build_render_args_theme_css() -> None:
    from side_projects.document_extraction.marp.render import DOC_RESTORE_THEME_CSS

    args = build_render_args(Path("deck.md"), Path("out"), "png",
                             theme_css=DOC_RESTORE_THEME_CSS)
    ti = args.index("--theme")
    assert args[ti + 1].endswith("doc_restore.css")
    assert DOC_RESTORE_THEME_CSS.exists(), "테마 CSS 파일이 패키지에 있어야 함"
    text = DOC_RESTORE_THEME_CSS.read_text(encoding="utf-8")
    assert "@theme doc-restore" in text, "CSS @theme 이름은 프론트매터와 일치해야 함"
    # theme_css 미지정이면 --theme 없음(하위호환)
    assert "--theme" not in build_render_args(Path("deck.md"), Path("out"), "png")
    print("[PASS] test_build_render_args_theme_css")


def test_resolve_marp_command_returns_list_or_none() -> None:
    cmd = resolve_marp_command()
    assert cmd is None or (isinstance(cmd, list) and cmd)
    print(f"[PASS] test_resolve_marp_command_returns_list_or_none (cmd={cmd})")


def test_render_deck_graceful_when_binary_missing() -> None:
    # 존재하지 않는 base 명령 -> FileNotFoundError 잡고 available=False, 예외 없음.
    with tempfile.TemporaryDirectory() as tmp:
        deck = Path(tmp) / "deck.md"
        deck.write_text("---\nmarp: true\n---\n# x\n", encoding="utf-8")
        res = render_deck(deck, Path(tmp) / "out",
                          marp_cmd=["/definitely/not/here/marp-xyz"])
        assert res.available is False
        assert res.ok is False
    print("[PASS] test_render_deck_graceful_when_binary_missing")


def test_render_deck_runs_injected_command() -> None:
    # 성공 plumbing: 무해한 base 명령(true)으로 rc=0 경로 검증(marp 불필요).
    with tempfile.TemporaryDirectory() as tmp:
        deck = Path(tmp) / "deck.md"
        deck.write_text("---\nmarp: true\n---\n# x\n", encoding="utf-8")
        res = render_deck(deck, Path(tmp) / "out", marp_cmd=["true"])
        assert res.available is True
        assert res.ok is True   # true -> rc 0
    print("[PASS] test_render_deck_runs_injected_command")


def main() -> int:
    test_build_render_args_png_per_slide()
    test_build_render_args_pptx_pdf_html()
    test_build_render_args_rejects_unknown_format()
    test_build_render_args_theme_css()
    test_resolve_marp_command_returns_list_or_none()
    test_render_deck_graceful_when_binary_missing()
    test_render_deck_runs_injected_command()
    print("\n[INFO] 모든 render 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
