"""문서 추출 엔트리포인트.

폴더를 지정하면 내부의 PPT/Excel/Word/PDF 파일을 순차적으로 열고,
각 파일명을 딴 하위 폴더에 페이지별 JPEG로 저장한다.

CLI 인자는 사용하지 않으며, 모듈 상단 기본값 또는 환경변수로 설정한다.
    INPUT_DIR   - 입력 폴더 경로
    OUTPUT_DIR  - 출력 폴더 경로
    OVERWRITE   - "1"이면 이미 존재하는 출력 폴더를 덮어쓴다 (기본: 스킵)
    RECURSIVE   - "1"이면 하위 폴더까지 재귀 탐색 (기본: 비재귀)
"""

import os
import shutil
import sys
import traceback
from pathlib import Path


# 단독 실행(`python extract.py`) 시에도 absolute import가 동작하도록 repo root를 sys.path에 추가
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "test_inputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs"


# 확장자 → 핸들러 모듈 경로(lazy import 위해 문자열로 둠)
HANDLER_DISPATCH: dict[str, str] = {
    ".ppt": "side_projects.document_extraction.ppt_handler",
    ".pptx": "side_projects.document_extraction.ppt_handler",
    ".pptm": "side_projects.document_extraction.ppt_handler",
    ".xls": "side_projects.document_extraction.excel_handler",
    ".xlsx": "side_projects.document_extraction.excel_handler",
    ".xlsm": "side_projects.document_extraction.excel_handler",
    ".doc": "side_projects.document_extraction.word_handler",
    ".docx": "side_projects.document_extraction.word_handler",
    ".docm": "side_projects.document_extraction.word_handler",
    ".pdf": "side_projects.document_extraction.pdf_handler",
}


def _import_handler(module_path: str):
    import importlib

    return importlib.import_module(module_path)


def _iter_source_files(input_dir: Path, *, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() in HANDLER_DISPATCH:
            yield path


def extract_folder(
    input_dir: Path,
    output_root: Path,
    *,
    recursive: bool = False,
    overwrite: bool = False,
) -> None:
    """폴더 내 지원 파일을 순회하며 페이지 이미지로 추출한다."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    sources = list(_iter_source_files(input_dir, recursive=recursive))
    if not sources:
        print(f"[WARNING] 지원 가능한 파일이 없습니다: {input_dir}")
        return

    print(f"[INFO] 추출 대상 {len(sources)}개 파일 발견 (입력: {input_dir})")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for source in sources:
        out_dir = output_root / source.stem

        if out_dir.exists() and any(out_dir.iterdir()):
            if not overwrite:
                print(f"[INFO] 스킵(이미 존재): {source.name} → {out_dir.name}")
                skip_count += 1
                continue
            shutil.rmtree(out_dir)

        module_path = HANDLER_DISPATCH[source.suffix.lower()]
        try:
            handler = _import_handler(module_path)
            page_count = handler.extract(source, out_dir)
            print(f"[INFO] ✓ {source.name}: {page_count}페이지 저장 → {out_dir}")
            success_count += 1
        except Exception as exc:
            print(f"[ERROR] ✗ {source.name}: {exc}")
            traceback.print_exc()
            fail_count += 1
            continue

    print(
        f"[INFO] 완료 — 성공: {success_count}, 실패: {fail_count}, "
        f"스킵: {skip_count}"
    )


def _resolve_dir(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _resolve_bool(env_name: str, default: bool = False) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    input_dir = _resolve_dir("INPUT_DIR", DEFAULT_INPUT_DIR)
    output_root = _resolve_dir("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    overwrite = _resolve_bool("OVERWRITE", default=False)
    recursive = _resolve_bool("RECURSIVE", default=False)

    print(f"[INFO] INPUT_DIR  = {input_dir}")
    print(f"[INFO] OUTPUT_DIR = {output_root}")
    print(f"[INFO] OVERWRITE  = {overwrite}")
    print(f"[INFO] RECURSIVE  = {recursive}")

    try:
        extract_folder(
            input_dir,
            output_root,
            recursive=recursive,
            overwrite=overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] 추출 중단: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
