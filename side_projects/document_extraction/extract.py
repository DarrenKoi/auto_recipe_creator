"""문서 추출 엔트리포인트.

폴더를 지정하면 내부의 PPT/Excel/Word/PDF 파일을 순차적으로 열고,
각 파일명을 딴 하위 폴더에 페이지별 JPEG로 저장한다.

CLI 인자는 사용하지 않는다. 매 실행마다 아래 모듈 상단 상수를 직접 수정해서 사용:
    INPUT_DIR   - 입력 폴더 경로 (필수)
    OUTPUT_DIR  - 출력 폴더 경로 (필수)
    OVERWRITE   - True면 이미 존재하는 출력 폴더를 덮어쓴다 (기본: 스킵)
    RECURSIVE   - True면 하위 폴더까지 재귀 탐색 (기본: 비재귀)
"""

import shutil
import sys
import traceback
from pathlib import Path


# 단독 실행(`python extract.py`) 시에도 absolute import가 동작하도록 repo root를 sys.path에 추가
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# === 실행 전 매번 채워 넣을 것 =================================================
INPUT_DIR: Path = Path("")   # 예: Path(r"C:\Users\me\Documents\문서더미")
OUTPUT_DIR: Path = Path("")  # 예: Path(r"C:\Users\me\Documents\extracted")
OVERWRITE: bool = False
RECURSIVE: bool = False
# ==============================================================================


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


def main() -> int:
    if not str(INPUT_DIR) or not str(OUTPUT_DIR):
        print("[ERROR] INPUT_DIR / OUTPUT_DIR 가 비어 있습니다. extract.py 상단을 수정해 경로를 지정하세요.")
        return 1

    input_dir = INPUT_DIR.expanduser().resolve()
    output_root = OUTPUT_DIR.expanduser().resolve()

    print(f"[INFO] INPUT_DIR  = {input_dir}")
    print(f"[INFO] OUTPUT_DIR = {output_root}")
    print(f"[INFO] OVERWRITE  = {OVERWRITE}")
    print(f"[INFO] RECURSIVE  = {RECURSIVE}")

    try:
        extract_folder(
            input_dir,
            output_root,
            recursive=RECURSIVE,
            overwrite=OVERWRITE,
        )
    except Exception as exc:
        print(f"[ERROR] 추출 중단: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
