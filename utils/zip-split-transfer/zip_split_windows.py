import hashlib
from pathlib import Path
from typing import List
from zipfile import ZIP_STORED, ZipFile

SOURCE_PATH = Path(r"C:\models\my-model")
OUTPUT_DIR = Path(r"C:\transfer\my-model-upload")
ARCHIVE_NAME = "my-model.zip"
CHUNK_SIZE = 2_000_000_000  # Keep each part below a typical 2 GB upload cap.
BUFFER_SIZE = 8 * 1024 * 1024
DELETE_ARCHIVE_AFTER_SPLIT = False


def ensure_output_is_outside_source(source_path: Path, output_dir: Path) -> None:
    if not source_path.is_dir():
        return

    source_root = source_path.resolve()
    output_root = output_dir.resolve()

    try:
        output_root.relative_to(source_root)
    except ValueError:
        return

    raise ValueError("OUTPUT_DIR must not be inside SOURCE_PATH.")


def build_zip(source_path: Path, archive_path: Path) -> None:
    if source_path.is_file():
        with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            archive.write(source_path, arcname=source_path.name)
        return

    base_parent = source_path.parent
    files = sorted(
        (path for path in source_path.rglob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )

    with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.relative_to(base_parent).as_posix())


def split_file(file_path: Path, chunk_size: int) -> List[Path]:
    parts = []
    part_number = 1

    with file_path.open("rb") as source_stream:
        while True:
            part_path = file_path.with_name(f"{file_path.name}.part{part_number:03d}")
            written = 0

            with part_path.open("wb") as part_stream:
                while written < chunk_size:
                    block = source_stream.read(min(BUFFER_SIZE, chunk_size - written))
                    if not block:
                        break
                    part_stream.write(block)
                    written += len(block)

            if written == 0:
                try:
                    part_path.unlink()
                except FileNotFoundError:
                    pass
                break

            parts.append(part_path)
            part_number += 1

    return parts


def sha256sum(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            block = stream.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source path does not exist: {SOURCE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_output_is_outside_source(SOURCE_PATH, OUTPUT_DIR)

    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    build_zip(SOURCE_PATH, archive_path)

    checksum = sha256sum(archive_path)
    checksum_path = archive_path.with_name(f"{archive_path.name}.sha256")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")

    parts = split_file(archive_path, CHUNK_SIZE)

    print(f"Created archive: {archive_path}")
    print(f"Created checksum: {checksum_path}")
    print("Upload these files:")
    for part_path in parts:
        print(f"  {part_path}")
    print(f"  {checksum_path}")

    if DELETE_ARCHIVE_AFTER_SPLIT:
        archive_path.unlink()
        print(f"Deleted original archive: {archive_path}")
    else:
        print(f"Kept original archive: {archive_path}")


if __name__ == "__main__":
    main()
