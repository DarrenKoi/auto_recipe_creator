import hashlib
import shutil
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

PARTS_DIR = Path("/home/ubuntu/uploads/my-model-upload")
ARCHIVE_NAME = "my-model.zip"
JOINED_ARCHIVE_PATH = PARTS_DIR / ARCHIVE_NAME
EXTRACT_DIR = Path("/home/ubuntu/models")
BUFFER_SIZE = 8 * 1024 * 1024
DELETE_JOINED_ARCHIVE_AFTER_EXTRACT = False


def find_parts(parts_dir: Path, archive_name: str) -> List[Path]:
    parts = sorted(parts_dir.glob(f"{archive_name}.part*"))
    if not parts:
        raise FileNotFoundError(f"No parts found in {parts_dir} for archive {archive_name}")
    return parts


def join_parts(part_paths: List[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_stream:
        for part_path in part_paths:
            with part_path.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output_stream, length=BUFFER_SIZE)


def sha256sum(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            block = stream.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def read_expected_checksum(parts_dir: Path, archive_name: str) -> Optional[str]:
    checksum_path = parts_dir / f"{archive_name}.sha256"
    if not checksum_path.exists():
        return None

    first_token = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    return first_token[0] if first_token else None


def safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_root = extract_dir.resolve()

    with ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupted ZIP entry detected: {bad_member}")

        for member in archive.infolist():
            destination = (extract_dir / member.filename).resolve()
            try:
                destination.relative_to(extract_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}") from exc

        archive.extractall(extract_dir)


def main() -> None:
    if not PARTS_DIR.exists():
        raise FileNotFoundError(f"Parts directory does not exist: {PARTS_DIR}")

    part_paths = find_parts(PARTS_DIR, ARCHIVE_NAME)
    join_parts(part_paths, JOINED_ARCHIVE_PATH)
    print(f"Rebuilt archive: {JOINED_ARCHIVE_PATH}")

    expected_checksum = read_expected_checksum(PARTS_DIR, ARCHIVE_NAME)
    if expected_checksum:
        actual_checksum = sha256sum(JOINED_ARCHIVE_PATH)
        if actual_checksum != expected_checksum:
            raise ValueError(
                "Checksum mismatch. "
                f"expected={expected_checksum} actual={actual_checksum}"
            )
        print("Checksum verified.")
    else:
        print("No .sha256 file found. Skipping checksum verification.")

    safe_extract_zip(JOINED_ARCHIVE_PATH, EXTRACT_DIR)
    print(f"Extracted archive into: {EXTRACT_DIR}")

    if DELETE_JOINED_ARCHIVE_AFTER_EXTRACT:
        JOINED_ARCHIVE_PATH.unlink()
        print(f"Deleted rebuilt archive: {JOINED_ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
