import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

PARTS_DIR = Path("/home/ubuntu/uploads/my-model-upload")
OUTPUT_FILE = Path("/home/ubuntu/models/model.safetensors")
BUFFER_SIZE = 8 * 1024 * 1024
DELETE_PARTS_AFTER_JOIN = False


def expected_part_name(file_name: str, part_number: int) -> str:
    return "%s.part%03d" % (file_name, part_number)


def find_parts(parts_dir: Path, output_file_name: str) -> List[Path]:
    parts = sorted(parts_dir.glob("%s.part*" % output_file_name))
    if not parts:
        raise FileNotFoundError(
            "No parts found in %s for output file %s" % (parts_dir, output_file_name)
        )

    for index, part_path in enumerate(parts, start=1):
        expected_name = expected_part_name(output_file_name, index)
        if part_path.name != expected_name:
            raise ValueError(
                "Unexpected part sequence. Expected %s but found %s"
                % (expected_name, part_path.name)
            )

    return parts


def join_parts(part_paths: List[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as output_stream:
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


def read_expected_checksum(parts_dir: Path, output_file_name: str) -> Optional[str]:
    checksum_path = parts_dir / ("%s.sha256" % output_file_name)
    if not checksum_path.exists():
        return None

    tokens = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if not tokens:
        return None

    return tokens[0]


def main() -> None:
    if not PARTS_DIR.is_dir():
        raise FileNotFoundError("Parts directory does not exist: %s" % PARTS_DIR)

    part_paths = find_parts(PARTS_DIR, OUTPUT_FILE.name)
    join_parts(part_paths, OUTPUT_FILE)
    print("Rebuilt safetensors file: %s" % OUTPUT_FILE)

    expected_checksum = read_expected_checksum(PARTS_DIR, OUTPUT_FILE.name)
    if expected_checksum:
        actual_checksum = sha256sum(OUTPUT_FILE)
        if actual_checksum != expected_checksum:
            raise ValueError(
                "Checksum mismatch. expected=%s actual=%s"
                % (expected_checksum, actual_checksum)
            )
        print("Checksum verified.")
    else:
        print("No checksum file found. Skipping checksum verification.")

    if DELETE_PARTS_AFTER_JOIN:
        for part_path in part_paths:
            part_path.unlink()
        print("Deleted uploaded parts after successful join.")


if __name__ == "__main__":
    main()
