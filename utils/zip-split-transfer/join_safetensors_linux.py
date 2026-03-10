import hashlib
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

PARTS_DIR = Path("/home/ubuntu/uploads/my-model-upload")
OUTPUT_DIR = Path("/home/ubuntu/models")
BUFFER_SIZE = 8 * 1024 * 1024
DELETE_PARTS_AFTER_JOIN = False
PART_FILE_PATTERN = re.compile(r"^(?P<file_name>.+)\.part(?P<part_number>\d{3})$")


def expected_part_name(file_name: str, part_number: int) -> str:
    return "%s.part%03d" % (file_name, part_number)


def parse_part_name(file_name: str) -> Optional[Tuple[str, int]]:
    match = PART_FILE_PATTERN.match(file_name)
    if not match:
        return None

    output_file_name = match.group("file_name")
    if not output_file_name.endswith(".safetensors"):
        return None

    return output_file_name, int(match.group("part_number"))


def find_part_groups(parts_dir: Path) -> List[Tuple[Path, List[Path]]]:
    grouped_parts: dict[Path, List[Tuple[int, Path]]] = {}

    for part_path in sorted(parts_dir.rglob("*")):
        if not part_path.is_file():
            continue

        parsed = parse_part_name(part_path.name)
        if parsed is None:
            continue

        output_file_name, part_number = parsed
        relative_dir = part_path.relative_to(parts_dir).parent
        relative_output_path = relative_dir / output_file_name
        grouped_parts.setdefault(relative_output_path, []).append((part_number, part_path))

    if not grouped_parts:
        raise FileNotFoundError("No .safetensors part files found under %s" % parts_dir)

    part_groups = []
    for relative_output_path in sorted(grouped_parts):
        numbered_parts = sorted(grouped_parts[relative_output_path], key=lambda item: item[0])
        for expected_part_number, (actual_part_number, actual_part_path) in enumerate(
            numbered_parts, start=1
        ):
            expected_name = expected_part_name(
                relative_output_path.name, expected_part_number
            )
            if actual_part_number != expected_part_number:
                raise ValueError(
                    "Unexpected part sequence for %s. Expected part%03d but found part%03d"
                    % (
                        relative_output_path,
                        expected_part_number,
                        actual_part_number,
                    )
                )
            if actual_part_path.name != expected_name:
                raise ValueError(
                    "Unexpected part name for %s. Expected %s but found %s"
                    % (relative_output_path, expected_name, actual_part_path.name)
                )
        part_groups.append((relative_output_path, [path for _, path in numbered_parts]))

    return part_groups


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


def read_expected_checksum(parts_dir: Path, relative_output_path: Path) -> Optional[str]:
    checksum_path = (
        parts_dir / relative_output_path.parent / ("%s.sha256" % relative_output_path.name)
    )
    if not checksum_path.exists():
        return None

    tokens = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if not tokens:
        return None

    return tokens[0]


def main() -> None:
    if not PARTS_DIR.is_dir():
        raise FileNotFoundError("Parts directory does not exist: %s" % PARTS_DIR)

    part_groups = find_part_groups(PARTS_DIR)
    print("[INFO] Parts directory: %s" % PARTS_DIR)
    print("[INFO] Found %d .safetensors file(s) to rebuild." % len(part_groups))
    print("[INFO] Non-.safetensors upload files are ignored.")

    for relative_output_path, part_paths in part_groups:
        output_file = OUTPUT_DIR / relative_output_path
        join_parts(part_paths, output_file)
        print("[INFO] Rebuilt safetensors file: %s" % output_file)

        expected_checksum = read_expected_checksum(PARTS_DIR, relative_output_path)
        if expected_checksum:
            actual_checksum = sha256sum(output_file)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    "Checksum mismatch for %s. expected=%s actual=%s"
                    % (relative_output_path, expected_checksum, actual_checksum)
                )
            print("[INFO] Checksum verified: %s" % relative_output_path)
        else:
            print(
                "[WARNING] No checksum file found for %s. Skipping checksum verification."
                % relative_output_path
            )

        if DELETE_PARTS_AFTER_JOIN:
            for part_path in part_paths:
                part_path.unlink()

    if DELETE_PARTS_AFTER_JOIN:
        print("[INFO] Deleted uploaded parts after successful join.")


if __name__ == "__main__":
    main()
