import hashlib
from pathlib import Path
from typing import List

SOURCE_DIR = Path(r"C:\models\my-model")
OUTPUT_DIR = Path(r"C:\transfer\my-model-upload")
CHUNK_SIZE = 2_000_000_000  # Keep each part below a typical 2 GB upload cap.
BUFFER_SIZE = 8 * 1024 * 1024


def sha256sum(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            block = stream.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def find_safetensors_files(source_dir: Path) -> List[Path]:
    return sorted(path for path in source_dir.rglob("*.safetensors") if path.is_file())


def split_file(file_path: Path, output_dir: Path, chunk_size: int) -> List[Path]:
    parts = []
    part_number = 1
    output_dir.mkdir(parents=True, exist_ok=True)

    with file_path.open("rb") as source_stream:
        while True:
            part_path = output_dir / ("%s.part%03d" % (file_path.name, part_number))
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


def write_checksum(file_path: Path, output_dir: Path, checksum: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = output_dir / ("%s.sha256" % file_path.name)
    checksum_path.write_text("%s  %s\n" % (checksum, file_path.name), encoding="utf-8")
    return checksum_path


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError("Source directory does not exist: %s" % SOURCE_DIR)

    source_files = find_safetensors_files(SOURCE_DIR)
    if not source_files:
        raise FileNotFoundError(
            "No .safetensors files found under source directory: %s" % SOURCE_DIR
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Source directory: %s" % SOURCE_DIR)
    print("[INFO] Found %d .safetensors file(s)." % len(source_files))
    print("[INFO] Non-.safetensors files are ignored.")
    print("Upload these files:")

    for source_file in source_files:
        relative_path = source_file.relative_to(SOURCE_DIR)
        parts_output_dir = OUTPUT_DIR / relative_path.parent
        checksum = sha256sum(source_file)
        checksum_path = write_checksum(source_file, parts_output_dir, checksum)
        part_paths = split_file(source_file, parts_output_dir, CHUNK_SIZE)

        print("  [FILE] %s" % relative_path)
        for part_path in part_paths:
            print("    %s" % part_path.relative_to(OUTPUT_DIR))
        print("    %s" % checksum_path.relative_to(OUTPUT_DIR))

    print("")
    print("Upload small files like config/tokenizer/index JSON normally.")
    print("Do not modify model.safetensors.index.json for this transfer split.")


if __name__ == "__main__":
    main()
