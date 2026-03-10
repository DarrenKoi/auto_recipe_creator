import hashlib
from pathlib import Path
from typing import List

SOURCE_FILE = Path(r"C:\models\my-model\model.safetensors")
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


def split_file(file_path: Path, output_dir: Path, chunk_size: int) -> List[Path]:
    parts = []
    part_number = 1

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


def write_checksum(file_path: Path, checksum: str) -> Path:
    checksum_path = OUTPUT_DIR / ("%s.sha256" % file_path.name)
    checksum_path.write_text("%s  %s\n" % (checksum, file_path.name), encoding="utf-8")
    return checksum_path


def main() -> None:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError("Source safetensors file does not exist: %s" % SOURCE_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checksum = sha256sum(SOURCE_FILE)
    checksum_path = write_checksum(SOURCE_FILE, checksum)
    part_paths = split_file(SOURCE_FILE, OUTPUT_DIR, CHUNK_SIZE)

    print("Source file: %s" % SOURCE_FILE)
    print("Created checksum: %s" % checksum_path)
    print("Upload these files:")
    for part_path in part_paths:
        print("  %s" % part_path)
    print("  %s" % checksum_path)
    print("")
    print("Upload small files like config/tokenizer/index JSON normally.")
    print("Do not modify model.safetensors.index.json for this transfer split.")


if __name__ == "__main__":
    main()
