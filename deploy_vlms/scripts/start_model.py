"""임의 instance 또는 family-size 조합으로 VLM 인스턴스를 시작한다.

사용법:
  python start_model.py <instance>
  python start_model.py <family> <size>

예시:
  python start_model.py ui-venus
  python start_model.py ui-venus 30b
  python start_model.py mai-ui-7b
"""

import subprocess
import sys
from pathlib import Path


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def resolve_instance() -> str:
    if len(sys.argv) == 2:
        return normalize_token(sys.argv[1])
    if len(sys.argv) == 3:
        return f"{normalize_token(sys.argv[1])}-{normalize_token(sys.argv[2])}"

    print(__doc__, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    instance = resolve_instance()
    script_dir = Path(__file__).resolve().parent
    sys.exit(subprocess.call([sys.executable, str(script_dir / "serve_vlm.py"), instance]))


if __name__ == "__main__":
    main()
