"""VLM 서버 상태 확인 스크립트 (Python 버전).

사용법:
  python check_vlm.py <base_url> <expected_model_name_or_instance>

예시:
  python check_vlm.py http://127.0.0.1:8001 ui-venus
  python check_vlm.py http://127.0.0.1:8130 ui-venus-30b
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


def read_env_value(path: Path, key: str) -> str:
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, _, value = line.partition("=")
            if current_key.strip() != key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value
    return ""


def resolve_expected_model(value: str) -> str:
    env_path = Path(value)
    if env_path.is_file():
        served_model_name = read_env_value(env_path, "SERVED_MODEL_NAME")
        if served_model_name:
            print(f"[INFO] Resolved env file {env_path} -> served-model-name {served_model_name}")
            return served_model_name
        return value

    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or str(script_dir.parent)
    config_root = os.environ.get("CONFIG_ROOT", "").strip() or os.path.join(deploy_vlms_root, "config")
    instance_env = Path(config_root) / "models" / f"{value}.env"
    if instance_env.is_file():
        served_model_name = read_env_value(instance_env, "SERVED_MODEL_NAME")
        if served_model_name:
            print(f"[INFO] Resolved instance {value} -> served-model-name {served_model_name}")
            return served_model_name

    return value


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    expected_model = resolve_expected_model(sys.argv[2])
    url = f"{base_url}/v1/models"

    print(f"[INFO] Checking {url}")

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[ERROR] Failed to reach {url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(raw)

    if expected_model in raw:
        print(f"[INFO] Model alias found: {expected_model}")
    else:
        print(f"[ERROR] Expected model alias not found: {expected_model}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
