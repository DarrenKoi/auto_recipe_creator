"""VLM 서버 상태 확인 스크립트 (Python 버전).

사용법:
  python check_vlm.py <base_url> <expected_model_name>

예시:
  python check_vlm.py http://127.0.0.1:8001 ui-venus
"""

import sys
import urllib.request
import urllib.error


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    expected_model = sys.argv[2]
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
