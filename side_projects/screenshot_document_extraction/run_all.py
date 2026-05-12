"""capture → extract → organize 세 단계를 순서대로 실행한다.

각 단계는 자기 책임 범위 안에서만 처리하며, 원장(ledger)으로 idempotent 하다.
중간에 멈춰도 같은 명령을 다시 실행하면 끝낸 페이지부터 이어서 처리한다.

사용법:
  uv run python side_projects/screenshot_document_extraction/run_all.py
"""

import _bootstrap

_bootstrap.ensure_paths()

from pipeline.capture import capture_runner  # noqa: E402
from pipeline.extract import extract_runner  # noqa: E402
from pipeline.organize import organize_runner  # noqa: E402


def main() -> None:
    """세 단계를 순서대로 실행한다."""
    print("[INFO] === Step 1: capture ===")
    capture_runner.run()
    print("[INFO] === Step 2: extract ===")
    extract_runner.run()
    print("[INFO] === Step 3: organize ===")
    organize_runner.run()
    print("[INFO] 전체 파이프라인 종료")


if __name__ == "__main__":
    main()
