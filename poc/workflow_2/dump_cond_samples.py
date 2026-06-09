"""golden 트리의 실제 cond.txt 내용을 진단용으로 덤프한다 (텍스트 digest).

verify_cond_cleaning.py 가 box/crosshair 0 을 보고할 때, 실제 !Cursor_info 가
어떤 인덱스에 값을 담는지(또는 구분자/키가 다른지) 눈으로 확인하기 위한 스크립트.
이미지 종류(IMAP=rcp / S·E=msr)별로 몇 개씩, raw 라인과 토큰 인덱스를 찍는다.

CLI 인자 없음. 루트는 ALIGN_GOLDEN_ROOT (기본 align_images_golden).
    uv run python poc/workflow_2/dump_cond_samples.py

출력 전체를 그대로 붙여주면 파서를 실데이터에 맞춘다.
"""

import os
from pathlib import Path

from poc.workflow_1 import WORKFLOW_1_DIR
from poc.workflow_3.vision.cond_file import parse_cond

GOLDEN_ROOT = Path(
    os.getenv("ALIGN_GOLDEN_ROOT", str(WORKFLOW_1_DIR / "align_images_golden"))
)
PER_GROUP = 3  # 그룹별 덤프 개수


def _group(image_name: str) -> str:
    up = image_name.upper()
    if up.startswith("IMAP"):
        return "IMAP (rcp)"
    if up.startswith("E"):
        return "E (msr fail)"
    if up.startswith("S"):
        return "S (msr success)"
    return "기타"


def main():
    print(f"[INFO] golden root: {GOLDEN_ROOT}")
    if not GOLDEN_ROOT.is_dir():
        print(f"[ERROR] 루트 없음: {GOLDEN_ROOT}")
        raise SystemExit(1)

    # .<이미지명>/cond.txt 들을 그룹별로 모은다.
    counts: dict[str, int] = {}
    for cond_path in sorted(GOLDEN_ROOT.rglob("cond.txt")):
        parent = cond_path.parent.name           # ".IMAP0002.jpeg"
        if not parent.startswith("."):
            continue
        image_name = parent[1:]                  # "IMAP0002.jpeg"
        grp = _group(image_name)
        if counts.get(grp, 0) >= PER_GROUP:
            continue
        counts[grp] = counts.get(grp, 0) + 1

        text = cond_path.read_text(encoding="utf-8", errors="replace")
        info = parse_cond(text)
        print("\n" + "=" * 70)
        print(f"[{grp}] {cond_path.relative_to(GOLDEN_ROOT)}")
        print(f"  파싱결과: scope={info.scope} pixel={info.pixel} "
              f"box={info.box_ltrb} crosshair={info.crosshair_xy}")
        cur = info.raw.get("cursor_info")
        print(f"  raw keys: {sorted(info.raw)}")
        if cur is None:
            print("  !! 'cursor_info' 키 없음 — 키 이름/구분자 다름 의심. 전체 라인:")
            for line in text.splitlines():
                print(f"     | {line}")
        else:
            print(f"  cursor_info 토큰 {len(cur)}개 (index: value):")
            print("   ", ", ".join(f"[{i}]={v!r}" for i, v in enumerate(cur)))

    if not counts:
        print("[WARNING] cond.txt 를 못 찾음.")
    else:
        print(f"\n[INFO] 덤프한 그룹/개수: {counts}")


if __name__ == "__main__":
    main()
