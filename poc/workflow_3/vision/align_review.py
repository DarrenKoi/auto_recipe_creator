"""Align point correction batch 결과를 *status 축* 으로 재배치한 review 뷰를 만든다.

문제: `align_point_correction.py` 는 recipe 마다 timestamp 폴더
(`align_correction/<eqp>__<class>__<recipe>__<ts>/`) 를 만든다. test set 이 100+ 로
늘면 폴더를 하나씩 열어 overlay 를 보는 게 불가능하다.

해결: 저장(=source of truth)은 recipe 폴더 그대로 두고, **review 뷰를 status 축으로 한 겹**
얹는다. "어떤 recipe 냐" 가 아니라 "어떤 실패 유형이냐" 가 디버깅 축이기 때문이다.

산출 (out_root = `debug_images/align_correction/` 아래):

    review_<batch_ts>/
    ├─ by_status/<status>/<recipe_tag>__<msr_stem>.jpg   # downscale thumbnail (복사)
    │    suspect_success/  not_distinctive/  low_match_both/  ...
    ├─ index.html        # status 섹션 + thumbnail grid, worst-first 정렬, 클릭→풀해상도
    └─ review_summary.json

설계 (recovery plan Phase 1 review 레이아웃 준수):
  - primary 그룹 축 = status. 한 폴더만 열면 전 recipe 의 같은 실패 유형이 모인다.
  - `tool_label_suspect=True` row 는 actual status 와 별개로 **suspect_success** 섹션에도 노출.
  - thumbnail 은 **복사** (Windows symlink 불안정). 풀해상도 원본은 recipe 폴더에만.
  - index.html 은 순수 표준 라이브러리 f-string → write_text. 의존성 없음.
    상대경로 `<img>` + `loading="lazy"` + 인라인 `<script>` status 토글. 서버 불필요.
  - overlay 경로는 row 에 저장된 절대경로 대신 recipe out_dir 에서 재구성 — 폴더 이동에 강건.

모드:
  - 실데이터: out_root 아래 최신 `batch_summary_*.json` 의 recipe 들을 aggregate.
  - 데이터 부재 (Mac dev): 합성 fixture 로 self-test (compare_align_images 패턴).

실행:
    uv run python poc/workflow_3/vision/align_review.py
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import DEBUG_IMAGE_DIR

# ====================================================================
# 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정.
# ====================================================================

# review thumbnail 가로 폭 (px). 풀해상도는 recipe 폴더 원본 클릭으로.
THUMB_WIDTH = 320

# status 표시 순서 = review 우선순위 (worst-first). 위에 있을수록 먼저 봐야 하는 유형.
# suspect_success 는 status 값이 아니라 tool_label_suspect 에서 파생된 pseudo-section.
STATUS_ORDER = (
    "processing_error",     # 코드 예외 — 가장 먼저 확인.
    "suspect_success",      # S 라벨인데 보정 거리 큼 — 사용자 최우선 관심.
    "not_distinctive",      # frame 안 다른 곳과 비슷한 점수 — 진짜 align 위치 아닐 수 있음.
    "low_match_both",       # 양쪽 template 점수 낮음 / pad-border 매칭.
    "no_templates",         # rcp 에 IMAP0001/0002 부재.
    "msr_unrecognizable",   # 전체 blur.
    "no_crosshair_drawn",   # 도구가 포기.
    "ambiguous_modality",   # OM/SEM 점수차 작음.
    "already_aligned",      # crosshair 와 거의 일치.
    "ok",                   # 정상.
)

# status 별 헤더 색 (CSS) — 심각도 직관.
_STATUS_COLOR = {
    "processing_error": "#c0392b",
    "suspect_success": "#e67e22",
    "not_distinctive": "#d35400",
    "low_match_both": "#e74c3c",
    "no_templates": "#7f8c8d",
    "msr_unrecognizable": "#8e44ad",
    "no_crosshair_drawn": "#2980b9",
    "ambiguous_modality": "#16a085",
    "already_aligned": "#27ae60",
    "ok": "#2ecc71",
}


# ====================================================================
# batch / recipe 수집.
# ====================================================================


def _safe(s: str | None) -> str:
    """경로/파일명에 안전한 토큰으로 변환."""
    return (s or "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _latest_batch_summary(out_root: Path) -> Path | None:
    """out_root 아래 가장 최근 batch_summary_*.json 을 찾는다 (없으면 None)."""
    candidates = sorted(out_root.glob("batch_summary_*.json"))
    return candidates[-1] if candidates else None


def _recipe_dirs_from_batch(out_root: Path, batch_summary_path: Path) -> list[Path]:
    """batch_summary 의 processed_recipes[].out_dir 을 Path 목록으로 돌려준다.

    out_dir 이 절대경로로 저장돼 있어도, 폴더 이름만 떼어 out_root 기준으로 다시 붙여
    다른 머신/이동된 트리에서도 동작하게 한다.
    """
    try:
        data = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] batch_summary 읽기 실패 ({batch_summary_path}): {exc}")
        return []
    dirs: list[Path] = []
    for entry in data.get("processed_recipes", []):
        out_dir = entry.get("out_dir")
        if not out_dir:
            continue
        local = out_root / Path(out_dir).name
        if local.is_dir():
            dirs.append(local)
        elif Path(out_dir).is_dir():
            dirs.append(Path(out_dir))
    return dirs


def _all_recipe_dirs(out_root: Path) -> list[Path]:
    """batch_summary 가 없을 때의 폴백 — results.jsonl 을 가진 모든 하위 폴더를 모은다.

    같은 recipe 의 timestamp 가 여러 개면 최신 ts 하나만 (폴더 이름 정렬로 최신 = 마지막).
    """
    by_recipe: dict[str, Path] = {}
    for jsonl in out_root.glob("*/results.jsonl"):
        d = jsonl.parent
        name = d.name
        # 폴더명 = <eqp>__<class>__<recipe>__<ts>. ts 를 떼고 recipe 키로 묶어 최신만.
        key = name.rsplit("__", 1)[0] if "__" in name else name
        prev = by_recipe.get(key)
        if prev is None or name > prev.name:
            by_recipe[key] = d
    return sorted(by_recipe.values())


def _iter_rows(recipe_dir: Path):
    """recipe_dir/results.jsonl 을 줄단위로 읽어 (row, overlay_full_path) 를 yield.

    overlay 경로는 row 의 저장값 대신 recipe_dir 에서 재구성 — 폴더 이동에 강건.
    """
    jsonl = recipe_dir / "results.jsonl"
    if not jsonl.is_file():
        return
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msr = row.get("msr_image") or ""
        stem = Path(msr).stem
        overlay = recipe_dir / "overlay" / f"{stem}_overlay.jpg"
        yield row, (overlay if overlay.is_file() else None)


# ====================================================================
# review 카드 빌드.
# ====================================================================


def _sections_for_row(row: dict) -> list[str]:
    """한 row 가 들어갈 review section key 들. actual status + (의심이면) suspect_success."""
    keys = [row.get("status", "ok")]
    if row.get("tool_label_suspect"):
        keys.append("suspect_success")
    return keys


def _make_thumb(src: Path, dst: Path, width: int = THUMB_WIDTH) -> bool:
    """src overlay 를 width 기준으로 축소해 dst 에 JPEG 로 저장. 성공 여부 반환."""
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    if w > width:
        scale = width / float(w)
        img = cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85]))


def _card_meta(row: dict) -> str:
    """카드 캡션용 한 줄 메타 (label · modality · score · dist · scale)."""
    label = row.get("tool_label", "?")
    modality = (row.get("winner_modality") or "-").upper()
    score = row.get("winner_score")
    dist = row.get("correction_magnitude_px")
    visit = row.get("visit_order")
    score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
    dist_s = f"{dist:.0f}px" if isinstance(dist, (int, float)) else "-"
    visit_s = f"A{visit:04d}" if isinstance(visit, int) else "A????"
    return f"{label} · {visit_s} · {modality} · score={score_s} · dist={dist_s}"


# ====================================================================
# HTML 렌더.
# ====================================================================


def _render_html(
    *,
    batch_ts: str,
    section_cards: dict[str, list[dict]],
    total: int,
) -> str:
    """status 섹션 + thumbnail grid 를 단일 HTML 문자열로. 인라인 CSS/JS, 서버 불필요."""
    # 상단 카운트 칩 + 토글 체크박스.
    chips = []
    toggles = []
    for status in STATUS_ORDER:
        cards = section_cards.get(status, [])
        if not cards:
            continue
        color = _STATUS_COLOR.get(status, "#555")
        chips.append(
            f'<span class="chip" style="background:{color}">{status} '
            f'<b>{len(cards)}</b></span>'
        )
        toggles.append(
            f'<label class="tg"><input type="checkbox" checked '
            f'onchange="toggle(\'{status}\',this.checked)"> {status}</label>'
        )

    sections_html = []
    for status in STATUS_ORDER:
        cards = section_cards.get(status, [])
        if not cards:
            continue
        color = _STATUS_COLOR.get(status, "#555")
        # worst-first — correction distance 큰 순 (None 은 뒤로).
        cards = sorted(cards, key=lambda c: (c["dist"] if c["dist"] is not None else -1.0), reverse=True)
        card_html = []
        for c in cards:
            if c["thumb"]:
                img = f'<img src="{c["thumb"]}" loading="lazy" alt="{c["caption"]}">'
            else:
                img = '<div class="noimg">no overlay</div>'
            href = c["full"] or "#"
            card_html.append(
                f'<a class="card" href="{href}" target="_blank" title="{c["recipe"]}">'
                f'{img}<div class="cap"><div class="rcp">{c["recipe"]}</div>'
                f'<div class="msr">{c["msr"]}</div><div class="meta">{c["meta"]}</div></div></a>'
            )
        sections_html.append(
            f'<section class="grp" data-status="{status}">'
            f'<h2 style="border-color:{color}"><span class="dot" style="background:{color}"></span>'
            f'{status} <small>({len(cards)})</small></h2>'
            f'<div class="grid">{"".join(card_html)}</div></section>'
        )

    style = (
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#1a1a1d;color:#e6e6e6}"
        "header{position:sticky;top:0;background:#111;padding:12px 16px;border-bottom:1px solid #333;z-index:9}"
        "h1{font-size:16px;margin:0 0 8px}.chip{display:inline-block;padding:2px 8px;margin:2px;border-radius:10px;"
        "font-size:11px;color:#fff}.chip b{font-size:12px}.tg{font-size:11px;margin:2px 8px 2px 0;color:#bbb;cursor:pointer}"
        ".grp{padding:8px 16px}.grp h2{font-size:13px;border-left:4px solid;padding-left:8px;color:#ddd}"
        ".dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}"
        ".grid{display:flex;flex-wrap:wrap;gap:10px}"
        f".card{{width:{THUMB_WIDTH}px;background:#222;border:1px solid #333;border-radius:6px;overflow:hidden;"
        "text-decoration:none;color:#ddd}.card img{display:block;width:100%}"
        ".noimg{height:120px;display:flex;align-items:center;justify-content:center;color:#777;background:#2a2a2a}"
        ".cap{padding:6px 8px;font-size:11px}.rcp{color:#8ab4f8;word-break:break-all}.msr{color:#aaa}"
        ".meta{color:#ddd;margin-top:2px}small{color:#888;font-weight:normal}"
    )
    script = (
        "function toggle(s,on){document.querySelectorAll('.grp[data-status=\"'+s+'\"]')"
        ".forEach(function(e){e.style.display=on?'':'none'})}"
    )
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>align review {batch_ts}</title><style>{style}</style></head><body>"
        f"<header><h1>Align Point Correction Review — {batch_ts} "
        f"<small>({total} images)</small></h1>"
        f'<div>{"".join(chips)}</div><div style="margin-top:6px">{"".join(toggles)}</div></header>'
        f'{"".join(sections_html)}'
        f"<script>{script}</script></body></html>"
    )


# ====================================================================
# 메인 빌드.
# ====================================================================


def build_review(
    out_root: Path,
    *,
    batch_summary_path: Path | None = None,
    thumb_width: int = THUMB_WIDTH,
) -> Path | None:
    """out_root 아래 batch 결과를 status 축으로 재배치한 review_<ts>/ 를 만든다.

    반환: 생성된 index.html 경로 (recipe 가 하나도 없으면 None).
    """
    out_root = Path(out_root)
    if batch_summary_path is None:
        batch_summary_path = _latest_batch_summary(out_root)

    if batch_summary_path is not None:
        recipe_dirs = _recipe_dirs_from_batch(out_root, batch_summary_path)
        batch_ts = batch_summary_path.stem.replace("batch_summary_", "")
    else:
        print("[WARNING] batch_summary_*.json 부재 — results.jsonl 폴더를 직접 스캔합니다.")
        recipe_dirs = _all_recipe_dirs(out_root)
        batch_ts = time.strftime("%Y%m%d_%H%M%S")

    if not recipe_dirs:
        print(f"[ERROR] review 할 recipe 결과를 찾지 못했습니다: {out_root}")
        return None

    review_dir = out_root / f"review_{batch_ts}"
    by_status = review_dir / "by_status"
    # 기존 동일 review 가 있으면 thumbnail 충돌 방지 위해 비우고 다시 만든다.
    if review_dir.exists():
        shutil.rmtree(review_dir, ignore_errors=True)
    by_status.mkdir(parents=True, exist_ok=True)

    section_cards: dict[str, list[dict]] = {}
    total = 0
    for recipe_dir in recipe_dirs:
        for row, overlay_full in _iter_rows(recipe_dir):
            total += 1
            recipe_tag = _safe(f"{row.get('eqp_id')}__{row.get('class_name')}__{row.get('recipe_id')}")
            msr = row.get("msr_image") or "unknown"
            stem = Path(msr).stem
            dist = row.get("correction_magnitude_px")
            meta = _card_meta(row)
            for status in _sections_for_row(row):
                thumb_rel = None
                if overlay_full is not None:
                    thumb_name = f"{recipe_tag}__{stem}.jpg"
                    thumb_path = by_status / status / thumb_name
                    if _make_thumb(overlay_full, thumb_path, width=thumb_width):
                        thumb_rel = f"by_status/{status}/{thumb_name}"
                full_rel = (
                    os.path.relpath(overlay_full, review_dir) if overlay_full is not None else None
                )
                section_cards.setdefault(status, []).append({
                    "recipe": recipe_tag,
                    "msr": msr,
                    "meta": meta,
                    "dist": float(dist) if isinstance(dist, (int, float)) else None,
                    "thumb": thumb_rel,
                    "full": full_rel,
                    "caption": f"{recipe_tag} {msr}",
                })

    html = _render_html(batch_ts=batch_ts, section_cards=section_cards, total=total)
    index_path = review_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    review_summary = {
        "batch_ts": batch_ts,
        "out_root": str(out_root),
        "recipe_count": len(recipe_dirs),
        "total_images": total,
        "section_counts": {k: len(v) for k, v in section_cards.items()},
        "index_html": str(index_path),
    }
    (review_dir / "review_summary.json").write_text(
        json.dumps(review_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"[INFO] review 생성 완료 → {index_path}")
    print(f"[INFO] {len(recipe_dirs)} recipes, {total} images")
    for status in STATUS_ORDER:
        n = len(section_cards.get(status, []))
        if n:
            print(f"        - {status}: {n}")
    return index_path


# ====================================================================
# 합성 self-test (데이터 부재 환경 — Mac dev).
# ====================================================================


def _synthetic_overlay(text: str, color: tuple[int, int, int]) -> np.ndarray:
    """가짜 overlay 한 장 — 단색 배경 + 라벨 텍스트."""
    img = np.full((240, 360, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (350, 230), color, 2)
    cv2.putText(img, text, (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def _build_synthetic_fixture(out_root: Path) -> Path:
    """recipe 2개 × msr 몇 장 분량의 가짜 batch 트리를 만든다. batch_summary 경로 반환."""
    recipes = [
        ("EQP01", "CLASS_A", "RCP_alpha", [
            ("S01_A0001-01AP", "ok", 2.0, False),
            ("E02_A0002-01AP", "not_distinctive", 41.0, False),
            ("S03_A0003-01AP", "ok", 28.0, True),   # suspect: S 라벨인데 dist 큼.
            ("E04_A0004-01AP", "msr_unrecognizable", None, False),
        ]),
        ("EQP02", "CLASS_B", "RCP_beta", [
            ("S01_A0001-01AP", "low_match_both", 60.0, False),
            ("S02_A0002-01AP", "no_crosshair_drawn", 12.0, False),
            ("E03_A0003-01AP", "already_aligned", 1.0, False),
        ]),
    ]
    processed = []
    for eqp, cls, rcp, rows_spec in recipes:
        ts = "20260529_000000"
        out_dir = out_root / f"{eqp}__{cls}__{rcp}__{ts}"
        overlay_dir = out_dir / "overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for msr, status, dist, suspect in rows_spec:
            color = (60, 60, 220) if status != "ok" else (80, 200, 80)
            cv2.imwrite(str(overlay_dir / f"{msr}_overlay.jpg"),
                        _synthetic_overlay(f"{msr} [{status}]", color),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({
                "recipe_id": rcp, "eqp_id": eqp, "class_name": cls, "msr_image": f"{msr}.jpg",
                "visit_order": int(msr[1:3]) if msr[1:3].isdigit() else None,
                "tool_label": msr[0], "winner_modality": "sem", "winner_score": 0.7,
                "correction_magnitude_px": dist, "tool_label_suspect": suspect, "status": status,
            })
        with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        processed.append({"recipe_dir": str(out_dir), "out_dir": str(out_dir),
                          "total_msr_images": len(rows)})
    batch_path = out_root / "batch_summary_20260529_000000.json"
    batch_path.write_text(json.dumps({"processed_recipes": processed}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return batch_path


def _self_test() -> bool:
    """합성 fixture 로 build_review 를 돌리고 산출물 존재/그룹핑을 검증한다."""
    tmp = Path(tempfile.mkdtemp(prefix="align_review_selftest_"))
    try:
        out_root = tmp / "align_correction"
        out_root.mkdir(parents=True)
        batch_path = _build_synthetic_fixture(out_root)
        index = build_review(out_root, batch_summary_path=batch_path)
        assert index is not None and index.is_file(), "index.html 미생성"
        html = index.read_text(encoding="utf-8")
        # suspect_success 섹션이 있어야 한다 (S03 가 suspect=True).
        assert 'data-status="suspect_success"' in html, "suspect_success 섹션 누락"
        assert 'data-status="not_distinctive"' in html, "not_distinctive 섹션 누락"
        # by_status 폴더에 thumbnail 이 복사됐는지.
        thumbs = list((index.parent / "by_status").rglob("*.jpg"))
        assert thumbs, "thumbnail 미복사"
        summary = json.loads((index.parent / "review_summary.json").read_text(encoding="utf-8"))
        assert summary["total_images"] == 7, f"total 불일치: {summary['total_images']}"
        # suspect 는 actual status(ok) + suspect_success 양쪽 카운트 → ok 1 + suspect 1.
        assert summary["section_counts"].get("suspect_success") == 1, "suspect 카운트 오류"
        print("[INFO] self-test 통과 — index.html, by_status thumbnails, 그룹핑 검증 완료.")
        print(f"[INFO] (합성 결과 위치, 확인용) {index}")
        # self-test 산출물은 임시 — 확인 후 정리. 보고 싶으면 위 경로 복사.
        return True
    except AssertionError as exc:
        print(f"[ERROR] self-test 실패: {exc}")
        return False
    finally:
        # tmp 는 남겨두면 디스크 누수 — 검증만 하고 지운다.
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> str:
    out_root = DEBUG_IMAGE_DIR / "align_correction"
    if out_root.is_dir() and (
        _latest_batch_summary(out_root) is not None or list(out_root.glob("*/results.jsonl"))
    ):
        index = build_review(out_root)
        return "success" if index is not None else "no_data"
    print("[WARNING] 실제 align_correction 결과 없음 — 합성 self-test 로 대체합니다.")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
