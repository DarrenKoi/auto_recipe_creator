"""Phase 0 하베스트 - DRM 적용 전 디지털 PDF에서 구조 레이어를 통째로 떠두는 일회성 스크립트.

목적
----
이 매뉴얼 PDF은 곧 회사 DRM 소프트웨어로 잠긴다(스케줄 적용). DRM 이 적용되면
텍스트 레이어/표/그림을 다시는 깔끔히 뽑을 수 없으므로, **지금 디지털 상태일 때**
같은 페이지를 서로 독립적인 네 경로로 *중복* 수확해 둔다(한 추출기가 놓쳐도 다른
추출기가 잡도록). 이 번들만 확보되면 RAG 빌드(Phase 1+)는 나중에 사외에서도 반복
가능하다.

수확하는 것(페이지별)
----------------------
1. 구조 텍스트  get_text("dict") - block/line/span + bbox + font/size (읽기순서·heading 추정).
   추가로 reading-order 평문(.txt)도 같이 저장.
2. 네이티브 표  PyMuPDF find_tables()(항상) + pdfplumber(설치돼 있으면) - OCR 아닌 *정확한* 셀.
3. 임베디드 그림  get_images() -> extract_image(xref) 원본 바이트(native 해상도). xref 기준 전역 dedup.
4. 페이지 렌더  get_pixmap(dpi) 무손실 PNG - 벡터 다이어그램(xref 아님) 포착 + DRM 후 OCR 폴백.

그 외 문서 단위로 메타데이터 / TOC(목차) / 페이지별 링크(상호참조)도 저장한다.

일회성 안전 설계(복구 불가이므로)
---------------------------------
- **단계별 독립 try/except**: 텍스트/표/그림/렌더/링크 각각을 따로 감싼다. 한 단계가
  깨져도 나머지 단계(특히 DRM 후 유일 소스인 렌더 PNG)는 그대로 저장된다.
- **렌더 먼저**: 가장 대체 불가한 페이지 PNG 를 맨 먼저 떠서 앞 단계 실패에 영향받지 않게.
- **점진적 manifest flush**: MANIFEST_FLUSH_EVERY 페이지마다 manifest 를 디스크에 쓴다.
  중간에 죽어도 어디까지 됐는지·실패가 무엇인지 기록이 남는다.
- **resume/skip**: 페이지 완료 시 sentinel(done/<stem>.done)을 남기고, 재실행 시 이미
  끝난 페이지는 건너뛴다(RESUME). 비싼 렌더를 다시 돌려 DRM 창을 날리지 않게.
- **디스크 가드**: 시작 시 여유 공간을 추정·경고(무손실 PNG x 페이지수는 수 GB).

원칙
----
- **무손실 우선**: 아카이브 렌더는 PNG(무손실). VLM 에 보낼 WebP 변환은 빌드 단계의 몫.
- **자급자족**: 패키지 내부 import 없음. PyMuPDF 만 필수, pdfplumber 는 선택.
- **CLI 인자 없음**: 실행 전 아래 상수를 직접 채운다.

실행
----
    uv pip install pdfplumber        # (선택) 표 2차 추출기 - 매뉴얼은 표가 많으니 권장
    uv run python side_projects/document_extraction/harvest/harvest_pdf.py
"""

import json
import shutil
import time
from pathlib import Path

import fitz  # PyMuPDF

# === 실행 전 매번 채워 넣을 것 =================================================
PDF_PATH: Path = Path(r"")          # 예: Path(r"C:\Users\me\Documents\tool_manual.pdf")
OUTPUT_DIR: Path = Path(r"")        # 예: Path(r"D:\harvest")  (여유 공간 넉넉한 드라이브)
RENDER_DPI: int = 200               # 아카이브 페이지 렌더 DPI. 표 글씨가 빽빽하면 300 권장(용량↑).
SAVE_RENDER: bool = True            # 페이지 PNG 렌더 저장(벡터 다이어그램·OCR 폴백). 끄지 말 것 권장.
SAVE_FIGURES: bool = True           # 임베디드 그림 원본 바이트 추출.
SAVE_TABLES: bool = True            # 표 추출(PyMuPDF + pdfplumber).
USE_PDFPLUMBER: bool = True         # pdfplumber 설치돼 있으면 표 2차 추출기로 사용.
RESUME: bool = True                 # 이미 완료된 페이지(done sentinel)는 건너뜀(재실행 회복).
OVERWRITE: bool = False             # True 면 RESUME 무시하고 전부 다시 수확.
PAGE_LIMIT: int = 0                 # 0=전체. 리허설 시 앞 N페이지만(예: 20)으로 한 번 점검 권장.
MANIFEST_FLUSH_EVERY: int = 25      # N페이지마다 manifest.json 을 디스크에 flush.
# ==============================================================================

# pdfplumber 는 선택 의존성 - 프로젝트 import-guard 패턴.
try:
    import pdfplumber  # type: ignore
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None  # type: ignore
    PDFPLUMBER_AVAILABLE = False


# ── 직렬화 헬퍼 ───────────────────────────────────────────────────────────────
def _clean(obj):
    """fitz.Rect/Point/IRect, bytes, set 등을 JSON 직렬화 가능한 형태로 재귀 변환."""
    if isinstance(obj, (fitz.Rect, fitz.IRect, fitz.Point)):
        return list(obj)
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes omitted>"
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, set):
        return [_clean(v) for v in obj]
    return obj


def _sanitize_text_dict(td: dict) -> dict:
    """get_text('dict') 안의 image block 은 원본 바이트('image' 키)를 담는다 - JSON 깨지므로 제거.
    그림은 어차피 figures/ 로 따로 추출하니 여기선 메타만 남긴다."""
    for block in td.get("blocks", []):
        if "image" in block:
            img = block["image"]
            block["image"] = f"<{len(img)} bytes omitted>" if isinstance(img, (bytes, bytearray)) else img
    return _clean(td)


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _ensure_dirs(root: Path) -> dict:
    dirs = {
        "text": root / "text",
        "tables": root / "tables",
        "figures": root / "figures",
        "figures_by_xref": root / "figures" / "by_xref",
        "render": root / "render",
        "links": root / "links",
        "done": root / "done",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── 페이지별 수확 (각자 독립 단계, 호출부에서 개별 try/except) ─────────────────
def harvest_text(page: fitz.Page, dirs: dict, stem: str) -> bool:
    """구조 텍스트(dict) + 평문(txt). 텍스트가 한 글자라도 있으면 True."""
    td = page.get_text("dict")
    _save_json(dirs["text"] / f"{stem}.json", _sanitize_text_dict(td))
    plain = page.get_text("text")
    (dirs["text"] / f"{stem}.txt").write_text(plain, encoding="utf-8")
    return bool(plain.strip())


def harvest_tables(page: fitz.Page, pdfplumber_page, dirs: dict, stem: str) -> int:
    """PyMuPDF find_tables(항상) + pdfplumber(있으면). 두 출처를 source 태그로 함께 보존.
    각 추출기는 따로 try - 하나가 깨져도 다른 출처는 남는다."""
    out = []
    try:
        finder = page.find_tables()
        for t in finder.tables:
            out.append({"source": "pymupdf", "bbox": _clean(t.bbox), "rows": t.extract(),
                        "row_count": getattr(t, "row_count", None),
                        "col_count": getattr(t, "col_count", None)})
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] {stem}: pymupdf find_tables 실패: {exc}")
    if pdfplumber_page is not None:
        try:
            for tbl in pdfplumber_page.extract_tables():
                out.append({"source": "pdfplumber", "bbox": None, "rows": tbl,
                            "row_count": len(tbl), "col_count": len(tbl[0]) if tbl else 0})
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {stem}: pdfplumber 표 추출 실패: {exc}")
        finally:
            # pdfplumber 페이지 캐시 해제 - 2000페이지 누적 OOM 방지.
            if hasattr(pdfplumber_page, "flush_cache"):
                try:
                    pdfplumber_page.flush_cache()
                except Exception:  # noqa: BLE001
                    pass
    if out:
        _save_json(dirs["tables"] / f"{stem}.json", out)
    return len(out)


def harvest_figures(doc: fitz.Document, page: fitz.Page, dirs: dict, stem: str,
                    seen_xrefs: dict) -> int:
    """임베디드 raster 그림을 원본 바이트로 추출. xref 전역 dedup(반복 로고 폭발 방지).
    같은 xref 가 한 페이지에 여러 번 배치되면 bbox 를 리스트로 모두 보존.
    그림 한 장의 추출/쓰기 실패가 페이지 전체를 막지 않도록 그림별로 try."""
    # 페이지 내 이미지 배치 bbox (xref -> [bbox, ...])
    bboxes_by_xref: dict = {}
    try:
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref")
            if xref:
                bboxes_by_xref.setdefault(xref, []).append(_clean(info.get("bbox")))
    except Exception:  # noqa: BLE001
        pass

    refs = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            if xref not in seen_xrefs:
                base = doc.extract_image(xref)
                image_bytes = base.get("image")
                if not image_bytes:
                    print(f"[WARNING] {stem}: xref {xref} 이미지 바이트 없음, 건너뜀")
                    continue
                ext = base.get("ext", "png")
                fname = f"xref_{xref:06d}.{ext}"
                (dirs["figures_by_xref"] / fname).write_bytes(image_bytes)
                seen_xrefs[xref] = {"file": f"by_xref/{fname}", "ext": ext,
                                    "width": base.get("width"), "height": base.get("height"),
                                    "colorspace": base.get("colorspace")}
            meta = dict(seen_xrefs[xref])
            meta["xref"] = xref
            meta["bboxes_on_page"] = bboxes_by_xref.get(xref, [])
            refs.append(meta)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {stem}: xref {xref} 그림 처리 실패(계속): {exc}")
            continue
    if refs:
        _save_json(dirs["figures"] / f"{stem}.json", refs)
    return len(refs)


def render_page(page: fitz.Page, dirs: dict, stem: str) -> bool:
    """페이지 무손실 PNG 렌더. 벡터 다이어그램 포착 + DRM 후 OCR 폴백 소스."""
    pix = page.get_pixmap(dpi=RENDER_DPI)
    pix.save(str(dirs["render"] / f"{stem}.png"))
    return True


def harvest_links(page: fitz.Page, dirs: dict, stem: str) -> int:
    """하이퍼링크/상호참조."""
    links = [_clean(lk) for lk in page.get_links()]
    if links:
        _save_json(dirs["links"] / f"{stem}.json", links)
    return len(links)


def _run_step(name: str, fn, rec: dict, manifest: dict, page_no: int, stem: str, default):
    """한 수확 단계를 독립 try 로 실행. 실패해도 다른 단계에 영향 없이 default 반환 + 기록."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        manifest["failures"].append({"page": page_no, "step": name, "error": str(exc)})
        rec.setdefault("errors", []).append(name)
        print(f"[ERROR] {stem}: {name} 실패(계속 진행): {exc}")
        return default


def _estimate_and_warn_disk(root: Path, page_count: int) -> None:
    """무손실 PNG x 페이지수는 수 GB. 시작 시 여유 공간을 추정·경고."""
    try:
        free_gb = shutil.disk_usage(root).free / 1024 ** 3
    except Exception:  # noqa: BLE001
        return
    per_page_mb = (0.6 * (RENDER_DPI / 200.0) ** 2 if SAVE_RENDER else 0.0) + 0.25
    need_gb = page_count * per_page_mb / 1024
    print(f"[INFO] 디스크: 여유 {free_gb:.1f}GB, 예상 필요 ~{need_gb:.1f}GB "
          f"(렌더 {RENDER_DPI}DPI 가정)")
    if free_gb < need_gb * 1.3:
        print(f"[WARNING] 여유 공간이 빠듯합니다(필요 추정의 1.3배 미만). 다른 드라이브를 "
              f"OUTPUT_DIR 로 쓰거나 RENDER_DPI 를 낮추는 것을 검토하세요.")


# ── 오케스트레이션 ────────────────────────────────────────────────────────────
def main() -> None:
    if not PDF_PATH or not str(PDF_PATH).strip():
        raise SystemExit("[ERROR] PDF_PATH 를 채워 주세요 (스크립트 상단 상수).")
    if not OUTPUT_DIR or not str(OUTPUT_DIR).strip():
        raise SystemExit("[ERROR] OUTPUT_DIR 를 채워 주세요 (스크립트 상단 상수).")
    if not PDF_PATH.exists():
        raise SystemExit(f"[ERROR] PDF 가 없습니다: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    if doc.needs_pass:
        raise SystemExit("[ERROR] PDF 가 암호/DRM 으로 잠겨 있습니다. 이미 늦었을 수 있습니다 "
                         "(DRM 적용 전 디지털 원본이 필요).")
    if doc.is_encrypted:
        print("[WARNING] 문서가 encrypted 로 표시됩니다. 텍스트가 비어 나오면 DRM 이 이미 적용된 것.")

    total_pages = doc.page_count
    root = OUTPUT_DIR / PDF_PATH.stem
    root.mkdir(parents=True, exist_ok=True)
    dirs = _ensure_dirs(root)

    print(f"[INFO] PDF: {PDF_PATH}  페이지수={total_pages}")
    print(f"[INFO] 출력: {root}")
    print(f"[INFO] pdfplumber: {'사용' if (USE_PDFPLUMBER and PDFPLUMBER_AVAILABLE) else '미사용'}"
          f"  렌더DPI={RENDER_DPI}  렌더={SAVE_RENDER} 그림={SAVE_FIGURES} 표={SAVE_TABLES}"
          f"  RESUME={RESUME} OVERWRITE={OVERWRITE}")
    _estimate_and_warn_disk(root, total_pages if PAGE_LIMIT == 0 else min(PAGE_LIMIT, total_pages))

    # 문서 단위 메타 + 목차 (page_sizes 는 per-page 루프에서 모은다 - 전체 사전스캔 금지).
    _save_json(root / "metadata.json", {
        "source_pdf": str(PDF_PATH), "page_count": total_pages,
        "metadata": _clean(doc.metadata),
    })
    try:
        # get_toc(simple=False) 의 dest 는 fitz.Point 를 담으므로 반드시 _clean.
        _save_json(root / "toc.json", _clean(doc.get_toc(simple=False)))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] 목차(TOC) 추출 실패: {exc}")

    # pdfplumber 는 자체 핸들로 한 번 연다(같은 루프에서 페이지 인덱싱).
    plumber_doc = None
    if SAVE_TABLES and USE_PDFPLUMBER and PDFPLUMBER_AVAILABLE:
        try:
            plumber_doc = pdfplumber.open(PDF_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] pdfplumber open 실패, PyMuPDF 표만 사용: {exc}")

    n_pages = total_pages if PAGE_LIMIT == 0 else min(PAGE_LIMIT, total_pages)

    # 재실행 회복: 기존 manifest 의 per_page 를 이어받아 요약 정확도 유지.
    manifest = {"per_page": [], "failures": [], "tools": {
        "pymupdf": fitz.VersionBind,
        "pdfplumber": getattr(pdfplumber, "__version__", None) if PDFPLUMBER_AVAILABLE else None,
        "render_dpi": RENDER_DPI,
    }}
    prior_recs: dict = {}
    mpath = root / "manifest.json"
    if RESUME and not OVERWRITE and mpath.exists():
        try:
            old = json.loads(mpath.read_text(encoding="utf-8"))
            for r in old.get("per_page", []):
                prior_recs[r.get("page")] = r
            manifest["failures"] = old.get("failures", [])
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] 기존 manifest 읽기 실패(무시): {exc}")

    seen_xrefs: dict = {}
    t0 = time.time()
    pages_with_text = 0

    for pno in range(n_pages):
        stem = f"page_{pno + 1:04d}"
        done_marker = dirs["done"] / f"{stem}.done"

        # resume: 이미 완료된 페이지는 건너뛴다(비싼 렌더 재실행 방지).
        if RESUME and not OVERWRITE and done_marker.exists():
            rec = prior_recs.get(pno + 1, {"page": pno + 1, "skipped": True})
            manifest["per_page"].append(rec)
            pages_with_text += int(bool(rec.get("text")))
            continue

        rec = {"page": pno + 1, "text": False, "tables": 0, "figures": 0,
               "links": 0, "rendered": False, "size": None}
        try:
            page = doc.load_page(pno)
        except Exception as exc:  # noqa: BLE001
            manifest["failures"].append({"page": pno + 1, "step": "load_page", "error": str(exc)})
            rec["errors"] = ["load_page"]
            print(f"[ERROR] {stem} load_page 실패(계속 진행): {exc}")
            manifest["per_page"].append(rec)
            continue

        # 렌더를 먼저(가장 대체 불가). 각 단계는 독립 try.
        if SAVE_RENDER:
            rec["rendered"] = bool(_run_step("render", lambda: render_page(page, dirs, stem),
                                             rec, manifest, pno + 1, stem, False))
        rec["text"] = bool(_run_step("text", lambda: harvest_text(page, dirs, stem),
                                     rec, manifest, pno + 1, stem, False))
        pages_with_text += int(rec["text"])
        if SAVE_TABLES:
            pl_page = plumber_doc.pages[pno] if (plumber_doc and pno < len(plumber_doc.pages)) else None
            rec["tables"] = _run_step("tables", lambda: harvest_tables(page, pl_page, dirs, stem),
                                      rec, manifest, pno + 1, stem, 0) or 0
        if SAVE_FIGURES:
            rec["figures"] = _run_step("figures",
                                       lambda: harvest_figures(doc, page, dirs, stem, seen_xrefs),
                                       rec, manifest, pno + 1, stem, 0) or 0
        rec["links"] = _run_step("links", lambda: harvest_links(page, dirs, stem),
                                 rec, manifest, pno + 1, stem, 0) or 0
        rec["size"] = _run_step("size",
                                lambda: [round(page.rect.width, 1), round(page.rect.height, 1)],
                                rec, manifest, pno + 1, stem, None)

        manifest["per_page"].append(rec)
        # 완전 성공(어떤 단계도 실패 안 함)일 때만 완료 sentinel. 부분 실패 페이지는
        # sentinel 을 남기지 않아 RESUME 재실행 시 빠진 단계를 다시 채운다.
        if "errors" not in rec:
            done_marker.write_text("", encoding="utf-8")

        flag = "" if rec["text"] else "  [NO-TEXT: 이미지페이지일 수 있음, 렌더 의존]"
        print(f"[INFO] {stem}  text={int(rec['text'])} tables={rec['tables']} "
              f"figs={rec['figures']} links={rec['links']}{flag}")

        # 점진적 manifest flush - 중간에 죽어도 진행/실패 기록이 남는다.
        if (pno + 1) % MANIFEST_FLUSH_EVERY == 0:
            _flush_manifest(mpath, manifest, seen_xrefs, total_pages, pages_with_text, t0)

    if plumber_doc is not None:
        plumber_doc.close()

    _flush_manifest(mpath, manifest, seen_xrefs, total_pages, pages_with_text, t0)
    summary = manifest["summary"]
    doc.close()

    print("\n[INFO] ===== 하베스트 완료 - 검증 요약 =====")
    print(f"[INFO] 처리 페이지       : {summary['pages_processed']} / {total_pages}")
    print(f"[INFO] 텍스트 있는 페이지 : {summary['pages_with_text']}  (없음 {len(summary['pages_no_text'])})")
    print(f"[INFO] 표(중복 출처 포함) : {summary['total_tables']}")
    print(f"[INFO] 그림 고유/참조     : {summary['unique_figures']} / {summary['total_figure_refs']}")
    print(f"[INFO] 렌더 PNG          : {summary['pages_rendered']}")
    print(f"[INFO] 실패(단계 단위)    : {summary['failures']}  (manifest.failures 참조)")
    print(f"[INFO] 소요               : {summary['elapsed_sec']}s")
    no_text = summary["pages_no_text"]
    if no_text:
        head = ", ".join(map(str, no_text[:20]))
        more = " ..." if len(no_text) > 20 else ""
        print(f"[WARNING] 텍스트 0 페이지({len(no_text)}개): {head}{more}")
        print("[WARNING] 이 페이지들은 이미지/벡터 전용일 수 있음. render PNG 가 유일 소스이니 "
              "DRM 전 렌더가 제대로 남았는지 꼭 확인.")
    if summary["failures"]:
        print("[ERROR] 실패 단계가 있습니다. DRM 적용 전이라면 RENDER_DPI 를 낮추거나 "
              "RESUME 으로 재실행해 빠진 페이지만 채우세요.")
    print(f"[INFO] 번들 위치: {root}")
    print("[INFO] 이 폴더 전체가 복구 불가능한 자산입니다. DRM 적용 전 백업까지 끝내세요.")


def _flush_manifest(mpath: Path, manifest: dict, seen_xrefs: dict,
                    total_pages: int, pages_with_text: int, t0: float) -> None:
    """manifest 에 요약을 채워 디스크로 쓴다(점진적 flush + 최종)."""
    per_page = manifest["per_page"]
    manifest["summary"] = {
        "pages_processed": len(per_page),
        "pages_with_text": sum(int(bool(r.get("text"))) for r in per_page),
        "pages_no_text": [r["page"] for r in per_page if not r.get("text")],
        "total_tables": sum(r.get("tables", 0) for r in per_page),
        "unique_figures": len(seen_xrefs),
        "total_figure_refs": sum(r.get("figures", 0) for r in per_page),
        "pages_rendered": sum(int(bool(r.get("rendered"))) for r in per_page),
        "failures": len(manifest["failures"]),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    _save_json(mpath, manifest)


if __name__ == "__main__":
    main()
