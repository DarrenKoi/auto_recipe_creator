"""consensus_gather 합성 self-test — office/실장비 없이 stage/replace/error 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/align/test_consensus_gather.py
"""

import shutil
import tempfile
import time as _time
from pathlib import Path as _Path
from pathlib import Path

from poc.workflow_3.align.consensus_gather import (
    StagedEvent,
    gather_success_images,
    _events_dir_for,
)


class _FakeDownloader:
    """dest_dir 에 합성 S*.jpeg + .<이미지명>/cond.txt 를 쓰는 테스트용 다운로더.

    실제 office 규약(숨김폴더 cond, 결정 2026-06-10)을 모델링한다.
    events_spec: list[(event_id, n_images)]. raise_exc=True 면 예외를 던진다.
    """

    def __init__(self, events_spec, *, raise_exc=False):
        self.events_spec = events_spec
        self.raise_exc = raise_exc
        self.calls = []

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
        self.calls.append((recipe_id, max_events, Path(dest_dir)))
        if self.raise_exc:
            raise RuntimeError("DB 연결 실패")
        staged = []
        for event_id, n_images in self.events_spec[:max_events]:
            ev_dir = Path(dest_dir) / event_id
            ev_dir.mkdir(parents=True, exist_ok=True)
            imgs, conds = [], []
            for i in range(n_images):
                img = ev_dir / f"S{i + 1:04d}.jpeg"
                cond_dir = ev_dir / f".S{i + 1:04d}.jpeg"
                cond_dir.mkdir(parents=True, exist_ok=True)
                cond = cond_dir / "cond.txt"
                img.write_bytes(b"\xff\xd8\xff\xd9")
                cond.write_text("!Cursor_info 10,20\n", encoding="utf-8")
                imgs.append(img)
                conds.append(cond)
            staged.append(StagedEvent(event_id=event_id, image_paths=imgs, cond_paths=conds))
        return staged


def _staged_files(events_dir):
    """events_dir 아래 모든 파일의 상대경로 set (검증용)."""
    if not events_dir.exists():
        return set()
    return {p.relative_to(events_dir).as_posix() for p in events_dir.rglob("*") if p.is_file()}


def test_stage_basic():
    root = Path(tempfile.mkdtemp())
    try:
        dl = _FakeDownloader([("EV1", 2), ("EV2", 2)])
        res = gather_success_images("EQP1", "CLS/RCP", downloader=dl, cache_root=root)
        files = _staged_files(res.events_dir)
        ok = (
            res.reason == "ok"
            and res.n_events == 2
            and res.n_images == 4
            and "EV1/S0001.jpeg" in files
            and "EV1/.S0001.jpeg/cond.txt" in files
            and "EV2/S0002.jpeg" in files
        )
        print(f"[{'PASS' if ok else 'FAIL'}] stage_basic: reason={res.reason} "
              f"events={res.n_events} images={res.n_images} files={len(files)}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_layout_nested():
    root = Path(tempfile.mkdtemp())
    try:
        dl = _FakeDownloader([("EV1", 1)])
        res = gather_success_images("EQP1", "CLS/RCP", downloader=dl, cache_root=root)
        expected = root / "EQP1" / "CLS" / "RCP" / "events"
        ok = res.events_dir == expected and (expected / "EV1" / "S0001.jpeg").exists()
        print(f"[{'PASS' if ok else 'FAIL'}] layout_nested: events_dir={res.events_dir}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_replace_swaps_to_latest():
    root = Path(tempfile.mkdtemp())
    try:
        gather_success_images("EQP1", "CLS/RCP",
                              downloader=_FakeDownloader([("OLD_A", 1), ("OLD_B", 1)]),
                              cache_root=root)
        res = gather_success_images("EQP1", "CLS/RCP",
                                    downloader=_FakeDownloader([("NEW_C", 1)]),
                                    cache_root=root)
        event_dirs = {f.split("/")[0] for f in _staged_files(res.events_dir)}
        ok = res.reason == "ok" and event_dirs == {"NEW_C"}  # 옛 set 사라짐.
        print(f"[{'PASS' if ok else 'FAIL'}] replace_swaps: event_dirs={event_dirs}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_empty_preserves_existing():
    root = Path(tempfile.mkdtemp())
    try:
        gather_success_images("EQP1", "CLS/RCP",
                              downloader=_FakeDownloader([("KEEP_A", 1)]),
                              cache_root=root)
        res = gather_success_images("EQP1", "CLS/RCP",
                                    downloader=_FakeDownloader([]),  # 빈 fetch.
                                    cache_root=root)
        event_dirs = {f.split("/")[0] for f in _staged_files(res.events_dir)}
        ok = res.reason == "empty" and event_dirs == {"KEEP_A"}  # 기존 보존.
        print(f"[{'PASS' if ok else 'FAIL'}] empty_preserves: reason={res.reason} dirs={event_dirs}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_downloader_raises():
    root = Path(tempfile.mkdtemp())
    try:
        gather_success_images("EQP1", "CLS/RCP",
                              downloader=_FakeDownloader([("KEEP_A", 1)]),
                              cache_root=root)
        res = gather_success_images("EQP1", "CLS/RCP",
                                    downloader=_FakeDownloader([], raise_exc=True),
                                    cache_root=root)
        event_dirs = {f.split("/")[0] for f in _staged_files(res.events_dir)}
        staging = root / "EQP1" / "CLS" / "RCP" / ".events_staging"
        ok = (
            res.reason.startswith("error:")
            and event_dirs == {"KEEP_A"}        # 기존 set 보존.
            and not staging.exists()            # staging 정리됨.
        )
        print(f"[{'PASS' if ok else 'FAIL'}] downloader_raises: reason={res.reason!r} "
              f"staging_exists={staging.exists()}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_skipped_no_recipe():
    root = Path(tempfile.mkdtemp())
    try:
        dl = _FakeDownloader([("EV1", 1)])
        res = gather_success_images("EQP1", "", downloader=dl, cache_root=root)
        ok = res.reason == "skipped" and res.n_events == 0 and len(dl.calls) == 0
        print(f"[{'PASS' if ok else 'FAIL'}] skipped_no_recipe: reason={res.reason} calls={len(dl.calls)}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_malformed_event_reports_error():
    """malformed StagedEvent(image_paths=None) 이 TypeError 를 유발해도 GatherResult 로 보고."""
    root = Path(tempfile.mkdtemp())
    try:
        # 기존 캐시 seed
        gather_success_images("EQP1", "CLS/RCP",
                              downloader=_FakeDownloader([("KEEP_A", 1)]),
                              cache_root=root)

        class _MalformedDownloader:
            """StagedEvent(image_paths=None) 을 반환하는 불량 다운로더."""
            calls = []

            def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
                self.calls.append((recipe_id, max_events, Path(dest_dir)))
                # image_paths=None 으로 malformed StagedEvent 반환
                from poc.workflow_3.align.consensus_gather import StagedEvent
                return [StagedEvent(event_id="BAD", image_paths=None, cond_paths=[])]

        dl = _MalformedDownloader()
        res = gather_success_images("EQP1", "CLS/RCP", downloader=dl, cache_root=root)
        event_dirs = {f.split("/")[0] for f in _staged_files(res.events_dir)}
        staging = root / "EQP1" / "CLS" / "RCP" / ".events_staging"
        ok = (
            res.reason.startswith("error:swap:")   # swap/count 예외 경로
            and event_dirs == {"KEEP_A"}            # 기존 캐시 보존
            and not staging.exists()               # staging 정리됨
        )
        print(f"[{'PASS' if ok else 'FAIL'}] malformed_event_reports_error: "
              f"reason={res.reason!r} dirs={event_dirs} staging_exists={staging.exists()}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_max_events_passthrough():
    """max_events 가 다운로더에게 그대로 전달되고 결과도 그에 맞게 잘린다."""
    root = Path(tempfile.mkdtemp())
    try:
        dl = _FakeDownloader([("EV1", 1), ("EV2", 1), ("EV3", 1), ("EV4", 1), ("EV5", 1)])
        res = gather_success_images("EQP1", "CLS/RCP", downloader=dl,
                                    max_events=3, cache_root=root)
        ok = (
            dl.calls[0][1] == 3    # max_events=3 이 다운로더에 전달됨
            and res.n_events == 3  # 결과도 3건
        )
        print(f"[{'PASS' if ok else 'FAIL'}] max_events_passthrough: "
              f"dl_max={dl.calls[0][1]} n_events={res.n_events}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _StubEvent:
    """TTL/atomic-swap 테스트용 최소 stub event (StagedEvent 대신 duck-typing)."""

    def __init__(self, n_images):
        self.event_id = "20260612_090000_r_lot"
        self.image_paths = [_Path(f"S{i}.jpeg") for i in range(n_images)]
        self.cond_paths = []


class _Downloader:
    """TTL/atomic-swap 테스트용 최소 stub 다운로더."""

    def __init__(self, events):
        self.events = events
        self.calls = 0

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
        self.calls += 1
        for ev in self.events:
            d = _Path(dest_dir) / ev.event_id
            d.mkdir(parents=True, exist_ok=True)
            for ip in ev.image_paths:
                (d / ip.name).write_bytes(b"x")
        return self.events


def test_zero_image_event_preserves_old_cache(tmp_path=None):
    """이미지 0장인 event 가 있어도 n_images==0 이면 기존 캐시 보존."""
    import tempfile
    root = _Path(tmp_path) if tmp_path else _Path(tempfile.mkdtemp())
    try:
        ev_dir = _events_dir_for("E1", "c/r", root)
        (ev_dir / "old").mkdir(parents=True)
        (ev_dir / "old" / "S0.jpeg").write_bytes(b"x")
        dl = _Downloader([_StubEvent(0)])           # 이미지 0장 event.
        res = gather_success_images("E1", "c/r", downloader=dl, cache_root=root,
                                    refresh_ttl_sec=0)
        ok = res.reason == "empty" and (ev_dir / "old" / "S0.jpeg").exists()
        print(f"[{'PASS' if ok else 'FAIL'}] zero_image_event_preserves_old_cache: "
              f"reason={res.reason} old_cache_exists={(ev_dir / 'old' / 'S0.jpeg').exists()}")
        return ok
    finally:
        if tmp_path is None:
            shutil.rmtree(root, ignore_errors=True)


def test_ttl_skips_download(tmp_path=None):
    """TTL 이내 캐시는 다운로더를 호출하지 않고 'fresh' 반환."""
    import tempfile
    root = _Path(tmp_path) if tmp_path else _Path(tempfile.mkdtemp())
    try:
        ev_dir = _events_dir_for("E1", "c/r", root)
        (ev_dir / "e1").mkdir(parents=True)
        (ev_dir / "e1" / "S0.jpeg").write_bytes(b"x")
        dl = _Downloader([_StubEvent(2)])
        res = gather_success_images("E1", "c/r", downloader=dl, cache_root=root,
                                    refresh_ttl_sec=3600)  # 방금 만든 캐시 -> TTL 내.
        ok = res.reason == "fresh" and dl.calls == 0
        print(f"[{'PASS' if ok else 'FAIL'}] ttl_skips_download: "
              f"reason={res.reason} dl_calls={dl.calls}")
        return ok
    finally:
        if tmp_path is None:
            shutil.rmtree(root, ignore_errors=True)


def test_nonempty_replaces_cache(tmp_path=None):
    """이미지 ≥1 이면 기존 캐시를 교체하고 events/ 가 유효하게 남는다."""
    import tempfile
    root = _Path(tmp_path) if tmp_path else _Path(tempfile.mkdtemp())
    try:
        ev_dir = _events_dir_for("E1", "c/r", root)
        (ev_dir / "old").mkdir(parents=True)
        (ev_dir / "old" / "S0.jpeg").write_bytes(b"x")
        dl = _Downloader([_StubEvent(2)])
        res = gather_success_images("E1", "c/r", downloader=dl, cache_root=root,
                                    refresh_ttl_sec=0)
        old_gone = not (ev_dir / "old").exists()
        ok = res.reason == "ok" and res.n_images == 2 and dl.calls == 1 and ev_dir.is_dir() and old_gone
        print(f"[{'PASS' if ok else 'FAIL'}] nonempty_replaces_cache: "
              f"reason={res.reason} n_images={res.n_images} dl_calls={dl.calls} "
              f"events_valid={ev_dir.is_dir()} old_gone={old_gone}")
        return ok
    finally:
        if tmp_path is None:
            shutil.rmtree(root, ignore_errors=True)


def main():
    print("[INFO] consensus_gather self-test 시작")
    results = [
        test_stage_basic(),
        test_layout_nested(),
        test_replace_swaps_to_latest(),
        test_empty_preserves_existing(),
        test_downloader_raises(),
        test_skipped_no_recipe(),
        test_malformed_event_reports_error(),
        test_max_events_passthrough(),
        test_zero_image_event_preserves_old_cache(),
        test_ttl_skips_download(),
        test_nonempty_replaces_cache(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
