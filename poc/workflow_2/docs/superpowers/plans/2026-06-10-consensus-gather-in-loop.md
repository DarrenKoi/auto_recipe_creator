# consensus S-image gather (workflow_3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** align fail 감지 시 그 recipe 의 최근 성공(S) 측정 이미지 + cond.txt 를 workflow_3 모니터 루프에서 비차단으로 stage 해 consensus 빌드 재료를 확보한다(빌드는 범위 밖, deferred).

**Architecture:** vision 에 *순수* orchestration(`gather_success_images`, disk layout 주인)을 두고, monitor glue(`success_gather`)가 office 다운로더 해석(2단 fallback) + daemon thread fire 로 감싼다. 루프(`align_fail_monitor.process_fail_rows`)는 호출 1줄. 실제 DB 조회·파일 쓰기는 office 다운로더(사용자, gitignore)가 담당. 실패는 모두 graceful degrade(루프 불사, 기존 캐시 보존).

**Tech Stack:** Python 3.10+, dataclass + typing.Protocol, `shutil`/`pathlib`(os.replace swap), threading(daemon), workflow_3 `Workflow3Settings`/`log_work2_event`. 테스트는 workflow_3 self-running 스크립트 규약([PASS]/[FAIL] print, argparse 미사용).

**Spec:** `poc/workflow_2/docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md`

**불변 제약(CLAUDE.md):** CLI 인자 금지 · Korean docstring · `[INFO]/[ERROR]/[WARNING]` print(+`log_work2_event`) · `from __future__` 금지 · 절대 임포트 · **workflow_3 는 legacy(wf1/wf2) import 금지** · commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**의존 순서(중요):** Task 1(상수) → Task 2(vision, 상수 import) → Task 3(config) → Task 4(monitor glue, config+vision import) → Task 5(루프) → Task 6(office, 사용자).

---

### Task 1: cache root 상수 추가 (`ALIGN_CONSENSUS_CACHE_DIR`)

**Files:**

- Modify: `poc/workflow_3/__init__.py` (ALIGN_IMAGES_DIR 블록 직후 + `__all__`)

`vision/consensus_gather.py`(Task 2)가 이 상수를 import 하므로 **먼저** 추가한다. `ALIGN_IMAGES_DIR` 와 같은 env-override 패턴.

- [ ] **Step 1: 상수 추가**

`poc/workflow_3/__init__.py` 의 `ALIGN_IMAGES_DIR = (...)` 블록(현재 64–69행) **바로 다음**에 삽입:

```python
# consensus S-image gather 캐시 루트. MES 산출물(align_images)이 아니라 우리가 만드는
# 파생 캐시라 위치 자유 — workflow_3 아래 둔다. env 로 override 가능.
_consensus_cache_env = os.environ.get("ALIGN_CONSENSUS_CACHE_DIR", "").strip()
ALIGN_CONSENSUS_CACHE_DIR = (
    Path(_consensus_cache_env)
    if _consensus_cache_env
    else WORKFLOW_3_DIR / "align_consensus_cache"
)
```

- [ ] **Step 2: `__all__` 에 등록**

`__all__` 리스트(현재 `"ALIGN_IMAGES_DIR",` 가 있는 곳)에 한 줄 추가:

```python
    "ALIGN_CONSENSUS_CACHE_DIR",
```

- [ ] **Step 3: import 검증**

Run: `uv run python -c "from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR; print(ALIGN_CONSENSUS_CACHE_DIR)"`
Expected: `...\poc\workflow_3\align_consensus_cache` 경로가 출력되고 에러 없음.

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/__init__.py
git commit -m "workflow_3(consensus-gather 1/5): ALIGN_CONSENSUS_CACHE_DIR 루트 상수

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: vision 순수 orchestration (`gather_success_images`)

**Files:**

- Create: `poc/workflow_3/vision/consensus_gather.py`
- Test: `poc/workflow_3/vision/test_consensus_gather.py`

orchestration = disk layout 주인(events_dir 결정 → 임시 dir 에 다운로더로 stage → ≥1 event 면 events/ 로 swap=replace-if-non-empty). office/threading import 0 → Mac 합성 다운로더로 완결 테스트.

- [ ] **Step 1: 실패 테스트 작성**

Create `poc/workflow_3/vision/test_consensus_gather.py`:

```python
"""consensus_gather 합성 self-test — office/실장비 없이 stage/replace/error 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/vision/test_consensus_gather.py
"""

import shutil
import tempfile
from pathlib import Path

from poc.workflow_3.vision.consensus_gather import (
    StagedEvent,
    gather_success_images,
)


class _FakeDownloader:
    """dest_dir 에 합성 S*.jpg + S*.txt 를 쓰고 StagedEvent 를 돌려주는 테스트용 다운로더.

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
                img = ev_dir / f"S{i + 1:04d}.jpg"
                cond = ev_dir / f"S{i + 1:04d}.txt"
                img.write_bytes(b"\xff\xd8\xff\xd9")
                cond.write_text("crosshair_x=10\ncrosshair_y=20\n", encoding="utf-8")
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
            and "EV1/S0001.jpg" in files
            and "EV1/S0001.txt" in files
            and "EV2/S0002.jpg" in files
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
        ok = res.events_dir == expected and (expected / "EV1" / "S0001.jpg").exists()
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


def main():
    print("[INFO] consensus_gather self-test 시작")
    results = [
        test_stage_basic(),
        test_layout_nested(),
        test_replace_swaps_to_latest(),
        test_empty_preserves_existing(),
        test_downloader_raises(),
        test_skipped_no_recipe(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run python poc/workflow_3/vision/test_consensus_gather.py`
Expected: `ModuleNotFoundError: No module named 'poc.workflow_3.vision.consensus_gather'` (또는 import 에러)로 실패.

- [ ] **Step 3: 최소 구현 작성**

Create `poc/workflow_3/vision/consensus_gather.py`:

```python
"""consensus S-image gather — 최근 성공(S) 측정 이미지를 stage 하는 순수 orchestration.

(설계: docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md)

이 모듈은 *disk layout 의 주인*이다: cache root 아래 events_dir 를 정하고, 임시 dir 에
다운로더로 stage 한 뒤 ≥1 event 면 events/ 로 교체(replace-if-non-empty)한다. DB 조회와
실제 파일 쓰기는 SuccessDownloader(office 구현)가 담당 — 이 모듈은 office/threading 을
import 하지 않아 Mac 합성 다운로더로 단위테스트된다. consensus *빌드*는 범위 밖(deferred).
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR

GATHER_MAX_EVENTS = 5


@dataclass
class StagedEvent:
    """다운로더가 한 measurement event 를 stage 한 결과(쓰여진 파일 경로)."""

    event_id: str
    image_paths: list[Path]   # 쓰여진 S*.jpg
    cond_paths: list[Path]    # 쓰여진 S*.txt(cond, crosshair 좌표)


@dataclass
class GatherResult:
    """gather_success_images 결과 + audit. (events 파일 경로는 events_dir 아래 그대로 존재)"""

    eqp_id: str
    recipe_id: str
    events_dir: Path
    n_events: int
    n_images: int
    reason: str               # "ok" | "empty" | "skipped" | "error:<msg>"


class SuccessDownloader(Protocol):
    """recipe 의 최근 성공(S) 측정 이미지를 dest_dir 에 쓰는 office 구현 계약."""

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir) -> list:
        """recipe_id('<class>/<recipe>')의 최근 성공 측정 max_events 건을 dest_dir/<event_id>/ 에
        S*.jpg + S*.txt(cond)로 쓰고 list[StagedEvent] 를 반환한다(align fail 측정 제외).

        dest_dir 는 호출부가 넘기는 *임시 staging dir* (최종 events/ 아님). 성공 측정이
        없으면 빈 리스트를 반환한다(호출부가 기존 캐시를 보존)."""
        ...


def _events_dir_for(eqp_id, recipe_id, cache_root):
    """이 recipe 의 최종 events/ 경로. recipe_id 가 '<class>/<recipe>' 라 3단 중첩."""
    return Path(cache_root) / eqp_id / recipe_id / "events"


def gather_success_images(eqp_id, recipe_id, *, downloader,
                          max_events=GATHER_MAX_EVENTS,
                          cache_root=ALIGN_CONSENSUS_CACHE_DIR) -> GatherResult:
    """최근 성공 S 이미지를 stage 한다(replace-if-non-empty). 예외는 삼켜 GatherResult 로 보고.

    절차: 임시 staging dir 에 downloader 로 받기 → ≥1 event 면 기존 events/ 제거 후 swap,
    0건/예외면 기존 events/ 보존. 어떤 경로든 staging 잔재는 정리한다.
    """
    events_dir = _events_dir_for(eqp_id, recipe_id, cache_root)
    if not recipe_id:
        return GatherResult(eqp_id, recipe_id, events_dir, 0, 0, "skipped")

    staging_dir = events_dir.parent / ".events_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        staged = downloader.download_recent_successes(
            recipe_id, max_events=max_events, dest_dir=staging_dir
        )
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return GatherResult(eqp_id, recipe_id, events_dir, 0, 0,
                            f"error:{type(exc).__name__}: {exc}")

    staged = staged or []
    n_events = len(staged)
    n_images = sum(len(ev.image_paths) for ev in staged)

    if n_events == 0:
        # 빈 fetch — 기존 events/ 보존(replace-if-non-empty).
        shutil.rmtree(staging_dir, ignore_errors=True)
        return GatherResult(eqp_id, recipe_id, events_dir, 0, 0, "empty")

    # swap: 기존 events/ 제거 후 staging → events (os.replace, 같은 볼륨 rename).
    if events_dir.exists():
        shutil.rmtree(events_dir, ignore_errors=True)
    events_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.replace(events_dir)

    return GatherResult(eqp_id, recipe_id, events_dir, n_events, n_images, "ok")


__all__ = [
    "GATHER_MAX_EVENTS",
    "GatherResult",
    "StagedEvent",
    "SuccessDownloader",
    "gather_success_images",
]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run python poc/workflow_3/vision/test_consensus_gather.py`
Expected: `[INFO] 6/6 cases passed`, exit 0, 모든 줄 `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/vision/consensus_gather.py poc/workflow_3/vision/test_consensus_gather.py
git commit -m "workflow_3(consensus-gather 2/5): vision 순수 orchestration (6/6)

gather_success_images = disk layout 주인(events_dir + 임시 staging → swap).
replace-if-non-empty: 빈/예외 fetch 시 기존 events/ 보존. SuccessDownloader Protocol.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: config 게이트 필드 (`gather_enabled` / `gather_max_events`)

**Files:**

- Modify: `poc/workflow_3/config.py` (Workflow3Settings 필드 + load_workflow3_settings)

- [ ] **Step 1: dataclass 필드 추가**

`poc/workflow_3/config.py` 의 `Workflow3Settings` 안, `# --- CV 보정 ---` 블록 **앞**(또는 `zoom_scroll_dy` 다음)에 새 블록 추가:

```python
    # --- consensus S-image gather ---
    gather_enabled: bool = True
    gather_max_events: int = 5
```

- [ ] **Step 2: env 배선 추가**

`load_workflow3_settings()` 의 `Workflow3Settings(...)` 생성 인자 목록에 두 줄 추가(예: `zoom_scroll_dy=...` 다음):

```python
        gather_enabled=env_flag("ALIGN_FAIL_GATHER_SUCCESS", default=True),
        gather_max_events=env_int("ALIGN_FAIL_GATHER_MAX_EVENTS", 5),
```

- [ ] **Step 3: 검증**

Run: `uv run python -c "from poc.workflow_3.config import load_workflow3_settings as L; s=L(); print(s.gather_enabled, s.gather_max_events)"`
Expected: `True 5` (에러 없음).

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/config.py
git commit -m "workflow_3(consensus-gather 3/5): Workflow3Settings gather 게이트

gather_enabled(ALIGN_FAIL_GATHER_SUCCESS, on) / gather_max_events(ALIGN_FAIL_GATHER_MAX_EVENTS, 5).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: monitor glue (`success_gather`) — office loader + 비차단 fire

**Files:**

- Create: `poc/workflow_3/monitor/success_gather.py`
- Test: `poc/workflow_3/monitor/test_success_gather.py`

vision 순수 함수를 office 다운로더(2단 fallback)와 daemon thread 로 감싼다. office 부재/예외 시 조용히 skip.

- [ ] **Step 1: 실패 테스트 작성**

Create `poc/workflow_3/monitor/test_success_gather.py`:

```python
"""success_gather 합성 self-test — gating + office 2단 loader 검증(스레드 join).

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/monitor/test_success_gather.py
"""

import dataclasses
import sys
import types
from pathlib import Path

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import success_gather
from poc.workflow_3.vision.consensus_gather import GatherResult


def _settings(**overrides):
    """기본 settings 에 일부 필드만 바꿔 만든다(frozen dataclass → replace)."""
    return dataclasses.replace(load_workflow3_settings(), **overrides)


def _install_recording_gather():
    """gather_success_images 를 호출 인자 기록용 fake 로 교체하고 calls 리스트를 돌려준다."""
    calls = []

    def _fake(eqp_id, recipe_id, *, downloader, max_events):
        calls.append({"eqp_id": eqp_id, "recipe_id": recipe_id,
                      "downloader": downloader, "max_events": max_events})
        return GatherResult(eqp_id, recipe_id, Path("x"), 1, 2, "ok")

    success_gather.gather_success_images = _fake
    return calls


def test_fires_when_enabled():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    sentinel = object()
    success_gather._DOWNLOADER = sentinel
    st = _settings(gather_enabled=True, gather_max_events=7)

    t = success_gather.gather_success_async("EQP1", "CLS/RCP", st)
    if t is not None:
        t.join(timeout=5)
    ok = (
        t is not None
        and len(calls) == 1
        and calls[0]["eqp_id"] == "EQP1"
        and calls[0]["recipe_id"] == "CLS/RCP"
        and calls[0]["downloader"] is sentinel
        and calls[0]["max_events"] == 7
    )
    print(f"[{'PASS' if ok else 'FAIL'}] fires_when_enabled: thread={t is not None} calls={len(calls)}")
    return ok


def test_skips_when_disabled():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    success_gather._DOWNLOADER = object()
    t = success_gather.gather_success_async("EQP1", "CLS/RCP", _settings(gather_enabled=False))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_when_disabled: thread={t} calls={len(calls)}")
    return ok


def test_skips_without_recipe():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = True
    success_gather._DOWNLOADER = object()
    t = success_gather.gather_success_async("EQP1", "", _settings(gather_enabled=True))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_recipe: thread={t} calls={len(calls)}")
    return ok


def test_skips_without_downloader():
    calls = _install_recording_gather()
    success_gather.DOWNLOADER_AVAILABLE = False
    success_gather._DOWNLOADER = None
    t = success_gather.gather_success_async("EQP1", "CLS/RCP", _settings(gather_enabled=True))
    ok = t is None and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] skips_without_downloader: thread={t} calls={len(calls)}")
    return ok


def test_loader_canonical():
    """정위치 모듈의 make_success_downloader 팩토리 결과를 돌려준다."""
    mod = types.ModuleType("poc.workflow_3.monitor.office_success_downloader")
    sentinel = object()
    mod.make_success_downloader = lambda: sentinel
    sys.modules["poc.workflow_3.monitor.office_success_downloader"] = mod
    try:
        got = success_gather._load_office_downloader()
        ok = got is sentinel
    finally:
        del sys.modules["poc.workflow_3.monitor.office_success_downloader"]
    print(f"[{'PASS' if ok else 'FAIL'}] loader_canonical: got_is_sentinel={ok}")
    return ok


def main():
    print("[INFO] success_gather self-test 시작")
    results = [
        test_fires_when_enabled(),
        test_skips_when_disabled(),
        test_skips_without_recipe(),
        test_skips_without_downloader(),
        test_loader_canonical(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run python poc/workflow_3/monitor/test_success_gather.py`
Expected: `ImportError`/`ModuleNotFoundError` (success_gather 미존재)로 실패.

- [ ] **Step 3: 최소 구현 작성**

Create `poc/workflow_3/monitor/success_gather.py`:

```python
"""consensus gather 의 office 접점 + 비차단 fire (monitor glue).

vision.consensus_gather 의 순수 orchestration 을 office 다운로더 해석(2단 fallback)과
daemon thread 로 감싼다. office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를
죽이지 않는다(alarm_source/notify 와 동일 철학).

(설계: docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md §3-A)
"""

import importlib
import threading

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.vision.consensus_gather import gather_success_images

LOG_COMPONENT = "consensus_gather"


def _load_office_downloader():
    """SuccessDownloader 구현을 정위치 → legacy 순서로 찾는다. 없으면 None.

    office_* 모듈은 gitignore 라 오피스 PC 에만 존재한다. 모듈은 인자 없는
    `make_success_downloader()` 팩토리를 노출해야 한다.
    """
    for module_path, is_legacy in (
        ("poc.workflow_3.monitor.office_success_downloader", False),
        ("poc.workflow_1.office_success_downloader", True),
    ):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        factory = getattr(module, "make_success_downloader", None)
        if factory is None:
            continue
        if is_legacy:
            print("[WARNING] office_success_downloader 가 legacy 위치(workflow_1)에서 "
                  "로드됨 — poc/workflow_3/monitor/ 로 복사하세요.")
        try:
            return factory()
        except Exception as exc:
            print(f"[WARNING] success downloader 생성 실패: {exc}")
            return None
    return None


_DOWNLOADER = _load_office_downloader()
DOWNLOADER_AVAILABLE = _DOWNLOADER is not None
if not DOWNLOADER_AVAILABLE:
    print("[INFO] success downloader 없음 — consensus gather 비활성(개발 PC/미구현).")


def gather_success_async(eqp_id, recipe_id, settings: Workflow3Settings):
    """recipe 의 최근 성공 S 이미지 stage 를 daemon thread 로 비차단 실행한다.

    gather_enabled off / recipe_id 없음 / downloader 부재면 아무것도 안 하고 None.
    실제 fire 하면 시작된 Thread 를 반환한다(테스트 join 용). 예외는 thread 안에서 삼킨다.
    """
    if not settings.gather_enabled or not recipe_id or not DOWNLOADER_AVAILABLE:
        return None

    def _run():
        try:
            result = gather_success_images(
                eqp_id, recipe_id,
                downloader=_DOWNLOADER,
                max_events=settings.gather_max_events,
            )
            print(f"[INFO] consensus gather: EQP_ID={eqp_id} recipe={recipe_id} "
                  f"reason={result.reason} events={result.n_events} images={result.n_images}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_done",
                eqp_id=eqp_id, recipe_id=recipe_id, reason=result.reason,
                n_events=result.n_events, n_images=result.n_images,
            )
        except Exception as exc:
            print(f"[WARNING] consensus gather 예외: EQP_ID={eqp_id}, error={exc}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_error", level="warning",
                eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


__all__ = ["DOWNLOADER_AVAILABLE", "gather_success_async"]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run python poc/workflow_3/monitor/test_success_gather.py`
Expected: `[INFO] 5/5 cases passed`, exit 0. (import 시 `[INFO] success downloader 없음...` 한 줄은 정상.)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/success_gather.py poc/workflow_3/monitor/test_success_gather.py
git commit -m "workflow_3(consensus-gather 4/5): monitor glue — office loader + 비차단 fire (5/5)

_load_office_downloader 2단 fallback(monitor→workflow_1), gather_success_async
daemon thread. gather_enabled/recipe_id/downloader 게이트, 예외 삼킴.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 루프 통합 — `process_fail_rows` 호출 1줄

**Files:**

- Modify: `poc/workflow_3/monitor/align_fail_monitor.py` (import + process_fail_rows 호출)

- [ ] **Step 1: import 추가**

`align_fail_monitor.py` 의 monitor import 블록(현재 `from poc.workflow_3.monitor.notify import (...)` 다음, 41행 근처)에 추가:

```python
from poc.workflow_3.monitor.success_gather import gather_success_async
```

- [ ] **Step 2: 호출 추가**

`process_fail_rows` 의 popup 블록(현재 316–323행, `if settings.popup_enabled: notify_align_fail_popup(...)`) **직후**, `# 알람별 사이클` 주석 **앞**에 삽입:

```python
        # consensus 재료 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
        # 게이트(gather_enabled/recipe_id/downloader)는 gather_success_async 내부에서 판정.
        gather_success_async(eqp_id, info["recipe_id"], settings)
```

- [ ] **Step 3: import/스모크 검증**

Run: `uv run python -c "import poc.workflow_3.monitor.align_fail_monitor as m; print('import ok', hasattr(m, 'process_fail_rows'))"`
Expected: `import ok True` (import 단계에서 `[INFO] success downloader 없음...` 한 줄 정상).

- [ ] **Step 4: 회귀 — 기존 self-test 통과 확인**

Run: `uv run python poc/workflow_3/vision/test_consensus_gather.py` → `6/6`
Run: `uv run python poc/workflow_3/monitor/test_success_gather.py` → `5/5`

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/align_fail_monitor.py
git commit -m "workflow_3(consensus-gather 5/5): 루프 통합 — process_fail_rows 비차단 호출

popup 직후 gather_success_async 1줄. 게이트는 함수 내부, 루프는 항상 호출(비차단).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: office downloader 스켈레톤 (사용자 구현, office-only)

**Files:**

- Create(사용자, office PC): `poc/workflow_3/monitor/office_success_downloader.py` (`**/office_*` → **gitignore, 커밋 안 됨**)

> 이 파일은 사용자가 측정이력 DB 에 붙여 채운다. Task 1–5(Mac 완결) 후, `SuccessDownloader`
> 계약이 코드로 확정된 상태에서 아래 스켈레톤을 드롭인한다. 계약(`StagedEvent` / `dest_dir` /
> 반환규약)은 Task 2 에서 import 가능.

- [ ] **Step 1: 스켈레톤 배치**

Create `poc/workflow_3/monitor/office_success_downloader.py`:

```python
"""office success downloader — recipe 최근 성공(S) 이미지 + cond 다운로더 (office-only).

SuccessDownloader Protocol(poc/workflow_3/vision/consensus_gather.py) 구현체.
**office PC 전용**: 사용자 측정이력 DB 에서 align fail 이 아니었던 최근 측정의 S 이미지와
cond.txt 를 dest_dir 에 쓴다. `**/office_*` gitignore 라 커밋되지 않는다.

monitor/success_gather.py 가 인자 없는 `make_success_downloader()` 팩토리로 인스턴스를 얻는다.
"""

from pathlib import Path

from poc.workflow_3.vision.consensus_gather import StagedEvent


class OfficeSuccessDownloader:
    """측정이력 DB → 최근 성공 S 이미지/cond 다운로드."""

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
        """recipe_id('<class>/<recipe>')의 최근 성공 측정 max_events 건을 stage.

        각 event 를 dest_dir/<event_id>/ 에 S0001.jpg, S0001.txt, ... 로 쓰고
        list[StagedEvent] 를 반환한다. align fail 이었던 측정은 제외(S-only).

        반환 규약(중요):
          - modality(OM/SEM)가 cond/파일명으로 구분 가능해야 한다(build 가 modality 별 묶음).
          - cond.txt 는 crosshair 좌표를 포함해야 한다(정렬·clean 필수).
          - 성공 측정이 없으면 빈 리스트(호출부가 기존 캐시 보존).
        """
        dest_dir = Path(dest_dir)
        staged: list[StagedEvent] = []

        # TODO(office): 측정이력 DB 조회 + 다운로드.
        #   events = db.query_recent_successes(recipe_id, limit=max_events)  # align fail 제외
        #   for ev in events:
        #       ev_dir = dest_dir / ev.event_id
        #       ev_dir.mkdir(parents=True, exist_ok=True)
        #       image_paths, cond_paths = [], []
        #       for i, item in enumerate(ev.success_images, start=1):
        #           img_path = ev_dir / f"S{i:04d}.jpg"
        #           cond_path = ev_dir / f"S{i:04d}.txt"
        #           <download item.image → img_path>
        #           <write item.crosshair_cond → cond_path>
        #           image_paths.append(img_path); cond_paths.append(cond_path)
        #       staged.append(StagedEvent(event_id=ev.event_id,
        #                                 image_paths=image_paths, cond_paths=cond_paths))

        return staged


def make_success_downloader():
    """success_gather._load_office_downloader() 가 호출하는 팩토리(인자 없음)."""
    return OfficeSuccessDownloader()
```

- [ ] **Step 2: office 1회 실행 검증**

office PC 에서 실제 fail 알람(또는 replay)으로 루프를 돌려 `align_consensus_cache/<eqp>/<class>/<recipe>/events/<event_id>/` 에 S*.jpg+S*.txt 가 stage 되는지 확인. 로그에 `consensus gather: ... reason=ok events=N images=M`.

- [ ] **Step 3: (커밋 없음)** `**/office_*` 는 gitignore — 이 파일은 추적되지 않는다.

---

## Self-Review

**Spec coverage:**

- §3-A 컴포넌트 6개 → Task 1(__init__) · Task 2(vision) · Task 3(config) · Task 4(monitor glue) · Task 5(루프) · Task 6(office). ✅
- §3-B 인터페이스(StagedEvent/GatherResult/SuccessDownloader/gather_success_images) → Task 2 그대로. ✅
- §3-C cache layout(`<eqp>/<class>/<recipe>/events/<event_id>/`) → Task 2 `_events_dir_for` + test_layout_nested. ✅
- §4 파라미터(gather_max_events 5, gather_enabled, replace-if-non-empty) → Task 3 + Task 2(swap/empty). ✅ (`min_s` 는 build 게이트 = deferred, 범위 밖 명시.)
- §5 error handling 1–7 → 루프 불사(Task 4 thread try/except) · office 미존재(Task 4 loader None) · off 스위치(Task 3+4) · recipe_id 게이트(Task 2 skipped + Task 4) · 빈/실패 보존(Task 2 empty/error) · 동시성(경로 분리, 코드상 자명). ✅
- §6 테스트 → vision 6 + monitor 5. ✅

**Placeholder scan:** office 스켈레톤의 `# TODO(office)` 는 의도된 사용자 구현 지점(Task 6, 비커밋). 그 외 모든 step 에 실제 코드/명령/기대출력 포함. ✅

**Type consistency:** `StagedEvent(event_id, image_paths, cond_paths)` · `GatherResult(eqp_id, recipe_id, events_dir, n_events, n_images, reason)` · `gather_success_images(eqp_id, recipe_id, *, downloader, max_events, cache_root)` · `download_recent_successes(recipe_id, *, max_events, dest_dir)` · `gather_success_async(eqp_id, recipe_id, settings)->Thread|None` · `_load_office_downloader()` · `make_success_downloader()` · settings.`gather_enabled`/`gather_max_events` · `ALIGN_CONSENSUS_CACHE_DIR` — 전 Task 일관. ✅
