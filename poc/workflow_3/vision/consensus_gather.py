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
