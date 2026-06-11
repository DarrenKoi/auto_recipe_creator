"""consensus S-image gather — 최근 성공(S) 측정 이미지를 stage 하는 순수 orchestration.

(설계: poc/workflow_2/docs/superpowers/specs/2026-06-10-consensus-gather-in-loop-design.md)

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
    image_paths: list[Path]   # 쓰여진 S*.jpeg
    cond_paths: list[Path]    # 쓰여진 .<이미지명>/cond.txt (crosshair 좌표 등)


@dataclass
class GatherResult:
    """gather_success_images 결과 + audit. (events 파일 경로는 events_dir 아래 그대로 존재)"""

    eqp_id: str
    recipe_id: str
    events_dir: Path
    n_events: int
    n_images: int
    reason: str               # "ok" | "empty" | "skipped" | "error:<msg>" | "error:swap:<msg>"


class SuccessDownloader(Protocol):
    """recipe 의 최근 성공(S) 측정 이미지를 dest_dir 에 쓰는 office 구현 계약."""

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir) -> list:
        """recipe_id('<class>/<recipe>')의 최근 성공 측정 max_events 건을 dest_dir/<event_id>/ 에
        S*.jpeg + cond 로 쓰고 list[StagedEvent] 를 반환한다(align fail 측정 제외).

        dest_dir 는 호출부가 넘기는 *임시 staging dir* (최종 events/ 아님). 성공 측정이
        없으면 빈 리스트를 반환한다(호출부가 기존 캐시를 보존).

        event_id 는 msr 측정이력의 유니크 string 을 그대로 쓴다(결정 2026-06-10):
        `yyyymmdd_hhmmss_<recipe_name>_<lot_id>` 형태 — 시각 prefix 라 이름 정렬 = 시간
        정렬이고, 같은 측정 재수집 시 같은 id 가 나온다. Windows 디렉토리명 금지 문자
        (콜론/슬래시/공백)가 들어오면 office 구현이 치환해야 한다.

        --- cond 파일 계약 (deferred consensus build 이 추가 변환 없이 소비하는 조건) ---

        - 레이아웃은 align_images/ 와 동일한 숨김폴더 규약(결정 2026-06-10, office MES
          원형 유지): 이미지는 dest_dir/<event_id>/S*.jpeg, cond 는
          dest_dir/<event_id>/.<이미지파일명>/cond.txt — `cond_file.load_cond(이미지경로)`
          가 그대로 읽는 위치다.
        - cond 내용은 `vision/cond_file.py` 의 `parse_cond()` 로 파싱 가능한 형식이어야
          한다(최소 `!Cursor_info`(crosshair 좌표) 포함).
        - modality(OM/SEM)는 구분 가능해야 한다(build 가 modality 별로 묶는다). msr 원문
          cond 에는 `Scope` 가 없으므로(2026-06-08 확인) `cond_file.msr_modality()` 가
          추론에 쓰는 키(`!OM_Brightness`/`Accelerating_voltage`/`Magnification`)를 보존할 것."""
        ...


def _events_dir_for(eqp_id, recipe_id, cache_root):
    """이 recipe 의 최종 events/ 경로. recipe_id 가 '<class>/<recipe>' 라 3단 중첩."""
    return Path(cache_root) / eqp_id / recipe_id / "events"


# S 이미지로 인정하는 확장자 (gather 가 쓰는 형식 + 안전 마진).
_S_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def count_staged_events(eqp_id, recipe_id, *,
                        cache_root=ALIGN_CONSENSUS_CACHE_DIR) -> tuple:
    """이미 stage 된 consensus event 수와 S 이미지 수를 센다(읽기 전용).

    feasibility 마킹/manifest 의 "consensus 자료 얼마나 있나" 컨텍스트용. gather 와
    같은 events/ 레이아웃을 쓰되 아무것도 쓰지 않는다. 없으면 (0, 0). `(n_events,
    n_images)` 반환. recipe_id 비거나 경로 부재/예외면 (0, 0).
    """
    if not recipe_id:
        return 0, 0
    events_dir = _events_dir_for(eqp_id, recipe_id, cache_root)
    if not events_dir.is_dir():
        return 0, 0
    n_events = 0
    n_images = 0
    try:
        for ev in sorted(events_dir.iterdir()):
            if not ev.is_dir():
                continue
            imgs = [
                p for p in ev.glob("S*")
                if p.is_file() and p.suffix.lower() in _S_IMAGE_EXTS
            ]
            if imgs:
                n_events += 1
                n_images += len(imgs)
    except OSError:
        return n_events, n_images
    return n_events, n_images


def gather_success_images(eqp_id, recipe_id, *, downloader,
                          max_events=GATHER_MAX_EVENTS,
                          cache_root=ALIGN_CONSENSUS_CACHE_DIR) -> GatherResult:
    """최근 성공 S 이미지를 stage 한다(replace-if-non-empty). 예외는 삼켜 GatherResult 로 보고.

    절차: 임시 staging dir 에 downloader 로 받기 → ≥1 event 면 기존 events/ 제거 후 swap,
    0건/예외면 기존 events/ 보존. 어떤 경로든 staging 잔재는 정리한다.
    다운로드 예외 → reason="error:<Type>: <msg>", swap/count 예외 → reason="error:swap:<Type>: <msg>".
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

    try:
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
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return GatherResult(eqp_id, recipe_id, events_dir, 0, 0,
                            f"error:swap:{type(exc).__name__}: {exc}")

    return GatherResult(eqp_id, recipe_id, events_dir, n_events, n_images, "ok")


__all__ = [
    "GATHER_MAX_EVENTS",
    "GatherResult",
    "StagedEvent",
    "SuccessDownloader",
    "count_staged_events",
    "gather_success_images",
]
