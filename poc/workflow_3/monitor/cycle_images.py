"""테이크 1건이 처리한 이미지를 한 폴더로 모은다.

알람 사이클이 도는 동안 이미지는 **9곳**에 흩어져 쌓인다. 그중 4곳만 tag 로 키가
잡히고, 나머지 5곳은 모델 slug 나 자기 타임스탬프로 키가 잡혀 "어느 테이크의
것인가" 를 이름만으로는 알 수 없다. 그래서 수집 규칙이 둘이다.

  * ``keying="tag"``   - tag 폴더를 정확히 가리킨다. 오수집이 원리상 없다.
  * ``keying="mtime"`` - 폴더 전체를 훑되 mtime 이 테이크 구간 안인 파일만 고른다.

mtime 창이 성립하는 근거는 ``align_fail_monitor`` 가 단일 프로세스이고 RCS 커서를
직렬화한다는 것 하나뿐이다. 동시 실행이 생기면 이 규칙은 곧바로 깨진다.

호출은 ``run_alarm_cycle`` 의 **finally 한 곳**에서만 한다. 녹화 스레드나 engineer
watch 에 훅을 걸면 안 된다 - 둘 다 테이크마다 존재하지 않기 때문이다: 보정이
성공하면 watch 가 아예 돌지 않고(cycle.py 가 status != "corrected" 로 게이트한다),
접속 단계에서 실패하면 녹화가 시작되기 전이라 세션 자체가 없다. 그 두 경우가
수집에서 통째로 빠지는 것이 이 위치를 정한 이유다.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

# 수집 결과를 넣는 폴더 이름. tag 키 소스(align_fail_cycle/<tag>) *안에* 살기
# 때문에, 수집할 때 반드시 자기 자신을 제외해야 한다 - 안 그러면 재실행마다
# 직전 수집본을 다시 복사해 눈덩이가 된다.
GATHER_DIR_NAME = "gathered"


@dataclass(frozen=True)
class GatheredImage:
    """테이크에 귀속된 이미지 1장."""

    source_path: Path
    stage: str
    label: str
    keying: str


@dataclass(frozen=True)
class ImageSource:
    """이미지가 쌓이는 한 곳 + 그것을 테이크에 귀속시키는 규칙."""

    directory: Path
    stage: str  # 사이클의 시간 순서 슬롯.
    keying: str  # "tag" | "mtime"
    label: str  # 복사본 파일명 접두어.


def take_image_sources(
    tag: str,
    eqp_id: str,
    *,
    debug_root: Path,
    model_slug: str,
) -> list[ImageSource]:
    """이 테이크의 이미지가 쌓였을 수 있는 9곳을 반환한다(존재 여부는 안 본다)."""
    root = Path(debug_root)
    return [
        ImageSource(root / model_slug, "00_connect", "mtime", "login"),
        ImageSource(root / "view_list_tab_rcs", "00_connect", "mtime", "listtab"),
        ImageSource(root / "workflow_select_tool", "00_connect", "mtime", "toolrow"),
        ImageSource(root / "row_occupant", "01_gates", "mtime", "occupant"),
        ImageSource(root / "share_request" / tag, "01_gates", "tag", "share"),
        ImageSource(root / "access_request" / tag, "01_gates", "tag", "access"),
        ImageSource(root / "align_fail_cycle" / tag, "03_correction", "tag", "correction"),
        ImageSource(root / "engineer_done" / f"{eqp_id}_{tag}", "04_engineer", "tag", "done"),
        ImageSource(root / "assist_score", "04_engineer", "mtime", "assist"),
    ]


def collect_take_images(
    sources,
    *,
    started_epoch: float,
    now: float,
) -> list[GatheredImage]:
    """소스 목록에서 이 테이크에 귀속되는 이미지를 고른다.

    tag 소스는 mtime 을 보지 않는다 - 폴더 자체가 이미 테이크 전용이라 시간으로
    한 번 더 거르면 시계 오차/느린 쓰기에 정상 산출물을 잃는다.
    """
    found: list[GatheredImage] = []
    for source in sources:
        if not source.directory.is_dir():
            continue
        for path in sorted(source.directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if GATHER_DIR_NAME in path.relative_to(source.directory).parts:
                continue
            if source.keying == "mtime":
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if not (started_epoch <= mtime <= now):
                    continue
            found.append(
                GatheredImage(
                    source_path=path,
                    stage=source.stage,
                    label=source.label,
                    keying=source.keying,
                )
            )
    return found


@dataclass
class GatherReport:
    """수집 결과 요약 - 콘솔 한 줄과 manifest 의 근거."""

    dest: Path
    copied: int = 0
    failed: int = 0
    total_bytes: int = 0
    recording_dir: str = ""
    recording_frames: int = 0
    already: bool = False
    images: list = field(default_factory=list)


def _dest_name(image: GatheredImage, taken: set) -> str:
    """`<stage>_<label>_<원래이름>`. 이름이 겹치면 뒤에 번호를 붙인다."""
    base = f"{image.stage}_{image.label}_{image.source_path.name}"
    if base not in taken:
        taken.add(base)
        return base
    stem, suffix = Path(base).stem, Path(base).suffix
    for n in range(2, 1000):
        candidate = f"{stem}__{n}{suffix}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    return base


def _resolve_dest(tag_dir: Path, started_epoch: float) -> tuple[Path, bool]:
    """이 테이크가 쓸 폴더와 "이미 모았는가" 를 파일시스템에서 판정한다.

    상태를 모니터가 아니라 **디스크에서 유도한다**. 프로세스가 재시작해도, 두
    진입점(align / check-only) 중 어디서 와도 같은 규칙이 성립해야 한다.

    tag 는 알람 UTC9 에서 나와 **테이크당 유일하지 않다** (cooldown 재시도가 같은
    tag 로 돌아온다). 그래서 manifest 의 ``started_epoch`` 를 테이크 식별자로 쓴다 -
    같으면 이미 모은 테이크이고, 다르면 재시도라 별도 폴더(``__a2``)를 준다.
    """
    base = tag_dir / GATHER_DIR_NAME
    for index in range(1, 100):
        candidate = base if index == 1 else Path(f"{base}__a{index}")
        manifest = candidate / "gathered_manifest.json"
        if not manifest.exists():
            return candidate, False
        try:
            recorded = float(json.loads(manifest.read_text(encoding="utf-8"))["started_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            return candidate, True  # 읽을 수 없으면 덮어쓰지 않는다.
        if abs(recorded - float(started_epoch)) < 0.5:
            return candidate, True
    return base, True


def gather_cycle_images(
    result,
    context,
    *,
    started_epoch: float,
    now: float | None = None,
    debug_root: Path | None = None,
    model_slug: str | None = None,
) -> GatherReport:
    """이 테이크가 처리한 파생 이미지를 한 폴더로 모으고 manifest 를 쓴다.

    녹화 프레임은 **복사하지 않는다** - 한 테이크가 수백~수천 장이라 복사하면
    알람마다 수백 MB 가 된다. manifest 에 경로와 장수만 남긴다.
    """
    context = context or {}
    if debug_root is None:
        from poc.workflow_3 import DEBUG_IMAGE_DIR

        debug_root = DEBUG_IMAGE_DIR
    if model_slug is None:
        from poc.workflow_3 import resolve_debug_model_name

        model_slug = resolve_debug_model_name()
    now = time.time() if now is None else now

    tag = str(getattr(result, "tag", "") or "")
    eqp_id = str(getattr(result, "eqp_id", "") or "")
    # 녹화 정보는 result 우선, 없으면 살아 있는 세션에서 읽는다. teardown 의
    # recording_stop 단계가 실패하면(run_teardown 은 나머지를 계속 돌린다) result 의
    # 녹화 필드는 빈 채로 남지만, 세션 자신은 out_dir/frames 를 알고 있다.
    rec_dir = str(getattr(result, "recording_dir", "") or "")
    rec_frames = int(getattr(result, "frame_count", 0) or 0)
    session = context.get("recording")
    if session is not None:
        rec_dir = rec_dir or str(getattr(session, "out_dir", "") or "")
        rec_frames = rec_frames or len(getattr(session, "frames", ()) or ())

    report = GatherReport(
        dest=Path(debug_root) / "align_fail_cycle" / tag / GATHER_DIR_NAME,
        recording_dir=rec_dir,
        recording_frames=rec_frames,
    )
    if not tag:
        # tag 없이 진행하면 dest 가 `align_fail_cycle//gathered` 로 접혀 서로 다른
        # 테이크가 한 폴더에 쌓인다. 모으지 않는 편이 낫다.
        report.already = True
        return report

    dest, already = _resolve_dest(Path(debug_root) / "align_fail_cycle" / tag, started_epoch)
    report.dest = dest
    report.already = already
    if already:
        return report

    sources = take_image_sources(tag, eqp_id, debug_root=debug_root, model_slug=model_slug)
    collected = collect_take_images(sources, started_epoch=started_epoch, now=now)

    dest.mkdir(parents=True, exist_ok=True)
    taken: set = set()
    for image in collected:
        name = _dest_name(image, taken)
        try:
            shutil.copy2(image.source_path, dest / name)
            report.total_bytes += (dest / name).stat().st_size
            report.copied += 1
            report.images.append(
                {
                    "stage": image.stage,
                    "label": image.label,
                    "keying": image.keying,
                    "name": name,
                    "source": str(image.source_path),
                }
            )
        except OSError as exc:
            report.failed += 1
            print(f"[WARNING] 이미지 수집 실패(건너뜀): {image.source_path} ({exc})")

    manifest = {
        "tag": tag,
        "eqp_id": eqp_id,
        "recipe_id": str(getattr(result, "recipe_id", "") or ""),
        "gathered_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "started_epoch": round(float(started_epoch), 3),
        "recording": {
            "dir": report.recording_dir,
            "frame_count": report.recording_frames,
            # 녹화 프레임은 참조만 한다(장수가 많아 복사하면 테이크당 수백 MB).
            "copied": False,
        },
        "counts": {"copied": report.copied, "failed": report.failed,
                   "total_bytes": report.total_bytes},
        "images": report.images,
    }
    (dest / "gathered_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def gather_and_report(result, context, *, started_epoch: float, **kwargs) -> bool:
    """수집하고 콘솔에 한 줄 요약을 찍는다. 예외는 절대 사이클로 올려보내지 않는다.

    호출부는 `run_alarm_cycle` 의 finally 한 곳뿐이다. teardown 뒤이므로 이미 끝난
    테이크의 결과를 여기서 던져 날리면 안 된다. 반환값은 "이번 호출이 실제로
    모았는가".
    """
    try:
        report = gather_cycle_images(result, context, started_epoch=started_epoch, **kwargs)
    except Exception as exc:
        print(f"[WARNING] 테이크 이미지 수집 실패(사이클 결과에는 영향 없음): {exc}")
        return False
    if report.already:
        return False
    failed = f", 실패 {report.failed}" if report.failed else ""
    print(
        f"[INFO] 테이크 이미지 수집: {report.copied}장"
        f" ({report.total_bytes / 1024.0:.0f}KB{failed})"
        f" + 녹화 {report.recording_frames} frames(참조)"
        f" -> {report.dest}"
    )
    return True


__all__ = [
    "gather_and_report",
    "GatherReport",
    "gather_cycle_images",
    "GatheredImage",
    "ImageSource",
    "collect_take_images",
    "take_image_sources",
    "IMAGE_SUFFIXES",
    "GATHER_DIR_NAME",
]
