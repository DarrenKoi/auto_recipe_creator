"""cycle_images 단위 테스트 — RCS/VLM 없이 Mac 에서 돈다.

파일시스템은 mock 하지 않고 tmp_path 에 진짜 트리를 만든다. 이 모듈이 판정하는
것이 곧 "디스크에 무엇이 있고 언제 쓰였나" 라서, mock 을 끼우면 검증 대상이
사라진다.
"""

from pathlib import Path

import pytest

from poc.workflow_3.monitor import cycle_images as ci


TAG = "260820_113045"
EQP = "MCD916"


def _sources(tmp_path: Path):
    return ci.take_image_sources(
        TAG, EQP, debug_root=tmp_path / "debug_images", model_slug="mai-ui-8b"
    )


def test_tag_keyed_sources_point_at_exact_tag_directories(tmp_path):
    """tag 로 키가 잡히는 4곳은 tag 폴더를 정확히 가리킨다(추측/스캔 없이)."""
    root = tmp_path / "debug_images"
    by_dir = {s.directory: s for s in _sources(tmp_path) if s.keying == "tag"}

    assert by_dir.keys() == {
        root / "align_fail_cycle" / TAG,
        root / "share_request" / TAG,
        root / "access_request" / TAG,
        root / "engineer_done" / f"{EQP}_{TAG}",
    }
    assert by_dir[root / "align_fail_cycle" / TAG].stage == "03_correction"
    assert by_dir[root / "share_request" / TAG].stage == "01_gates"
    assert by_dir[root / "access_request" / TAG].stage == "01_gates"
    assert by_dir[root / "engineer_done" / f"{EQP}_{TAG}"].stage == "04_engineer"


def _touch(path: Path, mtime: float, data: bytes = b"x") -> Path:
    """지정한 mtime 을 가진 파일을 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    import os

    os.utime(path, (mtime, mtime))
    return path


def test_mtime_source_excludes_images_written_before_the_take_started(tmp_path):
    """모델 slug 폴더는 여러 테이크가 공유한다 - 이전 테이크 것을 가져오면 안 된다."""
    root = tmp_path / "debug_images"
    stale = _touch(root / "mai-ui-8b" / "260820_090000_login.jpg", mtime=1000.0)
    fresh = _touch(root / "mai-ui-8b" / "260820_113045_login.jpg", mtime=2500.0)

    found = ci.collect_take_images(_sources(tmp_path), started_epoch=2000.0, now=3000.0)

    paths = {g.source_path for g in found}
    assert fresh in paths
    assert stale not in paths


def test_previously_gathered_bundle_is_not_collected_again(tmp_path):
    """수집 결과는 tag 소스 *안에* 산다 - 제외하지 않으면 재실행마다 눈덩이가 된다."""
    root = tmp_path / "debug_images"
    real = _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", mtime=2500.0)
    _touch(root / "align_fail_cycle" / TAG / ci.GATHER_DIR_NAME / "03_correction_x.jpg", 2600.0)

    found = ci.collect_take_images(_sources(tmp_path), started_epoch=2000.0, now=3000.0)

    assert [g.source_path for g in found] == [real]


class _Result:
    def __init__(self, recording_dir="", frame_count=0):
        self.tag = TAG
        self.eqp_id = EQP
        self.recipe_id = "CLASS-A/RDL"
        self.recording_dir = recording_dir
        self.frame_count = frame_count


def test_gather_copies_derived_images_and_writes_a_manifest(tmp_path):
    """파생 이미지는 복사하고, 녹화 폴더는 경로로만 참조한다."""
    root = tmp_path / "debug_images"
    _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", 2500.0, b"aaa")
    _touch(root / "mai-ui-8b" / "260820_113045_login.jpg", 2400.0, b"bb")
    rec = tmp_path / "recording"
    _touch(rec / f"{TAG}_rcs_0000_00000000ms.jpg", 2450.0)

    report = ci.gather_cycle_images(
        _Result(recording_dir=str(rec), frame_count=1284),
        {},
        started_epoch=2000.0,
        now=3000.0,
        debug_root=root,
        model_slug="mai-ui-8b",
    )

    dest = root / "align_fail_cycle" / TAG / ci.GATHER_DIR_NAME
    assert report.copied == 2
    assert report.total_bytes == 5
    names = sorted(p.name for p in dest.glob("*.jpg"))
    assert names == ["00_connect_login_260820_113045_login.jpg",
                     "03_correction_correction_paused_match.jpg"]

    import json

    manifest = json.loads((dest / "gathered_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tag"] == TAG
    assert manifest["recording"]["dir"] == str(rec)
    assert manifest["recording"]["frame_count"] == 1284
    assert manifest["recording"]["copied"] is False
    assert len(manifest["images"]) == 2


def _gather(tmp_path, started_epoch, now=3000.0):
    return ci.gather_cycle_images(
        _Result(), {}, started_epoch=started_epoch, now=now,
        debug_root=tmp_path / "debug_images", model_slug="mai-ui-8b",
    )


def test_gathering_the_same_take_twice_does_not_duplicate(tmp_path):
    """같은 테이크를 두 번 모으면 안 된다.

    이 판정이 `__a2` 규칙의 전제다 - "이미 모은 테이크" 와 "같은 tag 로 돌아온
    재시도" 를 started_epoch 로 가르지 못하면, 재시도마다 폴더가 늘거나 앞
    테이크를 덮거나 둘 중 하나가 된다.
    """
    root = tmp_path / "debug_images"
    _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", 2500.0, b"aaa")

    first = _gather(tmp_path, started_epoch=2000.0)
    second = _gather(tmp_path, started_epoch=2000.0)

    assert first.copied == 1 and first.already is False
    assert second.copied == 0 and second.already is True
    assert second.dest == first.dest
    assert len(list(first.dest.glob("*.jpg"))) == 1


def test_cooldown_retry_with_the_same_tag_gets_its_own_bundle(tmp_path):
    """tag 는 알람 키라 재시도가 같은 tag 로 돌아온다 - 앞 테이크를 덮으면 안 된다."""
    root = tmp_path / "debug_images"
    _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", 2500.0, b"aaa")
    first = _gather(tmp_path, started_epoch=2000.0)

    _touch(root / "align_fail_cycle" / TAG / "paused_match2.jpg", 5500.0, b"bbbb")
    retry = _gather(tmp_path, started_epoch=5000.0, now=6000.0)

    assert retry.already is False
    assert retry.dest != first.dest
    assert retry.dest.name == f"{ci.GATHER_DIR_NAME}__a2"
    assert (first.dest / "03_correction_correction_paused_match.jpg").exists()


class _ExplodingResult:
    @property
    def tag(self):
        raise RuntimeError("boom")


def test_gather_never_raises_into_the_cycle(tmp_path, capsys):
    """teardown 근처에서 도는 코드다 - 여기서 뜬 예외는 끝난 테이크를 통째로 날린다."""
    ok = ci.gather_and_report(
        _ExplodingResult(), {}, started_epoch=2000.0,
        debug_root=tmp_path / "debug_images", model_slug="mai-ui-8b",
    )

    assert ok is False
    assert "WARNING" in capsys.readouterr().out


def test_gather_and_report_prints_a_one_line_summary(tmp_path, capsys):
    """오피스에서 실시간으로 보는 것은 콘솔뿐이다."""
    root = tmp_path / "debug_images"
    _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", 2500.0, b"aaa")

    ok = ci.gather_and_report(
        _Result(recording_dir="/x/rec", frame_count=1284), {},
        started_epoch=2000.0, now=3000.0,
        debug_root=root, model_slug="mai-ui-8b",
    )

    out = capsys.readouterr().out
    assert ok is True
    assert "1" in out and "1284" in out and ci.GATHER_DIR_NAME in out


class _Session:
    def __init__(self, out_dir, frames):
        self.out_dir = Path(out_dir)
        self.frames = list(frames)


def test_recording_info_falls_back_to_the_live_session(tmp_path):
    """teardown 의 recording_stop 단계가 실패하면 result 의 녹화 필드가 빈 채로 남는다.

    run_teardown 은 한 단계가 던져도 나머지를 계속 돌리므로(그래서 수집이 실행된다),
    그 경우에도 세션 자신은 out_dir/frames 를 알고 있다.
    """
    root = tmp_path / "debug_images"
    _touch(root / "align_fail_cycle" / TAG / "paused_match.jpg", 2500.0, b"aaa")

    report = ci.gather_cycle_images(
        _Result(recording_dir="", frame_count=0),
        {"recording": _Session(tmp_path / "rec", ["a.jpg", "b.jpg", "c.jpg"])},
        started_epoch=2000.0, now=3000.0,
        debug_root=root, model_slug="mai-ui-8b",
    )

    assert report.recording_dir == str(tmp_path / "rec")
    assert report.recording_frames == 3


def test_missing_tag_gathers_nothing_instead_of_writing_to_a_garbage_path(tmp_path):
    """tag 가 비면 dest 가 `align_fail_cycle//gathered` 가 된다 - 모으지 않는다."""
    report = ci.gather_cycle_images(
        object(), {}, started_epoch=2000.0, now=3000.0,
        debug_root=tmp_path / "debug_images", model_slug="mai-ui-8b",
    )

    assert report.copied == 0
    assert report.already is True
    assert not (tmp_path / "debug_images").exists()
