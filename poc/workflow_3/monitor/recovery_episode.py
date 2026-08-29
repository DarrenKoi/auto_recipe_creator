"""Recovery Episode 수집 - ALID=9006 active interval 하나 = Episode 하나.

한 번의 Align Fail 이 뜨고 풀릴 때까지가 Episode 하나다. cooldown 재시도는 같은
Episode 의 attempt 2, 3... 이고, 알람이 poll 에서 사라지면 clearance 이벤트를 남기고
Episode 가 닫힌다. 그 뒤 같은 장비·레시피가 다시 실패하면 **새 Episode** 다.

식별 규약 둘:

  * `episode_id` 는 uuid4 다. 경로나 타임스탬프에서 재구성되지 않는다 - tag 는
    '어디에 쌓였는가'(위치)이고 identity 가 아니다. tag 는 알람 UTC9 초 해상도라
    같은 알람의 재시도가 같은 tag 로 돌아오는 반면, 서로 다른 Episode 가 같은 tag 를
    가질 수도 있다.
  * `fingerprint` 는 장비 + alarm code + recipe + 원 UTC9 의 안정 문자열이다.
    재시작 후 이 값이 **완전히** 일치할 때만 열린 Episode 를 재개한다.

파일은 절대 지우지 않는다. 수집이 깨진 Episode 는 `complete=false` + 사유로 남는다 -
없어진 근거는 되살릴 수 없고, "수집이 깨졌다" 자체가 관측이기 때문이다.

Mac 확인(실장비 없이 생성 -> attempt -> clearance -> closed 를 파일로 본다).
replay CSV 의 UTC9 는 `ALIGN_FAIL_WINDOW_SEC`(기본 60s) 안이어야 방출된다:

    ALIGN_FAIL_EPISODE_COLLECT=1 SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay \\
      ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \\
      uv run python poc/workflow_3/monitor/align_fail_monitor.py

RCS 모듈이 없으므로 attempt 는 실패로 끝나지만, 첫 poll 에 Episode 가 생기고 두 번째
빈 poll 에서 clearance 이벤트와 함께 닫히는 것이 `recovery_episode.json` 에 보인다.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR

SCHEMA_VERSION = "recovery_episode.v1"
OBSERVATION_CONTRACT = "align_fail_observation.v1"
EPISODE_FILENAME = "recovery_episode.json"

# rcs/rcs_screenshot.py 의 같은 이름 상수와 같은 값이다. 그쪽에서 import 하지 않는
# 이유는 그 모듈이 pywinauto(Windows 전용)를 최상단에서 끌어와 Mac 에서 import 자체가
# 실패하기 때문이다 - Episode 경로 계산은 실장비 없이도 성립해야 한다.
CAPTURED_RCS_DIRNAME = "captured_img_from_rcs"
UNREGISTERED_DIRNAME = "_unregistered"

# attempt 폴더에 남는 관측 record 파일 - (artifacts 키, 파일명). 존재하는 것만 참조로 건다.
_ATTEMPT_RECORD_FILES = (
    ("guards", "guards.json"),
    ("measurement_verification", "measurement_verification.json"),
    ("numerator_reads", "numerator_reads.jsonl"),
)

# 자동 보정이 **명시적으로** 엔지니어에게 넘긴 outcome status. 이 목록에 있을 때만
# handoff 기록을 남긴다 - 'handoff' 라는 노드/상태 이름만으로는 escalated 가 되지 않는다는
# 규약이라, 여기 없는 status(예: 사이클 실패, 관전)는 escalated 로 승격되지 않는다.
_HANDOFF_STATUSES = (
    "awaiting_engineer_ok",
    "escalated_ambiguous_key",
    "escalated_key_not_visible",
    "escalated_no_ok",
)


def _abort_record(abort_fn=None):
    """긴급 해제 래치가 걸렸으면 명시 abort 기록을 만든다(아니면 None)."""
    try:
        if abort_fn is not None:
            aborted, reason = abort_fn()
        else:
            from poc.workflow_3.util.abort_switch import abort_reason, is_aborted

            aborted, reason = is_aborted(), abort_reason()
    except Exception:
        return None
    if not aborted:
        return None
    return {"aborted": True, "reason": str(reason or "abort_latched")}


def _handoff_record(outcome_status: str):
    """보정이 명시적으로 엔지니어에게 넘긴 경우에만 handoff 기록을 만든다."""
    status = str(outcome_status or "")
    if status not in _HANDOFF_STATUSES:
        return None
    return {"explicit": True, "reason": status}


# attempt 가 '수집 완료' 로 볼 수 없는 run_status. GUI 를 아예 못 돌린 경우다.
_INCOMPLETE_RUN_STATUSES = {"error", "rcs_unavailable", "cycle_disabled"}


def _now_iso() -> str:
    """로컬 시각 ISO 문자열(초 해상도) - provenance 이며 identity 가 아니다."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def episode_root_for(images_root, eqp_id: str, recipe_id: str, tag: str) -> Path:
    """Episode 루트 = captured_img_from_rcs/<tag> (recipe 없으면 _unregistered/<tag>).

    `recipe_id` 는 실제로 ``<class>/<recipe>`` 형태라 슬래시가 그대로 단계 구분이 된다
    (`rcs_screenshot.captured_dir_for` 와 같은 규약). **순수 경로 함수**이며 Windows
    전용 모듈에 의존하지 않는다 - cycle 의 폴더 resolver 도 이것을 쓴다.
    """
    root = Path(images_root)
    recipe_rel = (recipe_id or "").replace("\\", "/").strip("/")
    parts = [part for part in recipe_rel.split("/") if part]
    if parts:
        return root.joinpath(eqp_id, *parts, CAPTURED_RCS_DIRNAME, str(tag))
    return root / eqp_id / UNREGISTERED_DIRNAME / str(tag)


def alarm_fingerprint(info) -> str:
    """장비 + alarm code + recipe + 원 UTC9 로 안정 fingerprint 를 만든다.

    재시작 후 Episode 재개 판정의 유일한 기준이라 **네 값이 모두** 들어간다. 하나라도
    다르면 다른 알람으로 본다(같은 장비의 다음 실패를 이전 Episode 에 붙이지 않는다).
    """
    info = info or {}
    return "|".join(
        str(info.get(key) or "").strip()
        for key in ("eqp_id", "alid", "recipe_id", "utc9")
    )


def _settings_snapshot(settings) -> dict:
    """attempt 에 남길 actuation 게이트 스냅샷 - dry-run 여부가 provenance 로 보인다."""
    return {
        "safe_mode": bool(getattr(settings, "safe_mode", False)),
        "correction_dry_run": bool(getattr(settings, "correction_dry_run", True)),
        "action_enabled": bool(getattr(settings, "action_enabled", False)),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    """temp 파일에 쓰고 os.replace 로 갈아끼운다(부분 기록된 정본을 남기지 않는다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _reject_unsafe_ref(value: str, where: str) -> None:
    """Episode-relative 가 아닌 artifact 참조를 거부한다.

    Episode 폴더는 통째로 옮겨질 수 있어야 하므로(오피스 -> 보관소) 저장 경로는
    루트 기준 상대 경로뿐이다. 절대 경로(POSIX `/`, Windows 드라이브/UNC)와 `..`
    탈출은 로드 시점에 막는다 - 파일을 읽는 쪽이 마지막 방어선이다.
    """
    text = str(value or "").replace("\\", "/")
    if not text:
        return
    if text.startswith("/") or text.startswith("//") or (len(text) > 1 and text[1] == ":"):
        raise ValueError(f"absolute artifact path in {where}: {value!r}")
    if ".." in [part for part in text.split("/") if part]:
        raise ValueError(f"parent-escaping artifact path in {where}: {value!r}")


def relative_ref(root, value) -> str:
    """Episode root 기준 상대 참조로 바꾼다. root 밖이거나 비면 빈 문자열.

    root 밖의 경로는 **참조하지 않는다** - Episode 폴더가 통째로 옮겨져도 provenance 가
    깨지지 않아야 하므로, 밖을 가리키는 절대 경로를 적느니 참조를 비워 둔다
    (runner journal 처럼 밖에 사는 것은 경로가 아니라 id 로 참조한다).
    """
    if not value:
        return ""
    try:
        rel = Path(str(value)).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return ""
    return rel.as_posix()


def load_episode(path) -> dict:
    """recovery_episode.json 을 읽고 artifact 경로 규약을 검증한다.

    검증에 실패하면 값을 고쳐 읽지 않고 ValueError 를 던진다 - 규약을 어긴 참조를
    조용히 정규화하면 어느 파일이 근거인지 알 수 없게 된다.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for attempt in data.get("attempts") or []:
        where = f"attempt_{attempt.get('attempt_seq')}"
        for key, value in (attempt.get("artifacts") or {}).items():
            if isinstance(value, str):
                _reject_unsafe_ref(value, f"{where}.{key}")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _reject_unsafe_ref(str(item), f"{where}.{key}")
    return data


@dataclass
class AttemptHandle:
    """열린 attempt 하나를 가리키는 값 - 호출부가 결과를 되돌려줄 때 쓴다."""

    episode_id: str
    attempt_seq: int
    tag: str
    root: Path


class EpisodeTracker:
    """장비 -> 열린 Episode 의 **메모리** 맵. 파일이 정본이고 이 맵은 캐시다."""

    def __init__(self, images_root=None):
        self.images_root = Path(images_root) if images_root else ALIGN_IMAGES_DIR
        self._open: dict[str, dict] = {}
        # 디스크 재구성은 프로세스당 한 번뿐이다(첫 poll). 이후의 진실은 메모리 맵이다.
        self._scanned = False
        # fallback Verification 이 요구하는 엄격 증가 표본 수. begin_attempt 에서
        # settings 로 갱신된다(설정과 판정이 갈리면 안 된다).
        self._min_increasing_reads = 3

    # ---- 내부 ----

    def _path_for(self, episode: dict) -> Path:
        return Path(episode["_root"]) / EPISODE_FILENAME

    def _persist(self, episode: dict) -> None:
        """정본을 원자적으로 쓴다. 실패해도 루프를 죽이지 않는다(경고만)."""
        payload = {key: value for key, value in episode.items() if not key.startswith("_")}
        try:
            _write_json_atomic(self._path_for(episode), payload)
        except Exception as exc:
            print(f"[WARNING] recovery_episode 기록 실패: {exc}")

    def _next_event(self, episode: dict, kind: str, attempt_seq=None, **detail) -> None:
        """Episode 전체를 관통하는 단조 event_seq 로 이벤트 1건을 append 한다."""
        seq = int(episode.get("next_event_seq", 1))
        episode["next_event_seq"] = seq + 1
        episode["events"].append({
            "event_seq": seq,
            "attempt_seq": attempt_seq,
            "kind": kind,
            "at": _now_iso(),
            "detail": detail,
        })

    def _new_episode(self, info, tag: str) -> dict:
        eqp_id = str(info.get("eqp_id") or "")
        root = episode_root_for(
            self.images_root, eqp_id, str(info.get("recipe_id") or ""), tag
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_contract": OBSERVATION_CONTRACT,
            "bindings_version": None,
            "episode_id": str(uuid.uuid4()),
            "alarm": dict(info),
            "fingerprint": alarm_fingerprint(info),
            "execution_mode": "live",
            "tag": tag,
            "state": "open",
            "opened_at": _now_iso(),
            "closed_at": None,
            "next_event_seq": 1,
            "attempts": [],
            "events": [],
            "outcome": "unknown",
            "recovery_actors": [],
            "complete": True,
            "incomplete_reasons": [],
            "_root": str(root),
        }

    # ---- 공개 API ----

    def resume_from_disk(self, current_fingerprints) -> None:
        """첫 poll 에 capture tree 를 한 번 훑어 열린 Episode 를 되찾는다.

        장비->Episode 맵은 메모리에만 있으므로, 프로세스가 재시작하면 진행 중이던
        Episode 가 디스크에만 남는다. 이 스캔이 **유일한** 디스크 재구성 경로다.

          * fingerprint 가 이번 poll 의 알람과 **완전히** 일치하면 재개한다.
          * 그렇지 않으면 `incomplete(alarm_gone_during_restart)` 로 닫는다.

        깨진 파일 하나 때문에 모니터가 뜨지 못하면 안 되므로, 스캔 전체와 파일 하나
        모두 예외를 삼키고 경고만 남긴다(파일은 지우지 않는다).
        """
        if self._scanned:
            return
        self._scanned = True
        wanted = {str(value) for value in (current_fingerprints or ())}
        try:
            paths = sorted(self.images_root.rglob(EPISODE_FILENAME))
        except Exception as exc:
            print(f"[WARNING] Episode 스캔 실패(건너뜀): {exc}")
            return
        resumed = orphaned = 0
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("state") != "open":
                    continue
                data["_root"] = str(path.parent)
                eqp_id = str((data.get("alarm") or {}).get("eqp_id") or "")
                if data.get("fingerprint") in wanted and eqp_id not in self._open:
                    self._open[eqp_id] = data
                    resumed += 1
                else:
                    self._mark_episode_incomplete(data, "alarm_gone_during_restart")
                    self._close(data, "alarm_gone_during_restart", eqp_id=eqp_id)
                    orphaned += 1
            except Exception as exc:
                print(f"[WARNING] Episode 파일을 건너뜀({path}): {exc}")
        if resumed or orphaned:
            print(
                f"[INFO] Episode 스캔: 재개={resumed}, "
                f"alarm_gone_during_restart={orphaned}"
            )

    def begin_attempt(self, info, settings, *, tag: str) -> AttemptHandle:
        """이 알람의 Episode 를 열거나 재개하고 attempt 하나를 시작한다.

        정본은 **첫 GUI step 전에** 디스크에 있다 - 사이클이 예외로 끝나도 "이 알람을
        건드렸다" 는 사실이 남아야 한다.

        같은 장비에 열린 Episode 가 있어도 fingerprint 가 하나라도 다르면 재개하지
        않는다 - 그건 같은 알람의 재시도가 아니라 **다음 실패**이며, 이어 붙이면 서로
        다른 두 사건이 한 Episode 로 뭉개진다.
        """
        eqp_id = str(info.get("eqp_id") or "")
        self._min_increasing_reads = max(
            1, int(getattr(settings, "engineer_done_numerator_increase_reads", 3) or 3)
        )
        episode = self._open.get(eqp_id)
        if episode is not None and episode.get("fingerprint") != alarm_fingerprint(info):
            self._mark_episode_incomplete(episode, "fingerprint_changed")
            self._close(episode, "fingerprint_changed", eqp_id=eqp_id)
            self._open.pop(eqp_id, None)
            episode = None
        if episode is None:
            episode = self._new_episode(info, tag)
            self._open[eqp_id] = episode

        attempt_seq = len(episode["attempts"]) + 1
        episode["attempts"].append({
            "attempt_seq": attempt_seq,
            "started_at": _now_iso(),
            "finished_at": None,
            "execution_mode": "live",
            "settings": _settings_snapshot(settings),
            "run_id": "",
            "run_status": "",
            "failed_step": "",
            "failure_class": "",
            "outcome_status": "",
            "artifacts": {"dir": attempt_dirname(attempt_seq)},
            # 명시 기록만 Outcome 을 escalated/aborted 로 만든다 - 노드나 status 이름은
            # 근거가 아니다. finish_attempt 가 실제로 관측된 경우에만 채운다.
            "abort": None,
            "handoff": None,
            "complete": False,
            "incomplete_reason": "in_progress",
        })
        self._next_event(episode, "attempt_started", attempt_seq)
        self._persist(episode)
        return AttemptHandle(
            episode_id=episode["episode_id"],
            attempt_seq=attempt_seq,
            tag=episode["tag"],
            root=Path(episode["_root"]),
        )

    def _attempt(self, handle: AttemptHandle):
        """handle 이 가리키는 (Episode, attempt) 를 찾는다. 없으면 (None, None)."""
        for episode in self._open.values():
            if episode["episode_id"] != handle.episode_id:
                continue
            for attempt in episode["attempts"]:
                if attempt["attempt_seq"] == handle.attempt_seq:
                    return episode, attempt
        return None, None

    def _mark_episode_incomplete(self, episode: dict, reason: str) -> None:
        """수집이 attempt 밖의 이유로 깨진 Episode 를 사유와 함께 미완으로 표시한다."""
        if reason not in episode["incomplete_reasons"]:
            episode["incomplete_reasons"].append(reason)
        episode["complete"] = False

    def _mark_incomplete(self, episode: dict, attempt: dict, reason: str) -> None:
        """attempt 를 미완 처리하고 Episode 의 사유 목록에 합친다(파일은 안 지운다)."""
        attempt["complete"] = False
        attempt["incomplete_reason"] = reason
        label = f"attempt_{attempt['attempt_seq']}:{reason}"
        if label not in episode["incomplete_reasons"]:
            episode["incomplete_reasons"].append(label)
        episode["complete"] = False

    def _attach_attempt_records(self, root: Path, attempt: dict) -> None:
        """attempt 폴더에 실제로 쓰인 관측 record 파일만 골라 참조로 건다.

        존재하는 파일만 적는다 - 없는 파일을 가리키는 참조는 provenance 가 아니라
        거짓말이다. 새 record 종류는 `_ATTEMPT_RECORD_FILES` 에 한 줄 추가한다.
        """
        attempt_dir = root / attempt_dirname(attempt["attempt_seq"])
        for key, name in _ATTEMPT_RECORD_FILES:
            if (attempt_dir / name).is_file():
                attempt["artifacts"][key] = f"{attempt_dirname(attempt['attempt_seq'])}/{name}"

    def finish_attempt(self, handle: AttemptHandle, cycle, *, abort_fn=None) -> None:
        """사이클 결과를 attempt 에 반영한다(`CycleResult` 를 plain 값으로 받아 적는다).

        `run_dir` 은 basename(run id)만 남긴다 - runner journal 은 Episode root 밖에
        있고 경로가 아니라 id 로 참조된다는 규약이다.
        """
        episode, attempt = self._attempt(handle)
        if attempt is None:
            return
        run_status = str(getattr(cycle, "run_status", "") or "")
        root = Path(episode["_root"])
        for key, value in (
            ("recording", getattr(cycle, "recording_dir", "")),
            ("prelude", getattr(cycle, "prelude_dir", "")),
        ):
            ref = relative_ref(root, value)
            if ref:
                attempt["artifacts"][key] = ref
        self._attach_attempt_records(root, attempt)
        attempt.update({
            "finished_at": _now_iso(),
            "run_id": Path(str(getattr(cycle, "run_dir", "") or "")).name,
            "run_status": run_status,
            "failed_step": str(getattr(cycle, "failed_step", "") or ""),
            "failure_class": str(getattr(cycle, "failure_class", "") or ""),
            "outcome_status": str(getattr(cycle, "outcome_status", "") or ""),
        })
        attempt["abort"] = _abort_record(abort_fn)
        attempt["handoff"] = _handoff_record(attempt["outcome_status"])
        if run_status in _INCOMPLETE_RUN_STATUSES:
            self._mark_incomplete(episode, attempt, f"run_status:{run_status}")
        else:
            attempt["complete"] = True
            attempt["incomplete_reason"] = ""
        self._next_event(episode, "attempt_finished", handle.attempt_seq,
                         run_status=run_status)
        self._persist(episode)

    def _read_attempt_evidence(self, root: Path, attempt: dict) -> dict:
        """attempt 폴더의 관측 record 를 읽어 **plain data** 로 만든다.

        workflow_4 의 evaluator 는 workflow_3 파일을 파싱하지 않으므로, 파일에서
        값을 꺼내는 일은 여기(생산자 쪽)가 한다. 읽기 실패는 그 항목을 비우고 넘어간다 -
        판정은 그때 unknown 이 되며 그것이 정직한 결과다.
        """
        attempt_dir = root / attempt_dirname(attempt.get("attempt_seq") or 1)
        evidence = {
            "attempt_seq": attempt.get("attempt_seq"),
            "measurement": None,
            "numerator_reads": [],
            "guards": [],
            "abort": attempt.get("abort"),
            "handoff": attempt.get("handoff"),
        }
        try:
            path = attempt_dir / "measurement_verification.json"
            if path.is_file():
                evidence["measurement"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARNING] Measurement record 읽기 실패(unknown 으로 진행): {exc}")
        try:
            path = attempt_dir / "guards.json"
            if path.is_file():
                evidence["guards"] = json.loads(
                    path.read_text(encoding="utf-8")
                ).get("guards") or []
        except Exception as exc:
            print(f"[WARNING] Guard record 읽기 실패(진행): {exc}")
        try:
            path = attempt_dir / "numerator_reads.jsonl"
            if path.is_file():
                evidence["numerator_reads"] = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        except Exception as exc:
            print(f"[WARNING] numerator 기록 읽기 실패(진행): {exc}")
        return evidence

    def build_episode_evidence(self, episode: dict) -> dict:
        """Episode 하나의 근거를 workflow_4 evaluator 가 받는 모양으로 모은다."""
        root = Path(episode["_root"])
        alarm = episode.get("alarm") or {}
        return {
            "episode_id": episode.get("episode_id", ""),
            "eqp_id": alarm.get("eqp_id", ""),
            "recipe_id": alarm.get("recipe_id", ""),
            "complete": bool(episode.get("complete", True)),
            "incomplete_reasons": list(episode.get("incomplete_reasons") or ()),
            "alarm_cleared": episode.get("state") == "closed",
            "attempts": [
                self._read_attempt_evidence(root, attempt)
                for attempt in episode.get("attempts") or ()
            ],
        }

    def _derive_and_report(self, episode: dict) -> None:
        """final Outcome 을 파생해 Episode 에 쓰고 digest 를 한 줄 찍는다.

        workflow_4 import 는 **지연·보호**한다 - workflow_3 가 workflow_4 에 하드
        의존하지 않는다는 규약이고(graph view mirror 와 같은 형태), 그 계층이 없다고
        알람 처리가 깨지면 안 된다. 없으면 outcome 은 `unknown` 으로 남는다.
        """
        try:
            from poc.workflow_4.playbook.outcome import (
                derive_outcome,
                format_episode_digest,
            )
        except Exception as exc:
            print(f"[WARNING] Outcome evaluator 없음 - outcome=unknown 유지: {exc}")
            return
        try:
            evidence = self.build_episode_evidence(episode)
            result = derive_outcome(
                evidence, min_increasing_reads=self._min_increasing_reads
            )
            episode["outcome"] = result.outcome
            episode["outcome_detail"] = {
                "verification_path": result.verification_path,
                "reason": result.reason,
                "deciding_attempt": result.deciding_attempt,
            }
            print(format_episode_digest(evidence, result))
        except Exception as exc:
            print(f"[WARNING] Outcome 파생 실패 - outcome=unknown 유지: {exc}")

    def _close(self, episode: dict, kind: str, **detail) -> None:
        """Episode 를 닫고(state/closed_at) final Outcome + digest 를 낸다."""
        self._next_event(episode, kind, None, **detail)
        episode["state"] = "closed"
        episode["closed_at"] = _now_iso()
        self._derive_and_report(episode)
        self._persist(episode)

    def close_cleared(self, current_eqp_ids) -> list:
        """이번 poll 의 알람 목록에 없는 열린 Episode 를 clearance 로 닫는다.

        판정 기준은 `active_tools` 가 아니라 **알람이 아직 보이는가** 다. cooldown 으로
        재시도를 유예한 tool 은 알람이 그대로라 닫히지 않아야 하고, 반대로 active 에
        등록되지 못한 채 알람만 사라진 tool 도 닫혀야 한다.
        """
        current = {str(eqp) for eqp in (current_eqp_ids or ())}
        closed = []
        for eqp_id in [key for key in self._open if key not in current]:
            episode = self._open.pop(eqp_id)
            self._close(episode, "alarm_cleared", eqp_id=eqp_id)
            closed.append(episode["episode_id"])
        return closed

    def fail_attempt(self, handle: AttemptHandle, reason: str) -> None:
        """사이클이 예외로 끝난 attempt 를 사유와 함께 미완으로 닫는다."""
        episode, attempt = self._attempt(handle)
        if attempt is None:
            return
        attempt["finished_at"] = _now_iso()
        self._mark_incomplete(episode, attempt, reason)
        self._next_event(episode, "attempt_error", handle.attempt_seq, reason=reason)
        self._persist(episode)


def attempt_dirname(attempt_seq: int) -> str:
    """attempt 폴더 이름 - Episode-relative 경로의 첫 단계."""
    return f"attempt_{int(attempt_seq)}"


__all__ = [
    "SCHEMA_VERSION",
    "OBSERVATION_CONTRACT",
    "EPISODE_FILENAME",
    "AttemptHandle",
    "EpisodeTracker",
    "alarm_fingerprint",
    "attempt_dirname",
    "episode_root_for",
    "load_episode",
    "relative_ref",
]
