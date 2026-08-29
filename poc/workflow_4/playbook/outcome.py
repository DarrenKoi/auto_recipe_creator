"""Recovery Outcome 파생 - Verification 우선순위와 Outcome 판정의 유일한 소유자.

입력은 plain data 다(workflow_3 파일을 여기서 파싱하지 않는다). 호출부가 Episode 하나의
근거를 아래 모양으로 모아 넘긴다:

    {
      "attempts": [
        {
          "attempt_seq": 1,
          "measurement": {"value": "success"|"failure"|"unknown", ...} | None,
          "numerator_reads": [{"decision": "...", "value": 3}, ...],
          "guards": [{"kind": "...", "value": True|False|None}, ...],
          "abort": {"aborted": True, "reason": "..."} | None,
          "handoff": {"explicit": True, "reason": "..."} | None,
        },
        ...
      ],
      "alarm_cleared": True,
    }

규칙은 세 줄로 요약된다.

  1. **primary 는 Measurement 판독이다.** `success` 만 recovered 를 만든다. `failure` 는
     recovered 가 아니며, fallback 을 열지도 않는다 - 관측된 실패를 카운터 증가로
     뒤집으면 "숫자가 올라갔으니 복구됐다" 가 되어 화면이 깨진 채로 성공이 된다.
  2. **fallback 은 Measurement 가 `unknown` 일 때만** 본다. 분자(N)가 **엄격히 증가한
     연속 표본**이 요구 횟수만큼 있을 때만 success 다. OCR miss, 같음·감소,
     재grounding reset 은 연속을 끊는다.
  3. **그 밖의 신호는 단독으로 recovered 를 만들지 못한다.** 알람 해제, OK 클릭,
     `corrected` status, runner 완료, 커서 정지, 창 닫힘, 닫기 정황은 전부 provenance 다.

Episode 값은 attempt 들을 훑어 정한다. 실패/unknown attempt 뒤에 자격 있는 recovered
attempt 가 오면 Episode 는 recovered 이고, 앞선 attempt 이력은 지워지지 않는다.
그 외에는 abort > 명시 handoff > unknown 순이다. **`handoff` 라는 이름만으로는
escalated 가 되지 않는다** - 호출부가 명시 기록을 넣어야 한다.
"""

from dataclasses import dataclass, field

RECOVERED = "recovered"
ESCALATED = "escalated"
ABORTED = "aborted"
UNKNOWN = "unknown"
OUTCOMES = (RECOVERED, ESCALATED, ABORTED, UNKNOWN)

# Verification 이 어느 경로로 답했는가. 값(성공/실패)과는 다른 축이다.
PATH_PRIMARY = "primary"
PATH_FALLBACK = "fallback"
PATH_UNKNOWN = "unknown"

MEASUREMENT_SUCCESS = "success"
MEASUREMENT_FAILURE = "failure"
MEASUREMENT_UNKNOWN = "unknown"

# 분자 연속을 이어 가는 판정. 그 밖(ocr_miss / equal_or_decrease / reground_reset)은 끊는다.
_RUN_DECISIONS = ("first_sample", "strictly_increasing")
# 표본을 만들지 않은 회차 - 연속을 끊지도, 세지도 않는다.
_NEUTRAL_DECISION = "not_sampled"
# 연속으로 인정하려면 최소한 한 번은 실제 증가가 있어야 한다.
_INCREASE_DECISION = "strictly_increasing"

DEFAULT_MIN_INCREASING_READS = 3

# Guard 값을 digest 한 글자로 줄일 때 쓰는 표기.
_GUARD_TEXT = {True: "true", False: "false", None: "unknown"}


@dataclass(frozen=True)
class AttemptVerdict:
    """attempt 하나의 Verification 판정 - Episode 판정의 재료."""

    attempt_seq: int
    recovered: bool
    verification_path: str
    reason: str


@dataclass(frozen=True)
class OutcomeResult:
    """Episode 하나의 최종 판정 + 어떻게 그렇게 됐는지."""

    outcome: str
    verification_path: str
    reason: str
    deciding_attempt: int = 0
    attempts: tuple = field(default_factory=tuple)


def _numerator_run_length(reads) -> int:
    """분자 기록에서 **마지막까지 이어진** 엄격 증가 연속의 표본 수를 센다.

    끊는 판정을 만나면 0 부터 다시 센다. `not_sampled` 는 표본이 아니므로 세지도,
    끊지도 않는다. 실제 증가(`strictly_increasing`)가 한 번도 없으면 0 이다 - 같은 값을
    세 번 읽은 것을 '증가' 로 볼 수는 없다.
    """
    run = 0
    saw_increase = False
    best = 0
    for read in reads or ():
        decision = str((read or {}).get("decision") or "")
        if decision == _NEUTRAL_DECISION:
            continue
        if decision not in _RUN_DECISIONS:
            run = 0
            saw_increase = False
            continue
        run += 1
        if decision == _INCREASE_DECISION:
            saw_increase = True
        if saw_increase:
            best = max(best, run)
    return best


def evaluate_attempt(
    attempt, *, min_increasing_reads: int = DEFAULT_MIN_INCREASING_READS
) -> AttemptVerdict:
    """attempt 하나가 '관측된 복구' 인지 판정한다(primary -> 필요시 fallback)."""
    attempt = attempt or {}
    seq = int(attempt.get("attempt_seq") or 0)
    measurement = attempt.get("measurement") or {}
    value = str(measurement.get("value") or "") or MEASUREMENT_UNKNOWN

    if value == MEASUREMENT_SUCCESS:
        return AttemptVerdict(seq, True, PATH_PRIMARY, "measurement_success")
    if value == MEASUREMENT_FAILURE:
        # 관측된 실패는 fallback 을 열지 않는다.
        return AttemptVerdict(seq, False, PATH_PRIMARY, "measurement_failure")

    run = _numerator_run_length(attempt.get("numerator_reads"))
    if run >= max(1, int(min_increasing_reads)):
        return AttemptVerdict(seq, True, PATH_FALLBACK, f"numerator_increasing:{run}")
    return AttemptVerdict(
        seq, False, PATH_UNKNOWN,
        f"measurement_unknown:numerator_run={run}",
    )


def _explicit(record) -> bool:
    """명시 기록인가 - 이름이 아니라 기록의 내용이 판정한다."""
    if not isinstance(record, dict):
        return False
    return bool(record.get("explicit") or record.get("aborted"))


def derive_outcome(
    evidence, *, min_increasing_reads: int = DEFAULT_MIN_INCREASING_READS
) -> OutcomeResult:
    """Episode 근거에서 Outcome 을 파생한다.

    우선순위: 자격 있는 recovered attempt > 명시 abort > 명시 handoff > unknown.
    abort 가 handoff 보다 앞인 이유는 긴급 해제가 더 구체적이고 종결적인 사실이기
    때문이다(엔지니어에게 넘겼는데 그 뒤 긴급 해제가 걸렸다면 그 세션은 abort 다).
    """
    evidence = evidence or {}
    attempts = list(evidence.get("attempts") or ())
    verdicts = tuple(
        evaluate_attempt(attempt, min_increasing_reads=min_increasing_reads)
        for attempt in attempts
    )

    for verdict in verdicts:
        if verdict.recovered:
            return OutcomeResult(
                RECOVERED, verdict.verification_path, verdict.reason,
                verdict.attempt_seq, verdicts,
            )

    for attempt in attempts:
        if _explicit(attempt.get("abort")):
            return OutcomeResult(
                ABORTED, PATH_UNKNOWN,
                str((attempt.get("abort") or {}).get("reason") or "abort_latched"),
                int(attempt.get("attempt_seq") or 0), verdicts,
            )
    for attempt in attempts:
        if _explicit(attempt.get("handoff")):
            return OutcomeResult(
                ESCALATED, PATH_UNKNOWN,
                str((attempt.get("handoff") or {}).get("reason") or "handoff_recorded"),
                int(attempt.get("attempt_seq") or 0), verdicts,
            )

    last = verdicts[-1] if verdicts else None
    return OutcomeResult(
        UNKNOWN,
        last.verification_path if last else PATH_UNKNOWN,
        last.reason if last else "no_attempts",
        last.attempt_seq if last else 0,
        verdicts,
    )


def _guard_text(attempt) -> str:
    """digest 용 Guard 3값 요약 - `<kind 축약>:<값>` 을 쉼표로 잇는다."""
    guards = (attempt or {}).get("guards") or ()
    if not guards:
        return "none"
    return ",".join(
        f"{str(guard.get('kind') or '?').split('_')[0]}:"
        f"{_GUARD_TEXT.get(guard.get('value'), 'unknown')}"
        for guard in guards
    )


def format_episode_digest(evidence, result: OutcomeResult) -> str:
    """오피스에서 집으로 복사할 한 줄 - 이미지 없이 이 줄만으로 상태가 읽혀야 한다.

    Episode id 는 앞 8자만 쓴다(사람이 옮겨 적을 수 있어야 하고, 충돌은 파일이 가른다).
    """
    evidence = evidence or {}
    attempts = list(evidence.get("attempts") or ())
    deciding = next(
        (a for a in attempts if int(a.get("attempt_seq") or 0) == result.deciding_attempt),
        attempts[-1] if attempts else {},
    )
    complete = "yes" if evidence.get("complete", True) else "no"
    reasons = evidence.get("incomplete_reasons") or ()
    if complete == "no" and reasons:
        complete = f"no({';'.join(str(r) for r in reasons)})"
    return (
        "[DIGEST] episode "
        f"id={str(evidence.get('episode_id') or '')[:8]} "
        f"eqp={evidence.get('eqp_id') or '-'} "
        f"recipe={evidence.get('recipe_id') or '-'} "
        f"attempts={len(attempts)} "
        f"outcome={result.outcome} "
        f"guards={_guard_text(deciding)} "
        f"verify={result.verification_path} "
        f"complete={complete}"
    )


__all__ = [
    "ABORTED",
    "DEFAULT_MIN_INCREASING_READS",
    "ESCALATED",
    "OUTCOMES",
    "PATH_FALLBACK",
    "PATH_PRIMARY",
    "PATH_UNKNOWN",
    "RECOVERED",
    "UNKNOWN",
    "AttemptVerdict",
    "OutcomeResult",
    "derive_outcome",
    "evaluate_attempt",
    "format_episode_digest",
]
