# Recovery Episode 수집

> CLAUDE.md 에서 옮겨 온 원문(2026-09-27). 이 기능을 고치기 전에 끝까지 읽을 것 - 계약은 오피스 실측과 설계 결정에서 나왔다.

- **Recovery Episode 수집** (2026-08-30, `ALIGN_FAIL_EPISODE_COLLECT=1`, **기본 off**):
  ALID=9006 active interval 하나 = Recovery Episode 하나. `monitor/recovery_episode.py`
  의 `EpisodeTracker` 가 알람 row 처리(`process_fail_rows`) 부수효과로 이벤트 폴더(`<eqp>-<tag>/`) 루트에
  `recovery_episode.json` 을 **첫 GUI step 전에** 원자적으로 쓰고, cooldown 재시도는 같은
  Episode 의 `attempt_2, 3...` 이 되며, 알람이 poll 에서 사라지면 clearance 이벤트와 함께
  닫힌다. identity 는 uuid4 이고 tag(알람 UTC9)는 **위치일 뿐 identity 가 아니다**;
  재개 판정은 fingerprint(장비+alid+recipe+UTC9) **완전 일치**뿐이며 프로세스당 1회
  이벤트 루트 스캔(고정 깊이)이 유일한 디스크 재구성 경로다(알람 없는 open Episode 는
  `incomplete(alarm_gone_during_restart)`). attempt 산출물은 `<eqp>-<tag>/attempt_<n>/` 아래로
  갈린다 - **cooldown 재시도가 같은 `recording/` 에 두 테이크를 섞던 tag 충돌 결함이 이
  구조로 닫힌다**(별도 수정 금지). 수집 on 이면 attempt 폴더에 `guards.json`
  (`monitor/guard_readings.py`, Episode-level Guard **정확히 셋**: 화면 관측 가능성 /
  점유·제어 / SEM mode+align key 가시성·유일성. 값은 `True/False/None` 이고 **관측 실패는
  전부 unknown** - `false` 로 새지 않는다. 읽은 OM/SEM 은 detail 에만, OK 컨트롤 가용성은
  Guard 가 아니라 precondition), `measurement_verification.json`
  (`monitor/measurement_verification.py`, primary Verification 3상태 record. 자동 reader 는
  **unknown-only stub** 이고 Assist 패널 crop 만 근거로 남긴다 - 열 분리 판독은 오피스
  캘리브레이션 gate 다), `numerator_reads.jsonl`(분자 per-read 판독; fallback Verification 은
  detector 의 boolean 이 아니라 이 기록을 읽는다)이 함께 남는다. Episode 를 닫을 때
  `poc/workflow_4/playbook` 의 순수 evaluator 가 Outcome 을 파생하고 `[DIGEST] episode ...`
  한 줄을 찍는다. 저장 경로는 **Episode-relative 만**(절대/`..` 는 로드가 거부).
  off 면 녹화 폴더·manifest·사이드카·Guard 파일 모두 종전과 동일하다. 스펙/티켓은
  `docs/issues/align-fail-recovery-playbook/`.
