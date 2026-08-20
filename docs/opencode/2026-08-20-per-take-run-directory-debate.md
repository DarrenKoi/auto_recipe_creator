# per-take run directory 설계 토론 (oc-discuss)

- 날짜: 2026-08-20
- 모델: `glm-5.3` (variant high, tier heavy), 2라운드, 208s + 44s
- 세션: `ses_fe2b7de4bffekzMpDch4tugSIw`
- 대상 설계: `poc/workflow_3/docs/superpowers/specs/2026-08-20-per-take-run-directory-design.md`
- 입장: 알람 사이클 1건의 모든 증거(녹화/보정/게이트/step 저널/콘솔)를
  `runs/<tag>_<eqp>_<recipe>/` 이벤트 폴더 하나로 모으고, `align_images/` 는 MES 입력
  전용으로 되돌린다. 반증 조건 - 엔지니어가 실패 알람 1건을 진단하는 데 여전히 루트를
  2개 이상 열어야 하거나, 이전 과정에서 프레임을 하나라도 잃으면 설계 실패.

## 합의된 것

- 방향 자체(테이크당 폴더 하나, 경로 구성점 단일화, join key 단일화)는 옳다.
  모델 1라운드: *"the direction (one folder per take, one path module, single join key)
  is right. But the design has one real hole and two under-specified risks."*
- `tag` 는 **테이크 식별자가 아니다**. 알람 UTC9 에서 유도되므로 cooldown 재시도가 같은 tag 를
  재사용한다. 설계 전제였던 "사이클은 이미 유일한 식별자를 갖는다" 는 거짓.
- 콘솔 tee 를 프로세스 중간에 갈아 끼우는 방식은 폐기. 기동 시 preflight 로 대체하고
  사이클 중 폴백은 두지 않는다.
- 테이크 suffix 로직은 `run_dirs.py` 가 소유한다. `workflow_3e` 와 check-only 진입점이
  자기 이름 규칙을 따로 두면 충돌이 재현된다.

## 모델의 반론 (verbatim)

> **1. `tag` is not unique per take — it's unique per alarm row. Worst objection, and it's
> fatal as specified.**
> `align_fail_monitor.py:445` derives the tag from the alarm's UTC9 (`_alarm_time_to_tag`),
> second resolution. A failed cycle goes to cooldown and the *same persistent alarm row* is
> retried with the *same tag* → same `runs/<tag>_<eqp>_<recipe>/`. Take 2 overwrites take 1's
> `summary.json` and `step_*.json`, and `RecordingSession` writes frames under the same tag
> into the same `02_recording/` — filename collisions overwrite the only evidence of the first
> take. This fails your own falsifiability test ("any frame lost") *by design, on every retry*,
> and retries of failed alarms are exactly the cycles that get diagnosed.

> **2. Per-cycle console tee is a one-line bullet hiding the hardest problem.**
> The monitor is one long-running process with background threads (recording thread, notifier
> watchdog, access watcher). "이 사이클만의 stdout 사본" means swapping a tee mid-process —
> cross-thread prints land in whichever cycle's log is active, and lines during teardown get
> misattributed or dropped. And encoding is unspecified: cp949 can't encode em-dash; a
> locale-default tee writer crashes cycles, and tee-ing raw cp949 bytes mangles content.

> **3. The step-5 auto-fallback contradicts the design's own arguments.**
> You reject more rollback switches because "consumers' globs stay dual forever" — then step
> 5's mkdir-failure fallback re-creates exactly that: two live-write layouts, legacy globs kept
> permanently, and when fallback fires the engineer *must* open two roots — your declared
> failure condition. Since office feedback is pasted console text, a silently fallen-back
> office runs the wrong layout for months (the MES-drift precedent).

2라운드에서 유지한 압박:

> **The tag-stamping for events.log is bigger than it reads.** `log_work2_event` calls in the
> correction path (`cycle.py`, feasibility, zoom ladder, `align/correction`) pass `eqp_id` but
> no `tag` today. The events.log replacement is only as good as retrofitting `tag=` onto every
> call site — miss the deep ones inside `correct_align_fail_auto` and the log silently drops
> the exact cycles you built it for. Count the call sites before committing to the migration
> step size.

> **Your stateless `__a2` suffix must live in `run_dirs.py`, not the monitor** — `workflow_3e/
> abort_cycle` mints tags the same way and will reintroduce the collision if it keeps its own
> naming.

모델이 스스로 기각한 반론:

> **Strongest objection I rejected:** moving recordings out of `align_images` breaks invisible
> gitignored office consumers of `captured_img_from_rcs`. Real, but engineers are the
> consumers, README documents the change, and `runs/` is strictly better for them.

## 내가 틀렸던 것

1. **"사이클은 이미 유일한 식별자 `tag` 를 갖는다"** - 거짓이다. `tag` 는 알람 UTC9 에서
   나오며(`align_fail_monitor.py:130-141`, 의도가 docstring 에 명시) cooldown 재시도가
   같은 값을 재사용한다. 설계의 헤드라인 주장이 이 전제 위에 서 있었다.
   → `tag` + 파일시스템 유도 `attempt` suffix 로 교체.
2. **"`align_images` 는 루프에게 읽기 전용"** - 거짓이다(모델도 못 잡았고 내가 코드에서
   찾았다). `rcp_msr_gather.py:164` 가 `ALIGN_IMAGES_DIR/<eqp>/<recipe_id>` 로 MES 자료를
   내려받는다. 정확한 계약은 **"MES 입력은 쓰되 증거물은 없다"**.
3. **사이클 중 mkdir 폴백** - 스스로 반대한 이중 레이아웃을 자기가 만들었다.
   기동 preflight 로 교체.
4. **콘솔 tee** - 장수 프로세스 + 배경 스레드에서 안전하지 않다. `log_work2_event` 감사
   로그의 테이크별 추출로 교체.

## 실측으로 확인한 것 (토론 후 검증)

- `log_work2_event` 호출부 **42곳, `tag=` 를 넘기는 곳 0곳**. 시그니처가 `**fields` 라
  필드 추가는 자유롭지만, 42곳 retrofit 대신 `logger.py` 의 테이크 바인딩 2곳 편집으로
  간다(순차 처리 전제에 기대는 한계를 설계서에 명시).
- `workflow_3e/dispatch.py:103` 이 `_alarm_time_to_tag` 를 **import 해서 그대로 쓴다**.
  같은 충돌을 공유하므로 `run_dirs.py` 공유가 필수.
- 프레임 파일명이 `<tag>_rcs_<seq>_<elapsed_ms>ms.jpg` 라, 재시도의 지배적 피해는
  덮어쓰기가 아니라 **두 테이크의 뒤섞임**이다(모델의 기술보다 조용하고 나쁘다).
  `make_demo_video` 의 시간축 복원과 `recording_filter` Stage 1 이 오염된다.

## 여전히 미해결

- 테이크 바인딩의 오귀속 - `gather_success_async` 처럼 사이클 밖에서 도는 비차단 작업의
  늦은 로그는 이전 테이크로 귀속될 수 있다. 순차 처리 전제가 깨지면 커진다.
  판별 방법: 오피스에서 한 사이클 돌린 뒤 `events.log` 의 tag 불일치 줄을 센다.
- 수동 녹화/시연을 `runs/` 로 통일할지는 범위 밖으로 남겼다.
