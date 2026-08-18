# oc-review — RCS 부재 시 자동 복구 (2026-08-19)

- **고정점**: `HEAD` (`f647f88`) 대비 **미커밋 작업 트리**. 새 파일 2개는 `git add -N`
  으로 등록해 `git diff HEAD` 에 나오게 했다.
- **티어**: `heavy` (glm-5.3, variant high). 파일 8개지만 프로세스를 띄우고 자격증명을
  무인 입력하는 live-adapter 표면이고 안전 기본값을 뒤집는 변경이라 상향.
- **표준 출처** (오버레이 없음, 폴백 탐지): `CLAUDE.md`(루트), `poc/workflow_3/README.md`.
  리뷰 중 모델이 `AGENTS.md` 도 인용했고 실재를 확인했다.
- **스펙**: `_workspace/spec-rcs-recovery.md` (구현 전 합의 내용을 대화에서 옮겨 적은 것.
  gitignore 되는 스크래치 경로 — opencode 샌드박스가 읽을 수 있게 저장소 안에 둠).
- **결과**: 두 축 모두 exit 0. Standards 3건(HARD 1, JUDGEMENT 2), Spec 4건(누락 0,
  범위이탈 2, 구현의심 2).

## Standards (exit 0) — verbatim

> ## Review: RCS recovery extraction + default-on
>
> **HARD — documented breach**
>
> 1. **Docs not updated with the default flip.** `poc/workflow_3/README.md:213` still documents `ALIGN_FAIL_RCS_RECOVERY | 0 | RCS 메인 창 부재 시 재실행+재로그인 복구`, but `config.py` now defaults `True` in both the dataclass and `env_flag(..., default=True)`. AGENTS.md: "Update docs together with code when changing ... safety behavior"; CLAUDE.md names the README env table as authoritative for env flags. Turning "launch an exe + auto-login" on by default is exactly the class of behavior change that section covers.
>
> **JUDGEMENT — baseline smells**
>
> 2. **Duplicated config representation** (`rcs_recovery.py:12`): `RECOVERY_WINDOW_TIMEOUT_SEC = env_float("ALIGN_FAIL_RCS_RECOVERY_WINDOW_SEC", 30.0)` is read at module import, outside `Workflow3Settings` where the sibling tunable `connect_window_timeout_sec` lives. Consequence beyond symmetry: `cycle.py` imports this module at load time, *before* the monitors' `__main__` calls `seed_env()`, so `workflow_3_config.py` can never tune it (the same documented limitation CLAUDE.md notes for `ALIGN_IMAGES_DIR`). Moving it into `Workflow3Settings` fixes both.
>
> 3. **Hidden positional coupling** (`test_rcs_recovery.py`, `_ensure_step`): `build_cycle_steps("MCD916")[0]` assumes the ensure-RCS step is index 0. A step reorder silently retargets all three wiring tests to the wrong step. Asserting `step.name`/kind would make the failure loud.
>
> **Checked, clean**
>
> - Korean docstrings throughout; no `__future__`; no argparse; absolute `poc.workflow_3.*` imports (including the new function-level ones).
> - All `print()` strings use `[INFO]`/`[WARNING]` prefixes and plain hyphens — no em-dash in any printed string (the em-dash in the `requirements.txt` comment is outside the cp949 rule, which is print-only).
> - `psutil` added consistently to `pyproject.toml`, `requirements.txt`, and `uv.lock`; the guarded-import `PSUTIL_AVAILABLE` pattern is respected, and the None/[] "unknown vs absent" split is a genuine robustness fix, well tested.
> - Default-on recovery retains the `ALIGN_FAIL_RCS_RECOVERY=0` rollback and matches the change's stated intent, so I don't flag it against the safe-default convention.
>
> Fix #1 (one-line README edit) and #2 before commit; #3 is optional.

## Spec (exit 0) — verbatim

> The wiring looks correct. Let me verify the tests actually pass on Mac (R6 requirement):
> One more check on the env reader semantics (R5 "셸 env 가 항상 이겨야 한다"):
> **Verification:** All 12 new tests pass on Mac (`uv run pytest poc/workflow_3/monitor/test_rcs_recovery.py` → 12 passed), satisfying R6's "Mac 에서 RCS 없이 전부 통과해야 한다".
>
> ## (a) Missing / partly done
>
> Essentially nothing material. All six requirements (R1–R6) are implemented: injected `recover_rcs_session` module (R1), unknown-vs-empty process scan distinction + psutil declared in `pyproject.toml`, `requirements.txt`, `uv.lock` (R2), `resolve_login_tool_name` with `""` = no tool / `None` = env, and `build_login_workflow_steps` confirms `""` skips all tool steps (workflow_login.py:228-229) (R3), status-not-exception outcomes wired to `failure_class` distinct from `rcs_unavailable`, window reused without re-scan (cycle.py:290) (R4), default True in both dataclass and env reader; plain `load_dotenv()` means shell env wins (R5). R6 red-first history can't be verified from a diff, only that tests exist and pass.
>
> ## (b) Scope creep (minor)
>
> 1. **New env tunable `ALIGN_FAIL_RCS_RECOVERY_WINDOW_SEC`** (rcs_recovery.py:15). The spec only fixed the broken branch — "그 분기는 코드로는 존재하나... 켜면 곧바로 깨지는 결함 2개를 갖고 있었고" — and never asked for a configurable window timeout. Harmless but unrequested.
> 2. **`RecoveryOutcome.launched` field** (rcs_recovery.py:33) is populated and asserted in tests but never consumed by `cycle`. Not asked for by any requirement line.
>
> ## (c) Implemented but looks wrong
>
> 1. **Unknown process state still attempts login** — spec: "'알 수 없음' 이면 실행을 보류한다" (R2). The code holds the *launch* but still runs `login_fn` (rcs_recovery.py:59-61 region). Defensible (login will fail honestly if RCS is truly dead, as the test docstring argues), but it means an office machine with a broken psutil gets a login workflow fired at a possibly-nonexistent dialog every cycle. Borderline-compliant reading of "실행을 보류한다".
> 2. **Module-import-time env read** — `RECOVERY_WINDOW_TIMEOUT_SEC = env_float(...)` (rcs_recovery.py:15) is evaluated at import, so runtime env changes after import are ignored. Same pattern exists elsewhere in the repo, so likely acceptable, but worth noting since the spec stresses rollback switches behaving predictably ("롤백 스위치는 `ALIGN_FAIL_RCS_RECOVERY=0` 이며 셸 env 가 항상 이겨야 한다" — that specific flag is read correctly at settings-load time; only the new window timeout has this rigidity).
>
> No incorrect behavior found in the R3 empty-string contract, R4 failure-class wiring, or R5 defaults.

## 인용 검증 (Claude)

지적을 적용하기 전에 인용된 근거를 전부 열어 확인했다.

| 인용 | 판정 |
|---|---|
| `poc/workflow_3/README.md:213` 의 기본값 `0` 표기 | **사실**. 문자 그대로 일치 |
| `AGENTS.md` "Update docs together with code ... safety behavior" | **사실**. `AGENTS.md:94` 에 verbatim 존재. 이번 세션에서 Claude 는 AGENTS.md 를 읽은 적이 없다 |
| 모듈 상수가 `seed_env()` 보다 먼저 읽힌다 | **사실**. `align_fail_monitor.py:644` 에서 `seed_env` 를 `__main__` 안에서 import·호출하므로 모듈 import 시점보다 늦다. 셸 env 는 먹고(`=99` 확인) `workflow_3_config.py` 는 못 먹는다 |
| `build_cycle_steps("MCD916")[0]` 가 index 가정 | **사실**. `['ensure_rcs_ready', 'close_alert_popup', ...]` |

## 조치

| 지적 | 조치 |
|---|---|
| Standards 1 (HARD) | 적용. README env 표 기본값 `0`→`1`, 동작 요약 추가 |
| Standards 2 + Spec (b)1 + Spec (c)2 | 적용(3건 동일 대상). 모듈 상수를 없애고 `Workflow3Settings.rcs_recovery_window_timeout_sec` 로 이동, README 표에 등재 |
| Standards 3 | 적용. `_ensure_step()` 을 step_id 조회로 변경 |
| Spec (b)2 | 적용. `cycle` 이 `recovery.launched` 를 로그로 소비 |
| Spec (c)1 | **미적용(의도)**. 로그인 워크플로의 `ensure_login_window` success_criteria 와 타이핑 step 의 preconditions 가 모두 `WINDOW_VISIBLE(login title)` 이라, 다이얼로그가 없으면 자격증명을 타이핑하기 전에 중단된다. 게다가 psutil 을 이제 선언했으므로 "조회 불가" 자체가 예외 상황이다 |

조치 후 `uv run pytest poc/workflow_3/monitor poc/workflow_3/rcs` → **277 passed**.

## 두 축이 모두 놓친 것 (Claude)

- **복구 경로는 실장비에서 한 번도 실행된 적이 없다.** 12개 테스트는 판정 로직만
  덮고, pywinauto 가 실제 로그인 다이얼로그를 상대하는 동작은 덮지 않는다. 기본값을
  켠 채 첫 오피스 실행은 사람이 지켜봐야 한다.
- **`align/diagnostics/test_verify_consensus_path.py` 2건 실패는 이 변경 이전부터**
  (`git stash` 로 확인). `_FakeCond` 픽스처에 `.pixel` 이 없다 — 손으로 적은
  consumer-shape 픽스처가 producer 와 어긋난 기존 패턴이다.
