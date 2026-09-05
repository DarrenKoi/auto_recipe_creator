# oc-* project overlay

Written 2026-09-03. Every entry below traces to `CLAUDE.md`, a doc it names, or
something observed in that session's run; nothing here is inferred from a skim
of the tree. Items that could not be sourced are under **Unverified** at the end.


## Standards sources
1. `CLAUDE.md` (root) — the authoritative one. Code Conventions, the entrypoint
   constants-block rule (2026-08-31), and the per-subsystem contract sets.
2. The subsystem's own README/docs: `poc/workflow_3/README.md`,
   `deploy_vlms/UPLOAD.md`, `poc/workflow_4/docs/study/adr/`.
3. `AGENTS.md` if the diff touches a directory it covers.

`CLAUDE.md` is in Korean. Do not treat "I could not read the standards" as
"there are no standards" — quote the Korean rule verbatim in the finding.

## Escalation surfaces
Always `heavy`, regardless of diff size, because **the local test suite passes
either way** — these paths only fail on the office Windows PC in front of real
RCS hardware, which no test here can reach:

- `poc/workflow_3/rcs/**`, `poc/workflow_3/sem_monitor/**` — drive a live tool
  window via pywinauto/pynput. A wrong click lands on someone's running
  equipment; Mac has no way to observe it.
- `poc/workflow_3/monitor/cycle.py`, `align_fail_monitor*.py` — the per-alarm
  loop. Teardown is guaranteed by `try/finally`, not by steps; a change that
  moves cleanup into a step silently drops it on the failure path.
- Anything reading `office_*` adapters — those modules are **gitignored and
  exist only on the office PC**, so a call-site signature change type-errors
  there and nowhere here. This has silently disabled a feature for two months
  before (`include_msr`).
- Correction/OK-click gates (`ALIGN_FAIL_OK_CLICK`, `SAFE_MODE`,
  `*_DRY_RUN`, `CORRECT_WHEN_OCCUPIED`) — a gate that defaults the wrong way
  actuates hardware.
- `align/matching/**` constants — must stay bit-parity with
  `poc/workflow_2/registration_lab.py`; the bench measures against them.

## Extra smells
- **Silent env-name drift** — a constant block naming an env var no reader
  actually reads. → cross-check against the reader; a typo does nothing, loudly.
- **Gate that is only cosmetic** — a new guard whose boundary value is already
  filtered by an equal threshold downstream. → trace the boundary input all the
  way to the consumer, not just to the edited line.
- **eqp_id re-added to a consensus path** — `_events_dir_for` deliberately omits
  it. → any `<eqp_id>` in a consensus cache path is a defect, not a refinement.
- **Em-dash (U+2014) inside `print()`** — the office console is cp949 and
  crashes on it. → ASCII hyphen in `print`; docstrings are fine.
- **`logging` module in workflow code** — house style is `[INFO]`/`[ERROR]`
  prints. → only `logger.py` may use `logging`.
- **New `argparse`/CLI flag** — banned. → a constant at the top of the entrypoint.
- **English docstring** — Korean docstrings throughout, except the
  `flask_api/vlm_serve/*.py` route stubs. → translate, or match the siblings.
- **A fallback that cannot succeed** — a retry aimed at a provider/model/path
  that is known-unreachable. → it converts one failure into two; delete it.

## Spec source
In order: an issue under `docs/issues/<slug>/` referenced by the commit
message; a spec under `poc/workflow_*/docs/superpowers/specs/`; then ask.
Specs outside the repo must be copied in first — opencode cannot read past the
working tree.

## Constraints an outsider cannot infer
- **Development is split Mac + office Windows.** Claude runs on Mac and cannot
  see, run, or screenshot RCS. Advice of the form "just run it and check" is
  off the table for anything in `rcs/`, `sem_monitor/`, or the monitor loop.
- **Fab images cannot leave the office.** No sample screenshots will ever be
  available on this machine; feedback arrives as pasted console text.
- **There is no Claude at the office** — only local models. Anything requiring
  an agent to fix it on-site is not a plan.
- **The working tree is shared with other concurrent sessions.** `git add -A`
  is banned; stage by pathspec. Line numbers can move mid-review.
- **Commit straight to `main`**; no feature branches.

## Logging
- Destination: `docs/opencode/YYYY-MM-DD-<title>.md`
- Language: Korean prose, technical terms in English (matches the repo's docs)
- Lint after writing: none

## Verify commands
From the repo root:
- `uv run pytest poc/workflow_3/monitor poc/workflow_3/recording_filter`
- `uv run pytest poc/workflow_4/` (49)
- `uv run pytest flask_api/model_upload deploy_vlms/scripts` (50)
- Engine smoke: `uv run python poc/workflow_3/align/matching/test_engine.py`

None of these touch RCS. A green run does **not** clear an escalation surface.

Transcribed from `CLAUDE.md`'s Testing section, not executed while writing this
file — confirm the counts still hold before quoting them in a report.

## Commit rules
- Stage by explicit pathspec only — never `git add -A` / `commit -a`.
- Verify scope with `git show --stat` before pushing.
- Commit directly to `main`; Mac pushes, the office pulls.
- Never force-push, and never push gitignored files.

## Unverified
- The escalation list is the set of surfaces `CLAUDE.md` describes as
  office-only or hardware-actuating. It has not been validated against a real
  missed finding, so treat it as a starting list to extend, not a closed one.
- No `Spec source` convention is documented for commits that reference an issue
  by number rather than by path; the ordering above is the skill's default
  fallback, not an observed repo habit.
