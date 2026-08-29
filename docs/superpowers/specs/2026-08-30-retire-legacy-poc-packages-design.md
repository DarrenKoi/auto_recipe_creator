# Retire Legacy POC Packages

Date: 2026-08-30

## Goal

Remove `poc/work2`, `poc/workflow_1`, and `poc/workflow_2` now that
`poc/workflow_3` is the active implementation, without losing local-only
artifacts or leaving active runtime dependencies on the retired packages.

## Recovery

Before deletion, create the ignored local archive
`.scratch/retired-poc-2026-08-30.tar.gz` containing all three directories.
Verify the archive by listing it and writing a SHA-256 checksum beside it.
The archive may contain `.env` values, so it must remain ignored and local.

Tracked files remain recoverable from the dedicated Git deletion commit and
its parent. The archive covers ignored files that Git cannot restore.

## Changes

- Delete `poc/work2`, `poc/workflow_1`, and `poc/workflow_2`.
- Delete the eight external tests whose only subject is `poc.work2`.
- Do not port runtime modules: repository-wide import search confirms
  `poc.workflow_3` does not import the retired packages.
- Keep `poc/workflow_3/align_images` as the default asset root; it is already
  the current default in `poc/workflow_3/__init__.py`.
- Update current repository contracts and active workflow_3 guidance to name
  workflow_3 as the canonical implementation and remove live instructions
  that require deleted paths.
- Remove obsolete runtime diagnostics or messages that probe deleted legacy
  roots.
- Leave historical plans and reports unchanged when their old paths describe
  past work rather than current instructions.

## Verification

- Search Python sources outside historical documentation for imports of
  `poc.work2`, `poc.workflow_1`, and `poc.workflow_2`; expect none.
- Collect and run focused workflow_3 tests that cover imports, asset paths,
  monitoring, alignment, and SEM Monitor logic and are runnable off-Windows.
- Run `git diff --check` and inspect the final deletion scope.
- Commit the cleanup as one logical change without staging unrelated files.

## Recovery Procedure

- Tracked source/docs: restore from the cleanup commit's parent with
  `git restore --source=<parent> -- poc/work2 poc/workflow_1 poc/workflow_2`.
- Local-only artifacts: extract
  `.scratch/retired-poc-2026-08-30.tar.gz` from the repository root.
