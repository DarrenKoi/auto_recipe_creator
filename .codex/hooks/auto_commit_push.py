#!/usr/bin/env python3

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / ".codex" / "hooks" / "state"
STATE_FILE = STATE_DIR / "auto_commit_push_state.json"
LOG_FILE = STATE_DIR / "auto_commit_push.log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} {message}\n")


def run_git(args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=capture_output,
        text=True,
        check=False,
    )


def git_path_set(args: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", *args, "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr_text}")
    return {
        item.decode("utf-8", errors="replace")
        for item in result.stdout.split(b"\0")
        if item
    }


def current_dirty_paths() -> set[str]:
    return (
        git_path_set(["diff", "--name-only"])
        | git_path_set(["diff", "--cached", "--name-only"])
        | git_path_set(["ls-files", "--others", "--exclude-standard"])
    )


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log("[WARNING] State file is invalid JSON; resetting it.")
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_commit_message(payload: dict) -> tuple[str, str | None]:
    prompt = ""
    for item in payload.get("input-messages") or []:
        if isinstance(item, str) and item.strip():
            prompt = item
            break

    assistant = payload.get("last-assistant-message")
    if not isinstance(assistant, str):
        assistant = ""

    turn_id = payload.get("turn-id")
    if not isinstance(turn_id, str):
        turn_id = ""

    summary = prompt or assistant or "apply changes"
    subject = f"Codex: {compact_text(summary, 64)}"

    body_lines: list[str] = []
    if turn_id:
        body_lines.append(f"Turn: {turn_id}")
    if assistant:
        if body_lines:
            body_lines.append("")
        body_lines.append("Assistant summary:")
        body_lines.append(compact_text(assistant, 400))

    if not body_lines:
        return subject, None
    return subject, "\n".join(body_lines)


def main() -> int:
    payload_arg = sys.argv[1] if len(sys.argv) > 1 else "{}"

    try:
        payload = json.loads(payload_arg)
    except json.JSONDecodeError:
        log("[WARNING] Skipping auto-commit: notify payload was not valid JSON.")
        return 0

    if payload.get("type") != "agent-turn-complete":
        return 0

    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        log(f"[WARNING] Skipping auto-commit: {REPO_ROOT} is not a git repository root.")
        return 0

    if (
        (git_dir / "MERGE_HEAD").exists()
        or (git_dir / "rebase-merge").exists()
        or (git_dir / "rebase-apply").exists()
    ):
        log("[WARNING] Skipping auto-commit: merge or rebase is in progress.")
        return 0

    try:
        dirty_paths = current_dirty_paths()
        state = load_state()

        # The baseline shields preexisting dirty files from being swept into the
        # first auto-commit after the hook is enabled.
        if not state:
            save_state(
                {
                    "created_at": now_iso(),
                    "baseline_paths": sorted(dirty_paths),
                }
            )
            log(f"[INFO] Initialized baseline with {len(dirty_paths)} dirty path(s).")
            return 0

        baseline_paths = set(state.get("baseline_paths") or [])
        active_baseline = baseline_paths & dirty_paths

        if active_baseline != baseline_paths:
            state["baseline_paths"] = sorted(active_baseline)
            save_state(state)

        candidate_paths = sorted(dirty_paths - active_baseline)
        if not candidate_paths:
            return 0

        staged_paths = git_path_set(["diff", "--cached", "--name-only"])
        conflicting_staged = sorted(staged_paths - set(candidate_paths))
        if conflicting_staged:
            log(
                "[WARNING] Skipping auto-commit because staged changes already exist outside the hook candidate set: "
                + ", ".join(conflicting_staged)
            )
            return 0

        add_result = run_git(["add", "--all", "--", *candidate_paths])
        if add_result.returncode != 0:
            log(f"[ERROR] git add failed: {add_result.stderr.strip()}")
            return 0

        subject, body = build_commit_message(payload)
        commit_args = ["commit", "-m", subject]
        if body:
            commit_args.extend(["-m", body])

        commit_result = run_git(commit_args)
        if commit_result.returncode != 0:
            combined = "\n".join(
                part.strip() for part in [commit_result.stdout, commit_result.stderr] if part.strip()
            )
            if "nothing to commit" in combined.lower():
                log("[INFO] Skipping auto-commit: git reported nothing to commit after staging.")
                return 0
            log(f"[ERROR] git commit failed: {combined}")
            return 0

        branch_result = run_git(["branch", "--show-current"])
        branch_name = branch_result.stdout.strip()
        if branch_result.returncode != 0 or not branch_name:
            log("[WARNING] Commit succeeded, but push was skipped because the current branch could not be determined.")
            return 0

        origin_result = run_git(["remote", "get-url", "origin"])
        if origin_result.returncode != 0:
            log("[WARNING] Commit succeeded, but push was skipped because no origin remote is configured.")
            return 0

        push_result = run_git(["push", "origin", branch_name])
        if push_result.returncode != 0:
            combined = "\n".join(
                part.strip() for part in [push_result.stdout, push_result.stderr] if part.strip()
            )
            log(f"[ERROR] git push failed: {combined}")
            return 0

        log("[INFO] Auto-commit and push completed for paths: " + ", ".join(candidate_paths))
    except Exception as exc:
        log(f"[ERROR] Unexpected auto-commit hook failure: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
