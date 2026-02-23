---
name: commit-push
description: Commit all changes and push to remote
disable-model-invocation: true
allowed-tools: Bash(git *), Read, Glob, Grep
---

# Commit and Push

Commit all staged/unstaged changes and push to the current remote branch.

## Steps

1. Run `git status` (never use `-uall`), `git diff` (staged + unstaged), and `git log --oneline -5` in parallel to understand the changes and match existing commit style.
2. Analyze the diff and draft a concise imperative commit message (1-2 sentences) following the project's commit style: `Add ...`, `Fix ...`, `Update ...`, `Replace ...`, etc.
3. Stage the relevant changed files by name (prefer explicit filenames over `git add -A`). Never stage `.env`, credentials, or secret files.
4. Create the commit. Always append the co-author trailer:
   ```
   Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
   ```
   Use a HEREDOC for the message:
   ```bash
   git commit -m "$(cat <<'EOF'
   Commit message here

   Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
   EOF
   )"
   ```
5. Push to the remote (`git push`).
6. Report the commit hash and a brief summary to the user.

## Rules

- Do NOT skip pre-commit hooks (`--no-verify`).
- Do NOT amend previous commits.
- Do NOT force push.
- If there are no changes, tell the user — do not create an empty commit.
- If $ARGUMENTS is provided, use it as the commit message instead of drafting one.
