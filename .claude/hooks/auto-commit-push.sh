#!/bin/bash
# Auto commit & push hook — runs when Claude stops after code generation/editing.
# Only commits if there are actual changes in the working tree.

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Check if there are any changes (staged, unstaged, or untracked)
if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    # No changes — skip
    exit 0
fi

# Stage all changes
git add -A

# Generate commit message with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "auto: Claude Code changes (${TIMESTAMP})

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Push to current branch
git push 2>/dev/null || git push --set-upstream origin "$(git branch --show-current)" 2>/dev/null
