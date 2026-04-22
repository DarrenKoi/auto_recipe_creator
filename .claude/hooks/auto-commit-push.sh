#!/bin/bash
# Auto commit & push hook — runs when Claude stops after code generation/editing.
# Only commits if there are actual changes. Generates a descriptive commit
# message from the staged diff (file list + insertions/deletions).

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Skip if no changes (staged, unstaged, or untracked)
if git diff --quiet HEAD 2>/dev/null && \
   git diff --cached --quiet 2>/dev/null && \
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
    exit 0
fi

git add -A

NAMESTATUS=$(git diff --cached --name-status)
STAT=$(git diff --cached --stat)

# Tallies by change kind
ADDED=$(printf '%s\n' "$NAMESTATUS" | awk '$1=="A"' | wc -l | tr -d ' ')
MODIFIED=$(printf '%s\n' "$NAMESTATUS" | awk '$1=="M"' | wc -l | tr -d ' ')
DELETED=$(printf '%s\n' "$NAMESTATUS" | awk '$1=="D"' | wc -l | tr -d ' ')
TOTAL=$(printf '%s\n' "$NAMESTATUS" | awk 'NF>0' | wc -l | tr -d ' ')

# Pick verb (single-file = its kind; multi-file = dominant kind)
if [ "$TOTAL" = "1" ]; then
    KIND=$(printf '%s\n' "$NAMESTATUS" | awk 'NF>0 {print $1; exit}')
    case "$KIND" in
        A)   VERB="add"    ; VERB_KIND="A" ;;
        D)   VERB="delete" ; VERB_KIND="D" ;;
        M)   VERB="update" ; VERB_KIND="M" ;;
        R*)  VERB="rename" ; VERB_KIND="R" ;;
        C*)  VERB="copy"   ; VERB_KIND="C" ;;
        *)   VERB="change" ; VERB_KIND=""  ;;
    esac
else
    if [ "$ADDED" -gt "$MODIFIED" ] && [ "$ADDED" -gt "$DELETED" ]; then
        VERB="add"    ; VERB_KIND="A"
    elif [ "$DELETED" -gt "$MODIFIED" ] && [ "$DELETED" -gt "$ADDED" ]; then
        VERB="delete" ; VERB_KIND="D"
    else
        VERB="update" ; VERB_KIND="M"
    fi
fi

# Primary path: prefer a file that matches the chosen verb, then any non-deleted, then any
PRIMARY=""
if [ -n "$VERB_KIND" ]; then
    PRIMARY=$(printf '%s\n' "$NAMESTATUS" | awk -v k="$VERB_KIND" '$1==k && NF>0 {print $NF; exit}')
fi
if [ -z "$PRIMARY" ]; then
    PRIMARY=$(printf '%s\n' "$NAMESTATUS" | awk '$1!="D" && NF>0 {print $NF; exit}')
fi
if [ -z "$PRIMARY" ]; then
    PRIMARY=$(printf '%s\n' "$NAMESTATUS" | awk 'NF>0 {print $NF; exit}')
fi

if [ "$TOTAL" = "1" ]; then
    SUBJECT="auto: ${VERB} ${PRIMARY}"
else
    OTHERS=$((TOTAL - 1))
    SUBJECT="auto: ${VERB} ${PRIMARY} (+${OTHERS} more)"
fi

# Keep subject within ~72 chars for git log readability
if [ ${#SUBJECT} -gt 72 ]; then
    SUBJECT="${SUBJECT:0:69}..."
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

git commit -m "${SUBJECT}

${STAT}

Timestamp: ${TIMESTAMP}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Push to current branch
git push 2>/dev/null || git push --set-upstream origin "$(git branch --show-current)" 2>/dev/null
