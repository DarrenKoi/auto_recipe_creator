# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root. Read it before changing align-fail, recipe, or asset-path behavior (per `AGENTS.md`).
- **Per-workflow ADRs** in `poc/workflow_*/docs/study/adr/` — read ADRs that touch the area you're about to work in. There is no root `docs/adr/`; new ADRs go in the owning workflow's folder. `poc/workflow_2/docs/` may additionally hold runbooks, handoffs, and generated status artifacts.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/
│   ├── agents/                           ← this skill's config
│   └── issues/                           ← local-markdown issue tracker
└── poc/
    └── workflow_*/docs/study/adr/        ← workflow-scoped decisions (1, 2, 4 today)
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. The glossary is Korean-first; preserve Korean terms and established English technical identifiers (file paths, commands, APIs, model names) as written.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_