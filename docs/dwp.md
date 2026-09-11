# Deep Work Plan in this repository

## Version policy

The repository pins the **Deep Work Plan skill v4.0.0**. The release changed plan creation to a Lite-first lifecycle and removed the previous refined-draft workflow. `.dwp/drafts/` is therefore not used in this repository.

The DWP engine is installed per developer/agent environment; it is not vendored into this repository. Repository-local `.agents/commands/` files are thin aliases so lifecycle behavior remains owned by the installed skill.

## Install / upgrade

```bash
bash scripts/bootstrap-dwp.sh
```

The bootstrap script:

1. checks required tools;
2. clones exactly `DailybotHQ/deepworkplan-skill@v4.0.0` into a temporary directory;
3. runs the upstream installer for GitHub Copilot;
4. installs for Codex as well when Codex is available/configured;
5. creates `.dwp/plans/` and `tmp/` locally;
6. runs the repository harness verification.

## Commands

Use these repository aliases in supported coding agents:

```text
#dwp-create <goal>
#dwp-execute
#dwp-status
#dwp-refine
#dwp-resume
#dwp-verify
```

## Lite-first lifecycle

- A new plan is represented as **Lite** when compact task contracts are sufficient.
- A plan becomes **Full** only when detailed per-task files are needed.
- Both representations must remain executable and verifiable.
- Scope, acceptance criteria and validation gates are mandatory regardless of representation.
- Creation/promotion never executes product work.
- Guided plans require approval before execution; trust/auto behavior is used only when explicitly selected.

## Operational workspace

DWP writes under:

```text
.dwp/
└── plans/
    └── PLAN_<slug>/
```

The directory is gitignored by design. Do not commit plan state. If a decision must survive after the plan, write it to tracked repository documentation.

## Upgrades

Do not silently float to the latest DWP release. Upgrade the pin deliberately in a dedicated PR after reviewing release notes and checking compatibility with `lineamientos-transversales.md`, `AGENTS.md`, aliases and bootstrap behavior.
