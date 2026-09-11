# Troubleshooting

## `deepworkplan-*` skill is not found

Run:

```bash
bash scripts/bootstrap-dwp.sh
```

Then restart/reload the coding agent so it rescans its skills directory.

## `.dwp/` does not exist after cloning

This is expected: `.dwp/` is intentionally gitignored. Recreate it with:

```bash
bash scripts/bootstrap-dwp.sh
```

or minimally:

```bash
mkdir -p .dwp/plans tmp
```

## DWP plan still references `.dwp/drafts/`

That plan/process belongs to an older lifecycle. DWP v4.0.0 removed the refined-draft directory. Use `#dwp-refine`/migration behavior from the installed skill rather than manually rewriting execution state.

## Copilot ignores DWP aliases

Confirm `.github/copilot-instructions.md` exists and points to `AGENTS.md`, then use the explicit installed skill name (for example `#deepworkplan-create`) if the short alias is not surfaced by the UI.

## Codex does not detect DWP

Confirm the bootstrap installed under `$HOME/.codex/skills` and restart Codex. The repository also includes `.codex/instructions.md` pointing back to `AGENTS.md`.

## Harness verification fails

Run:

```bash
bash scripts/verify-dwp-harness.sh
```

Fix every reported missing file/symlink/ignore rule before using DWP for model work.
