# Repository agent kit

This directory contains repository-local agent commands and documentation. It intentionally does **not** vendor the Deep Work Plan engine: the engine is installed once per developer/agent environment and this repository keeps only thin delegators plus project-specific context.

Pinned engine release: `DailybotHQ/deepworkplan-skill@v4.0.0`.

Use `bash scripts/bootstrap-dwp.sh` to install/upgrade the pinned release for GitHub Copilot and Codex on the current machine and to create the local `.dwp/plans/` workspace.

See `.agents/docs/catalog.md` for available repository aliases.
