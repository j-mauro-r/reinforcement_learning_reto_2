# Development workflow

## Git workflow

1. Start from updated `main`.
2. Create a dedicated branch.
3. Create/approve a DWP plan for substantive work.
4. Implement one bounded task at a time.
5. Run the validation gate for that task.
6. Update durable documentation when a lasting decision changed.
7. Open a Pull Request; do not push implementation directly to `main`.

## DWP bootstrap

Run once per machine and again when intentionally upgrading the pinned DWP version:

```bash
bash scripts/bootstrap-dwp.sh
```

This installs DWP v4.0.0 for GitHub Copilot and Codex, then creates `.dwp/plans/` and `tmp/` locally.

## Validation hierarchy

For documentation/harness work:

```bash
bash -n scripts/bootstrap-dwp.sh scripts/verify-dwp-harness.sh
git diff --check
bash scripts/verify-dwp-harness.sh
```

For model work, perform the staged gates defined in `AGENTS.md` and `lineamientos-transversales.md`. Never jump directly to a long GPU training run before smoke validation.

## Reproducibility

Every final experiment must identify at least:

- environment/version;
- algorithm;
- seed;
- relevant hyperparameters;
- total timesteps/episodes;
- library versions;
- hardware/device;
- training duration;
- final model path;
- evaluation results.

Evaluation seeds must be controlled and separate from training seeds.
