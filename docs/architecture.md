# Architecture

## Repository archetype

This repository is an **individual academic repository**, not an orchestration hub. It contains three independent RL problem implementations sharing transversal engineering rules.

## Intended top-level layout

```text
.
├── AGENTS.md
├── enunciado-reto.md
├── lineamientos-transversales.md
├── Gymnasium_Robotics.ipynb
├── 1-AdroitHandDoor/
├── 2-FetchReachDense/
├── 3-HandReachDense/
├── docs/
├── .agents/
├── .dwp/              # local only, gitignored
└── tmp/               # local only, gitignored
```

The three model folders are independent execution/delivery boundaries. Shared rules belong at repository root; model-specific configuration and artifacts belong inside the corresponding model folder.

## Model boundaries

Each model should eventually contain only what it needs to execute and deliver the assignment, typically:

```text
<model>/
├── README.md
├── <model>.ipynb
├── config/ or config.yaml
├── src/                # only when notebook clarity justifies extraction
├── models/
├── videos/
└── results/
```

Do not create empty architecture for its own sake. A directory is added when the implementation needs it.

## Design principles

- Notebook = narrative, orchestration, experiment evidence and report.
- Configuration = one source of hyperparameters, seeds and paths.
- Reusable code = small modules only when complexity/duplication warrants extraction.
- Model artifact = canonical final inference artifact; checkpoints are training recovery artifacts, not competing finals.
- DWP = operational control plane under `.dwp/`, never a permanent substitute for tracked documentation.
