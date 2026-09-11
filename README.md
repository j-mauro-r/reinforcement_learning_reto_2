# Reinforcement Learning — Reto II

Repositorio del Reto II de Reinforcement Learning sobre **Gymnasium Robotics**.

## Problemas

- `AdroitHandDoor-v1`
- `FetchReachDense-v4`
- `HandReachDense-v3`

Los requisitos académicos están en `enunciado-reto.md` y las reglas obligatorias de ingeniería/entrega en `lineamientos-transversales.md`.

## Deep Work Plan

El repositorio está preparado para **Deep Work Plan v4.0.0**, con flujo Lite-first, integración para GitHub Copilot/Codex, `AGENTS.md`, aliases en `.agents/`, documentación del harness y workspace `.dwp/` local ignorado por Git.

Después de clonar el repositorio:

```bash
bash scripts/bootstrap-dwp.sh
```

Validar configuración:

```bash
bash scripts/verify-dwp-harness.sh
```

Comandos principales en agentes compatibles:

```text
#dwp-create <objetivo>
#dwp-execute
#dwp-status
#dwp-refine
#dwp-resume
#dwp-verify
```

Ver `docs/dwp.md` y `AGENTS.md` para el flujo completo.
