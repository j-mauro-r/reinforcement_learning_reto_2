# HandReachDense — PPO

[HandReachDense.ipynb](HandReachDense.ipynb) desarrolla capítulos 1–5. Capítulo 6 y ejecución real en Colab quedan pendientes de la entrega académica. El target de esta ejecución es local.

## Entorno local

Python 3.12, en venv separado. Desde la raíz:

```bash
uv venv 3-HandReachDense/.venv --python 3.12
uv pip install --python 3-HandReachDense/.venv/bin/python gymnasium-robotics==1.4.2 gymnasium==1.3.0 mujoco==3.2.7 stable-baselines3==2.9.0 torch==2.14.0 numpy==2.5.3 pandas==3.0.5 matplotlib==3.11.2 pillow==12.3.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 ipykernel nbformat nbclient
```

Ejecutar el notebook en orden con ese intérprete. Solo la celda Colab instala paquetes desde el notebook; los pins tienen evidencia local y esperan validación real en Colab. Configuración y rutas se centralizan en el capítulo 1.

## Contrato

`HandReachDense-v3`: Dict `(63,15,15)`, 20 acciones, 24 qpos/qvel, control absoluto (`relative_control=False`). Reward = negativo de la norma global de los 15 errores cartesianos. Éxito: norma global estrictamente menor que 0.01 m; la métrica principal por episodio es `success_final`. Horizonte 50, 20 substeps, timestep 0.002 y 25 Hz.

El probe valida imports, make/reset/step/render/close, shapes, física, reward/success, horizonte y objetivos de pulgar+dedo o configuración inicial; guarda `results/runtime_contract.json`. No se modifica XML, control, reward ni distribución de objetivos. Antes de full se requieren smoke, recarga, evaluación y MP4.

## Agente y seeds

PPO + MultiInputPolicy conserva el Dict, con actor/value `[256,256]`, ReLU, learning rate 3e-4, rollout 2048, batch 256, 10 epochs y clipping 0.2. La exploración es la política estocástica de PPO, sin action noise externo. Se elige CPU con un thread. La configuración ejecutable es `CONFIG` del notebook.

Entrenamiento 42; baseline aleatorio 101–120 (20 episodios); tuning/selección 201–210; finales reservadas 1001–1010; diagnóstico runtime 5001–5100. `success_final` se fija antes de entrenar. Se conserva la estrategia global SAC/TD3/PPO y no se añade normalización, shaping ni extractores personalizados.

## Pipeline y validación

`train_model(profile)` comparte código smoke/full; smoke solicita 8192 pasos y full 1.000.000. PPO completa rollouts de 2048: el total real puede superar ligeramente el solicitado. Se registran ambos y las épocas de optimización (el contador `_n_updates` de PPO cuenta épocas, no minibatches).

Cada run guarda Monitor, métricas PPO (KL aproximado, clip fraction, entropy/value loss y std), configuración, cronómetro y candidato. Evaluación cada 50k, checkpoints cada 100k y al terminar. Checkpoints periódicos ocurren durante rollout y son para recuperación aproximada; no prometen reproducir una ejecución ininterrumpida bit a bit. Los candidatos permanecen en `runs/` ignorado. El ZIP canónico se reserva hasta Task 7.

Gates previos a full: runtime, smoke, pesos finitos/actualizados, recarga en entorno nuevo, evaluación sin cambios de pesos y MP4 con overlay. `git diff --check` valida whitespace.

## Estado de ejecución: bloqueado en selección

Tasks 1–6 completados; Task 7 pendiente tras baseline y los dos ajustes permitidos. Los tres runs terminaron 1.001.472 pasos y pasaron persistencia/evaluación técnica, pero todos obtuvieron **0/10 success_final** en las mismas seeds de selección 201–210.

| Experimento | Cambio | Tiempo | Retorno medio ± DE | Distancia global final |
| --- | --- | --- | --- | --- |
| Baseline | configuración original | 767,22 s | −3,766 ± 0,398 | 74,9 mm |
| Ajuste 1 | LR 3e-5; clipping 0.1 | 776,04 s | −3,506 ± 0,346 | 70,1 mm |
| Ajuste 2 | ajuste 1 + log_std_init −1 | 777,75 s | −3,404 ± 0,328 | 67,5 mm |

Son resultados de **selección**, no evaluación final. La tarea requiere norma global menor que 10 mm y el plan pide al menos 80% de éxito para promover candidato. No existe modelo canónico ni videos finales; capítulos 4–5 quedan pendientes. Los MP4 smoke verifican únicamente el pipeline de render.

`results/experiments.csv`, `selection_status.json` y `experiment_*` preservan comparaciones y diagnósticos. Cada candidato local se carga con `PPO.load(MODEL_ROOT / "runs" / run_id / "candidate.zip", device="cpu")`. Los checkpoints siguen ignorados; no son entrega final. Cargar solo archivos propios/confiables.

El último ajuste se estancó cerca de 0.067–0.068 m, sin evidencia suficiente para duplicar su presupuesto. Se necesita refinar Task 7 antes de más corridas o técnicas como normalización de observaciones. Mantener PPO, la separación de seeds y el criterio físico original. Una causa posible a investigar es la escala heterogénea de posiciones, velocidades y goals; estos experimentos no demuestran por sí solos esa causalidad.

El notebook sigue siendo un trabajo en curso: `full` reproduce las tres corridas actuales, con coste total medido de aproximadamente 38,7 minutos de entrenamiento; `smoke` ejecuta solo los gates cortos. Seeds finales 1001–1010 intactas. Colab y capítulo 6 siguen pendientes.
