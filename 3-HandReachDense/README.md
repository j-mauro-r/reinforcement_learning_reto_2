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
