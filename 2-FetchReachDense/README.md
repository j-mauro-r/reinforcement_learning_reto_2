# FetchReachDense — TD3

Notebook principal: [FetchReachDense.ipynb](FetchReachDense.ipynb). Este desarrollo cubre capítulos 1–5; el informe del capítulo 6 y la ejecución real en Colab quedan pendientes de la entrega académica.

## Ejecución local

Python 3.12 en un entorno aislado; no reutilizar el entorno de Adroit. Desde la raíz:

```bash
uv venv 2-FetchReachDense/.venv --python 3.12
uv pip install --python 2-FetchReachDense/.venv/bin/python gymnasium-robotics==1.4.2 gymnasium==1.3.0 mujoco==3.2.7 stable-baselines3==2.9.0 torch==2.14.0 numpy==2.5.3 pandas==3.0.5 matplotlib==3.11.2 pillow==12.3.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 ipykernel nbclient nbformat
```

Abrir el notebook con ese intérprete y ejecutar en orden. `CONFIG` concentra perfiles, seeds, hiperparámetros y video; las rutas derivan de una raíz. La celda de instalación solo actúa en Colab. Allí usar un runtime limpio; los pins son candidatos comprobados localmente, no evidencia de Colab.

## Contrato y gates

`FetchReachDense-v4`, TD3 + `MultiInputPolicy`, Dict `(10,)+(3,)+(3,)`, acción `(4,)`, reward `-distance`, éxito principal `success_final`. Sin HER, shaping ni normalización añadida.

El capítulo 1 valida imports, shapes, reward/success, TimeLimit, goal sampling, gripper bloqueado y render. Guarda `results/runtime_contract.json`. MuJoCo 3.13.0 falló en `set_joint_qpos` durante creación; se eligió 3.2.7 tras el probe real, conservando 20 substeps, timestep 0.002 y control a 25 Hz.

Antes del entrenamiento full se requieren smoke de 5.000 pasos, save/reload, evaluación y MP4. Los capítulos siguientes incorporarán esos gates de forma secuencial. Validación adicional: `git diff --check`.

## Agente y evaluación

TD3 usa dos críticos, actor retrasado (`policy_delay=2`) y ruido objetivo 0.2, recortado a 0.5. MLP `[400, 300]` con ReLU; exploración gaussiana sigma 0.1. Learning rate 0.001, gamma 0.99, tau 0.005, batch 256, buffer 200.000. El perfil smoke tiene 5.000 pasos y warmup 500; full 100.000 y warmup 1.000. La configuración ejecutable vive solo en el notebook.

Seeds: entrenamiento 42, caracterización aleatoria 101–110, tuning 201–205, evaluación final reservada 1001–1010. Selección por éxito al final del episodio, luego retorno, permanencia, variabilidad y tiempo. Fuente de implementación: [SB3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html).
