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

Antes del entrenamiento full se requieren smoke de 5.000 pasos, save/reload, evaluación y MP4. El notebook ejecuta esos gates antes de habilitar full. Validación adicional: `git diff --check`.

## Agente y evaluación

TD3 usa dos críticos, actor retrasado (`policy_delay=2`) y ruido objetivo 0.2, recortado a 0.5. MLP `[400, 300]` con ReLU; exploración gaussiana sigma 0.1. Learning rate 0.001, gamma 0.99, tau 0.005, batch 256, buffer 200.000. El perfil smoke tiene 5.000 pasos y warmup 500; full 100.000 y warmup 1.000. La configuración ejecutable vive solo en el notebook.

Seeds: entrenamiento 42, caracterización aleatoria 101–110, seguimiento 201–205, selección 201–210, evaluación final reservada 1001–1010. Selección por éxito al final del episodio, luego retorno, permanencia, variabilidad y tiempo. Fuente de implementación: [SB3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html).

## Entrenamiento y persistencia

`train_model(profile)` es el único pipeline para smoke/full. Cada corrida guarda configuración, versiones, hardware, cronómetro, Monitor CSV, evaluaciones de tuning, `candidate.zip` y checkpoints por timestep bajo `runs/<run_id>/` (ignorado). Un smoke comprueba pesos actualizados, save/reload, evaluación sin aprendizaje y video MP4 antes de habilitar full. Evaluación cada 5.000 pasos; checkpoint cada 10.000 y al terminar; replay más reciente conservado para recuperación. Los checkpoints periódicos son capturados durante rollout; reiniciar entrenamiento no promete equivalencia bit a bit con una corrida ininterrumpida.

El artefacto canónico se reserva para la selección final; los candidatos no se presentan como modelo entregable. El video usa 25 FPS, coherente con el tiempo físico de Fetch (50 pasos = 2 segundos).

## Modelo seleccionado

Baseline de 100.000 pasos, run `20260912T005116707634Z_full`, Apple M2 Pro / 32 GiB, CPU (un thread). Tiempo de aprendizaje con callbacks: 345,82 s. Las diez seeds de selección (201–210) dieron 100% success_final, retorno −0,847 ± 0,151 (desviación muestral), permanencia 96,8% y distancia final media 14,2 mm. No hicieron falta corridas adicionales. Estas métricas son de selección, no la evaluación final.

`models/fetch_reach_td3.zip` es el único modelo canónico. Cargar con `TD3.load(path, device="cpu")` y usar `predict(obs, deterministic=True)`. Conserva todos los pesos del candidato; los estados aprendidos de los optimizadores y el replay no se exportan. Los optimizadores vacíos permiten usar el cargador estándar. Para continuar una corrida, usar el checkpoint y replay de `runs/`.

La metadata adyacente registra procedencia, configuración, selección y SHA-256. Cargar solo artefactos propios/confiables; la serialización SB3 incluye metadatos Python. Un checksum verifica integridad, no autenticidad del origen.

## Evidencia final local

| Métrica principal | Resultado |
| --- | --- |
| Tiempo full (incluye callbacks; excluye smoke y evaluación final) | 345,82 s / 5,76 min |
| Retorno, 10 episodios reservados | −0,803 ± 0,195 (DE muestral) |
| Tasa de éxito `success_final` | 100% (10/10) |

Random: −9,532 ± 3,179 y 0/10 éxitos en las mismas seeds. Distancia final media del agente: 12,6 mm; permanencia dentro del umbral: 96,6%. Resultados de una sola seed de entrenamiento y diez episodios; no prueban robustez frente a todas las inicializaciones.

- [Evaluación por episodio](results/evaluation.csv) y [métricas](results/metrics.json).
- Tres figuras en `results/figures/`, también visibles en el notebook.
- [Training Process](videos/training_process.mp4): checkpoint real de 50.000 pasos.
- [Trained Agent](videos/trained.mp4): artefacto canónico de 100.000 pasos.
- Ambos videos: 50 frames / 25 FPS / 2 s, títulos y reward visibles. JSON adyacentes vinculan source/checksum.

## Validación reproducible

`CONFIG["profile"] = "smoke"` ejecuta gates de entorno, 5.000 pasos, persistencia, evaluación y MP4; no promueve ni evalúa el artefacto final. El notebook entregado usa `full`; **Run all vuelve a entrenar** y regenera la entrega si el candidato satisface el criterio de selección. Las salidas actuales fueron producidas durante la ejecución por tareas del mismo notebook, con las corridas identificadas en sus metadatos.

Para comprobar el smoke en un kernel nuevo y un directorio vacío sin sobrescribir la entrega, desde la raíz y con el entorno activo:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
notebook = nbformat.read(root / "2-FetchReachDense/FetchReachDense.ipynb", as_version=4)
for cell in notebook.cells:
    if "config" in cell.metadata.get("tags", []):
        cell.source = cell.source.replace('"profile": "full"', '"profile": "smoke"')
(root / "tmp").mkdir(exist_ok=True)
with TemporaryDirectory(dir=root / "tmp") as scratch:
    NotebookClient(notebook, timeout=1200, resources={"metadata": {"path": scratch}}).execute()
print("Clean smoke: PASS")
```

Gates finales adicionales: checksum/model reload, igualdad de pesos con candidato, coherencia de CSV/metadata y decodificación completa de MP4. Se usó ImageIO + FFmpeg porque `ffprobe` no estaba instalado. No hay suite universal de tests Python en el repositorio; las aserciones del notebook son los gates del modelo.

Pendientes de entrega académica: ejecutar en Colab limpio y redactar capítulo 6. El revisor automático local de DWP no está instalado; se realizó revisión manual del diff y artefactos.
