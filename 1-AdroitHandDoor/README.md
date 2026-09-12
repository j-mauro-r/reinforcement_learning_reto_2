# AdroitHandDoor — SAC

Desarrollo incremental en `AdroitHandDoor.ipynb`. El DWP actual cubre capítulos 1–5; el informe técnico del capítulo 6 está pendiente para una entrega posterior.

## Ejecución local

Desde la raíz del repositorio, con Python 3.12 y un entorno virtual aislado:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install gymnasium-robotics==1.3.0 gymnasium==1.0.0 mujoco==3.1.6 stable-baselines3==2.4.1 torch==2.5.1 numpy==1.26.4 ipykernel nbformat nbclient imageio imageio-ffmpeg pillow
```

Seleccionar `.venv` como kernel del notebook y ejecutar desde el principio. La celda `DEPENDENCIES` del notebook es la matriz autoritativa; el comando reproduce esa matriz. El notebook no instala paquetes en local. En macOS se usa el render nativo de MuJoCo, sin Xvfb. La configuración usa CPU y un thread de PyTorch para evitar sobrecostos en las redes pequeñas de SAC.

## Google Colab

Abrir el notebook en un runtime limpio y ejecutar desde la primera celda. La instalación ocurre en una única celda y utiliza EGL para el render. Las rutas se derivan de una sola raíz y el pipeline es el mismo. **Validación Colab pendiente**: las versiones indicadas solo han sido probadas localmente y todavía no constituyen pins finales validados en Colab.

## Contrato y validación

El capítulo 1 verifica instalación/imports, observación `(39,)`, acción `(28,)`, recompensa reconstruida paso a paso, 200 pasos con truncation, render RGB y cierre del entorno. Guarda `results/runtime_contract.json`, incluyendo versiones, hardware, rangos de articulación y de control.

`AdroitHandDoor-v1` conserva el término histórico `+0.1 * distancia(palma, manija)`. No se altera la recompensa. Los límites de articulación y los rangos de control son distintos: por ejemplo, `THJ4` tiene límite articular ±1.047 y control ±1.0. Consultar la ficha técnica para la distinción.

Smoke local validado: 5000 pasos, 4500 actualizaciones; modelo recargado y evaluado durante dos episodios. Entrenamiento full y evaluación final pendientes.

## Agente baseline

SAC `MlpPolicy` con red `[256, 256]` ReLU; learning rate `3e-4`, gamma `0.99`, tau `0.005`, batch `256`, replay buffer `500000`, una actualización por paso y entropía automática. Se usa por control continuo y exploración estocástica, como hipótesis a contrastar frente al baseline aleatorio; TD3 y PPO quedan propuestos para los otros dos modelos. Sin normalización, HER, shaping ni curriculum.

Perfiles: `smoke` = 5000 pasos, `learning_starts=500`; `full` = 500000 pasos, `learning_starts=10000`. Evaluación de seguimiento cada 25000; checkpoints cada 50000. Seeds: entrenamiento 42, caracterización 101–110, tuning 201–205, video 301, evaluación final 1001–1010. Selección: éxito, retorno medio, variabilidad y tiempo. El objetivo de 80% es interno.

## Entrenamiento y recuperación

Cambiar únicamente `CONFIG['profile']` a `full` y ejecutar las celdas en orden para el baseline de 500000 pasos. `run_training` comparte el flujo entre ambos perfiles. Cada ejecución tiene un `run_id` distinto y conserva `training.monitor.csv`, `monitoring.csv`, candidato y `metadata.json` (configuración, versiones, dispositivo, tiempo y checksum). El smoke permanece bajo `tmp/adroit_smoke/`; los candidatos full y checkpoints quedan en `runs/`, ignorado por Git.

El candidato todavía no se promueve al nombre canónico. Los checkpoints intermedios incluyen run_id y timestep. `latest_replay_buffer.pkl` corresponde al checkpoint más reciente del mismo directorio; permite continuar aprendizaje junto con ese ZIP usando `SAC.load`, `load_replay_buffer` y `learn(..., reset_num_timesteps=False)`. Una recuperación debe identificarse y documentarse como tal; no se afirma reproducción bit a bit de un proceso interrumpido.

Los candidatos/checkpoints conservan el estado estándar de entrenamiento de SB3. La promoción final preparará una exportación de política sin optimizer ni replay buffer y verificará su recarga e igualdad de acciones. Los MP4 de smoke comprueban el render y overlay; no son los videos finales de entrega.

## Modelo seleccionado

La corrida `20260911T231624899082Z_full` completó 500000 pasos en 2820.04 segundos (47.0 minutos). La selección compara checkpoints guardados con las mismas cinco seeds de tuning. Se eligió el checkpoint de **450000 pasos**: retorno medio **1492.84 ± 887.73**, éxito **80%**. El candidato posterior a 500000 pasos obtuvo 40% al recargarlo. La curva es temporalmente inestable; no se afirma convergencia estable. Se omitieron corridas adicionales al disponer de un checkpoint que alcanza el objetivo interno. Las seeds finales no participaron en esta decisión.

Artefacto único: `models/adroit_hand_door_sac.zip`, con `models/adroit_hand_door_sac.metadata.json`. Es una exportación de inferencia sin replay buffer ni estados de optimizer. **Se carga con `InferenceSAC.load`**, usando la pequeña subclase definida en el notebook; `SAC.load` espera estados de entrenamiento que esta exportación excluye. Las acciones y los resultados de los cinco episodios se verificaron idénticos frente al checkpoint original. Para reanudar entrenamiento se usan los checkpoints estándar de `runs/`, no el artefacto canónico.

`results/checkpoint_selection.csv` conserva la comparación, `results/experiments.csv` registra la única corrida y `results/baseline_*` preserva logs y metadatos. Los 500000 pasos y el tiempo completo describen el costo de la corrida; los 450000 pasos identifican exactamente los pesos entregados.

## Evaluación final

Sobre 10 episodios reservados (seeds 1001–1010), el artefacto seleccionado obtuvo **1704.81 ± 1202.73** de retorno y **70% de éxito (7/10)**. La política aleatoria obtuvo **−45.84 ± 0.84** y **0% de éxito**, con iguales seeds y horizonte. Las desviaciones son poblacionales (`ddof=0`), no intervalos de confianza. Estos resultados no se utilizaron para volver a seleccionar o ajustar el modelo.

Las tres figuras están en `results/figures/` y visibles en el notebook. `results/evaluation.csv`, `results/random_baseline_final.csv` y `results/metrics.json` permiten comprobar las métricas. La entrega muestra una mejora clara, con fallos en tres episodios; no se declara solución infalible ni estabilidad temporal del entrenamiento. El informe académico del capítulo 6 sigue pendiente.
