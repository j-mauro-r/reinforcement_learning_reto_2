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

Aún no se ha ejecutado entrenamiento ni evaluación final.
