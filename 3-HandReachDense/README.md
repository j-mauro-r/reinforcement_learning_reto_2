# HandReachDense — PPO Vectorizado y Normalizado

Notebook principal: [HandReachDense.ipynb](HandReachDense.ipynb). Implementación autónoma para `HandReachDense-v3` empleando Proximal Policy Optimization (PPO) sobre entornos concurrentes y normalización vectorial de estados, preservando la diversidad algorítmica exigida por el reto (SAC en Adroit, TD3 en Fetch y PPO en HandReach).

## Entorno y Ejecución

Python 3.12 en entorno virtual aislado.

```bash
# Creación del entorno
py -3.12 -m venv 3-HandReachDense/.venv

# Instalación de dependencias fijadas
.\3-HandReachDense\.venv\Scripts\pip.exe install gymnasium-robotics==1.4.2 gymnasium==1.3.0 mujoco==3.2.7 stable-baselines3==2.9.0 torch==2.14.0 numpy==2.5.3 pandas==3.0.5 matplotlib==3.11.2 pillow==12.3.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 ipykernel nbformat nbclient
```

### Ejecución en Google Colab
El notebook detecta automáticamente el entorno Colab mediante `"google.colab" in sys.modules`, activa renderizado acelerado con EGL (`MUJOCO_GL="egl"`) e instala las versiones exactas especificadas en la matriz `DEPENDENCIES`.

## Contrato Físico del Entorno

- **Identificador:** `HandReachDense-v3`
- **Espacio de observación:** `Dict` con claves:
  - `observation`: vector continuo de dimensión `(63,)` (posiciones angulares $q_{pos}$, velocidades angulares $q_{vel}$ y coordenadas alcanzadas).
  - `achieved_goal`: vector continuo `(15,)` (posiciones tridimensionales de las 5 puntas de los dedos).
  - `desired_goal`: vector continuo `(15,)` (posiciones objetivo 3D para cada dedo).
- **Espacio de acción:** `Box(-1.0, 1.0, (20,), float32)` correspondiente a los 20 actuadores de las articulaciones de la mano Shadow Dexterous.
- **Función de recompensa:** Densa, definida como el negativo de la distancia euclidiana global:
  $$r = - \|\text{achieved\_goal} - \text{desired\_goal}\|_2$$
- **Criterio de éxito físico:** Norma global estrictamente inferior a $0.01\text{ m}$ ($10\text{ mm}$ de error acumulado). La señal de éxito por paso se captura en `info["is_success"]` y el éxito del episodio en `success_final`.
- **Horizonte temporal:** 50 pasos por episodio (frecuencia de control a $25\text{ Hz}$, $dt = 0.04\text{ s}$).

## Arquitectura de Solución (PPO Vectorizado)

Frente a la formulación previa secuencial no normalizada (que colapsó la exploración estancándose en $67.5\text{ mm}$ de error), la presente arquitectura introduce:

1. **Entornos Vectorizados Concurrentes (`DummyVecEnv` con 8 workers):**
   - Recolecta lotes simultáneos de 8 subentornos independientes.
   - Rompe la autocorrelación temporal de las trayectorias on-policy y reduce drásticamente la varianza del estimador de ventaja ($GAE$).
2. **Normalización Vectorial de Observaciones (`VecNormalize`):**
   - Estandariza en línea las 93 variables del espacio de entrada (`norm_obs=True, norm_reward=False, clip_obs=10.0`), homogeneizando la escala entre radianes, velocidades angulares y metros.
3. **Regulación de Entropía:**
   - Coeficiente de entropía activo (`ent_coef = 0.005`) para prevenir el colapso prematuro de la varianza en los 20 actuadores continuos.
4. **Red de Política y Valor:**
   - Arquitectura MLP separada `dict(pi=[256, 256], vf=[256, 256])` con inicialización ortogonal (`ortho_init=True`) y activación ReLU.

## Matriz de Semillas y Separación Experimental

- **Entrenamiento:** Semilla 42 (+ rank de worker para subentornos).
- **Caracterización Aleatoria Inicial:** Semillas 101–120 (20 episodios).
- **Seguimiento Intermedio (Tuning):** Semillas 201–210.
- **Selección de Candidato:** Semillas 201–210.
- **Evaluación Final Reservada:** Semillas 1001–1010.
- **Generación de Video:** Semilla 301.

## Estado de Validación de Gates

- **Gate 1 (Runtime Probe):** Superado. Formas de tensor, límites articulares, física y cálculo de distancias verificado en `results/runtime_contract.json`.
- **Gate 2 (Smoke Test):** Superado.
  - Ejecución de 8.192 timesteps con 8 workers en 15.1 segundos.
  - Generación y verificación de integridad de `candidate.zip` y `vec_normalize.pkl`.
  - Prueba de recarga desacoplada superada (`reload_evaluation.csv` idéntico a inferencia en memoria).
  - Renderizado de video con overlay informativo `smoke.mp4` (50 frames a 25 FPS).
  - Generación de gráficas analíticas en `results/figures/`.
  - Generación de videos oficiales en `videos/` (`training_process.mp4` y `trained.mp4`).
- **Gate 3 (Full Training):** Listo para ejecución bajo demanda modificando `CONFIG["profile"] = "full"`.
