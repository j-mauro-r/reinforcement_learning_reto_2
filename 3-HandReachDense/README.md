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
- **Criterio de éxito físico:** Norma global estrictamente inferior a $0.01\text{ m}$ ($10\text{ mm}$ de error acumulado en las 5 puntas). La señal de éxito por paso se captura en `info["is_success"]` y el éxito del episodio en `success_final`.
- **Horizonte temporal:** 50 pasos por episodio (frecuencia de control a $25\text{ Hz}$, $dt = 0.04\text{ s}$).

## Arquitectura de Solución (PPO Vectorizado de Alta Precisión)

1. **Entornos Vectorizados Concurrentes (`DummyVecEnv` con 8 workers):**
   - Recolecta lotes simultáneos de 8 subentornos independientes, descorrelacionando trayectorias y estabilizando el cálculo de ventajas $GAE$.
2. **Normalización Vectorial de Observaciones (`VecNormalize`):**
   - Estandariza en línea las 93 variables de entrada (`norm_obs=True, norm_reward=False, clip_obs=10.0`), homogeneizando las magnitudes entre ángulos, velocidades y metros.
3. **Control Fino de Entropía y Convergencia Milimétrica:**
   - Decaimiento de la entropía (`ent_coef = 0.0001`) y tasa de aprendizaje fina (`1e-4`) para concentrar la densidad de probabilidad en los 20 actuadores y reducir las oscilaciones residuales a escala milimétrica.
4. **Red de Política y Valor:**
   - Arquitectura MLP separada `dict(pi=[256, 256], vf=[256, 256])` con inicialización ortogonal (`ortho_init=True`) y activación ReLU.

## Resultados de la Corrida Full Optimizada (1.500.000 de pasos)

- **Tiempo total de entrenamiento:** $\approx 38\text{ minutos}$ ($2.300\text{ s}$) sobre 8 workers concurrentes.
- **Evolución del error de distancia:**
  - Línea base aleatoria: $130.7\text{ mm}$ de error medio.
  - $250.000$ pasos: $42.7\text{ mm}$
  - $500.000$ pasos: $25.3\text{ mm}$
  - $1.000.000$ pasos: $16.9\text{ mm}$
  - **$1.500.000$ pasos (Fine-Tuning de Precisión):** **$12.6 - 13.5\text{ mm}$** (promedio de $\approx 2.7\text{ mm}$ por punta de dedo).

### Tabla Comparativa de Evaluación Final (10 Episodios Reservados, Semillas 1001–1010)

| Métrica Formal | Política Aleatoria (Baseline) | Agente PPO Entrenado | Variación / Mejora |
| :--- | :--- | :--- | :--- |
| **Retorno Medio** ($\pm \text{DE}$) | $-6.216 \pm 0.533$ | **$-0.775 \pm 0.106$** | **+87.5% de incremento** |
| **Distancia Final Media** | $130.7\text{ mm}$ | **$13.5\text{ mm}$** | **-89.6% de reducción del error** |
| **Tasa de Éxito Físico Global** ($< 10\text{ mm}$) | $0.0\%$ | **$10.0\%$ - $20.0\%$** | Distancia a solo $3.5\text{ mm}$ del umbral global de 5 dedos |

## Artefactos Canónicos y Evidencias

- **Modelo Canónico:** [models/hand_reach_ppo.zip](models/hand_reach_ppo.zip)
- **Estadísticas de Normalización:** [models/vec_normalize.pkl](models/vec_normalize.pkl)
- **Metadatos de Procedencia:** [models/hand_reach_ppo.metadata.json](models/hand_reach_ppo.metadata.json)
- **Evaluaciones Cuantitativas:** [results/evaluation.csv](results/evaluation.csv), [results/metrics.json](results/metrics.json) y [results/random_baseline_final.csv](results/random_baseline_final.csv).
- **Figuras Analíticas:**
  - `results/figures/figure_1_training_curves.png`: Curva de recompensa acumulada.
  - `results/figures/figure_2_evaluation_distances.png`: Reducción de la distancia euclidiana hacia el umbral físico.
  - `results/figures/figure_3_training_diagnostics.png`: Pérdida de valor, entropía y fracción de clipping.
- **Videos Generados:**
  - [videos/training_process.mp4](videos/training_process.mp4): Checkpoint intermedio con overlay informativo.
  - [videos/trained.mp4](videos/trained.mp4): Agente canónico final con telemetría en tiempo real (25 FPS).
