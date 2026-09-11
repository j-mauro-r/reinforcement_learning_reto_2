# Ficha técnica — HandReachDense

> **Ambiente objetivo:** `HandReachDense-v3`  
> **Proyecto:** Reto II — Reinforcement Learning / Gymnasium Robotics  
> **Propósito:** caracterizar integralmente el tercer ambiente antes de definir su DWP, política, hiperparámetros y entrenamiento. Esta ficha establece el contrato técnico conocido del environment, sus variables, rangos, dinámica, reward, criterio de éxito, aleatoriedad, restricciones y validaciones que deberán ejecutarse en runtime.

---

## 1. Alcance y fuentes

El tercer problema del reto utiliza la **Shadow Dexterous Hand** en la tarea **Reach**. El objetivo es llevar las puntas de los dedos a posiciones cartesianas objetivo. El enunciado exige la variante densa `HandReachDense-v3`; el objetivo académico común es maximizar la recompensa acumulada considerando también estabilidad y tiempo de entrenamiento.

Fuentes utilizadas:

1. `enunciado-reto.md`.
2. `lineamientos-transversales.md`.
3. `Gymnasium_Robotics.ipynb` como referencia de instalación, creación y render.
4. Código oficial `gymnasium_robotics/envs/shadow_dexterous_hand/reach.py`.
5. Código base `gymnasium_robotics/envs/shadow_dexterous_hand/hand_env.py`.
6. Código base `gymnasium_robotics/envs/robot_env.py`.
7. Registro de environments en `gymnasium_robotics/__init__.py`.
8. XML MuJoCo `gymnasium_robotics/envs/assets/hand/reach.xml` y robot asociado.

### 1.1 Regla de versión

El reto fija el ID `HandReachDense-v3`, pero no fija aún un pin concreto de Gymnasium Robotics, Gymnasium, MuJoCo, PyTorch o Stable-Baselines3.

La versión final de las dependencias deberá fijarse en el DWP después de validar una combinación funcional tanto localmente como en Google Colab.

**Regla:** cualquier diferencia material entre esta ficha y el environment realmente instalado deberá bloquear un entrenamiento largo hasta identificar, reproducir y documentar la causa.

---

## 2. Resumen ejecutivo del environment

| Propiedad | Contrato conocido |
| --- | --- |
| Environment ID | `HandReachDense-v3` |
| Familia | Shadow Dexterous Hand |
| Tarea | Reach |
| Motor físico | MuJoCo |
| Mano | antropomórfica |
| Joints físicos | 24 |
| Grados de libertad motorizados | 20 |
| Action space | `Box(-1.0, 1.0, shape=(20,), dtype=float32)` |
| Tipo de control | posiciones angulares absolutas escaladas al rango real del actuador |
| `relative_control` baseline | `False` |
| Observation space | `Dict` goal-aware |
| `observation` | `(63,)` |
| `achieved_goal` | `(15,)` = XYZ de cinco puntas |
| `desired_goal` | `(15,)` = XYZ objetivo de cinco puntas |
| Reward Dense | `-||achieved_goal - desired_goal||₂` |
| Mejor reward instantáneo | `0.0` |
| Success threshold | norma 2 global `< 0.01 m` |
| Señal de éxito | `info["is_success"]` |
| Horizonte por defecto | 50 pasos |
| `terminated` | `False` |
| `truncated` | por `TimeLimit` a 50 pasos |
| MuJoCo timestep | `0.002 s` |
| Substeps por acción | 20 |
| Tiempo por decisión | `0.04 s` |
| Frecuencia de control | 25 Hz |
| Duración simulada de 50 pasos | `2.0 s` |
| Goal principal | pulgar + un dedo aleatorio se aproximan sobre la palma |
| Goal especial | 10%: mantener todas las puntas en posición inicial |
| Render modes | `human`, `rgb_array` |
| Render FPS | 25 |
| Cámara base | distance 0.5, azimuth 55°, elevation -25° |

---

## 3. Objetivo físico de la tarea

El agente debe coordinar muñeca y dedos para minimizar simultáneamente la distancia entre las cinco puntas y sus posiciones objetivo.

En la mayoría de episodios:

1. se elige aleatoriamente uno de los cuatro dedos no pulgar;
2. ese dedo y el pulgar reciben targets próximos a un punto de encuentro sobre la palma;
3. los otros tres dedos mantienen como target su posición inicial;
4. la política controla 20 actuadores para minimizar el error global de las 15 coordenadas objetivo.

En aproximadamente 10% de los episodios el objetivo es mantener las cinco puntas en su configuración inicial.

La tarea es de **coordinación multiarticular**: el objetivo se expresa en espacio cartesiano, pero la política actúa sobre posiciones angulares normalizadas.

---

## 4. Sistema físico de la Shadow Dexterous Hand

### 4.1 Joints y actuadores

La mano tiene 24 joints y 20 DoF motorizados:

- muñeca: 2 DoF;
- índice: 4 joints físicos, 3 actuados;
- medio: 4 joints físicos, 3 actuados;
- anular: 4 joints físicos, 3 actuados;
- meñique: 5 joints físicos, 4 actuados;
- pulgar: 5 joints y 5 actuados.

En índice, medio, anular y meñique, el joint distal `J0` está acoplado mediante tendón al `J1`. Por tanto, `FFJ0`, `MFJ0`, `RFJ0` y `LFJ0` aparecen en el estado físico pero no son acciones independientes.

### 4.2 Nomenclatura

| Prefijo | Parte |
| --- | --- |
| `WR` | wrist / muñeca |
| `FF` | forefinger / índice |
| `MF` | middle finger |
| `RF` | ring finger / anular |
| `LF` | little finger / meñique |
| `TH` | thumb / pulgar |

### 4.3 Control absoluto

El environment usa por defecto `relative_control=False`. Cada acción normalizada se transforma internamente al rango del actuador:

```text
center = (max + min) / 2
half_range = (max - min) / 2
ctrl = center + action * half_range
ctrl = clip(ctrl, min, max)
```

De esta forma:

- `-1` corresponde al límite inferior del actuador;
- `0` al centro del rango;
- `+1` al límite superior.

El baseline no deberá activar control relativo sin una decisión explícita y validada en el DWP.

---

## 5. Espacio de acciones

### 5.1 Contrato

```text
Box(low=-1.0, high=1.0, shape=(20,), dtype=float32)
```

Todas las acciones son continuas.

### 5.2 Mapa de acciones

| idx | Joint | Movimiento | Rango normalizado | Rango angular aproximado |
| ---: | --- | --- | --- | --- |
| 0 | `WRJ1` | muñeca radial/ulnar | `[-1,1]` | `[-0.489, 0.140] rad` |
| 1 | `WRJ0` | muñeca flex/ext | `[-1,1]` | `[-0.698, 0.489] rad` |
| 2 | `FFJ3` | índice MCP abd/add | `[-1,1]` | `[-0.349, 0.349]` |
| 3 | `FFJ2` | índice MCP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 4 | `FFJ1` | índice PIP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 5 | `MFJ3` | medio MCP abd/add | `[-1,1]` | `[-0.349, 0.349]` |
| 6 | `MFJ2` | medio MCP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 7 | `MFJ1` | medio PIP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 8 | `RFJ3` | anular MCP abd/add | `[-1,1]` | `[-0.349, 0.349]` |
| 9 | `RFJ2` | anular MCP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 10 | `RFJ1` | anular PIP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 11 | `LFJ4` | meñique CMC oposición | `[-1,1]` | `[0, 0.785]` |
| 12 | `LFJ3` | meñique MCP abd/add | `[-1,1]` | `[-0.349, 0.349]` |
| 13 | `LFJ2` | meñique MCP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 14 | `LFJ1` | meñique PIP flex/ext | `[-1,1]` | `[0, 1.571]` |
| 15 | `THJ4` | pulgar CMC horizontal | `[-1,1]` | `[-1.047, 1.047]` |
| 16 | `THJ3` | pulgar CMC vertical | `[-1,1]` | `[0, 1.222]` |
| 17 | `THJ2` | pulgar MCP abd/add | `[-1,1]` | `[-0.209, 0.209]` |
| 18 | `THJ1` | pulgar MCP flex/ext | `[-1,1]` | `[-0.524, 0.524]` |
| 19 | `THJ0` | pulgar IP flex/ext | `[-1,1]` | `[-1.571, 0]` |

### 5.3 Joints acoplados no actuados

No existen acciones independientes para:

```text
FFJ0, MFJ0, RFJ0, LFJ0
```

Su movimiento está acoplado a los joints proximales correspondientes mediante tendones MuJoCo.

### 5.4 Saturación

La API recorta acciones fuera de `[-1,1]`, y la clase de mano recorta el control final contra `actuator_ctrlrange`. El agente no debe depender de esta saturación como estrategia de control. Conviene medir la fracción de acciones cercanas a ±1.

---

## 6. Espacio de observaciones

### 6.1 Contrato GoalEnv

```text
Dict(
  observation:   Box(-inf, inf, shape=(63,)),
  achieved_goal: Box(-inf, inf, shape=(15,)),
  desired_goal:  Box(-inf, inf, shape=(15,))
)
```

No es un vector plano.

### 6.2 Composición de `observation (63,)`

La implementación concatena:

```text
24 qpos + 24 qvel + 15 coordenadas XYZ de puntas = 63
```

Índices `0..23`: posiciones articulares, en este orden:

```text
WRJ1, WRJ0,
FFJ3, FFJ2, FFJ1, FFJ0,
MFJ3, MFJ2, MFJ1, MFJ0,
RFJ3, RFJ2, RFJ1, RFJ0,
LFJ4, LFJ3, LFJ2, LFJ1, LFJ0,
THJ4, THJ3, THJ2, THJ1, THJ0
```

Índices `24..47`: velocidades articulares de los mismos 24 joints en el mismo orden.

Índices `48..62`: coordenadas de las puntas:

| Índices | Punta | Site |
| --- | --- | --- |
| 48–50 | índice XYZ | `robot0:S_fftip` |
| 51–53 | medio XYZ | `robot0:S_mftip` |
| 54–56 | anular XYZ | `robot0:S_rftip` |
| 57–59 | meñique XYZ | `robot0:S_lftip` |
| 60–62 | pulgar XYZ | `robot0:S_thtip` |

### 6.3 `achieved_goal (15,)`

Contiene las posiciones actuales de las cinco puntas en el mismo orden:

```text
índice XYZ | medio XYZ | anular XYZ | meñique XYZ | pulgar XYZ
```

### 6.4 `desired_goal (15,)`

Tiene la misma estructura, pero representa las posiciones objetivo.

### 6.5 Redundancia deliberada

Las posiciones de las cinco puntas aparecen al final de `observation` y también en `achieved_goal`. Es parte del contrato GoalEnv y no debe eliminarse silenciosamente.

### 6.6 Implicación para Stable-Baselines3

Si el DWP mantiene PPO como hipótesis transversal, deberá validar una policy compatible con `spaces.Dict`, previsiblemente `MultiInputPolicy`, o justificar un flattening explícito. El baseline debería conservar el Dict si la librería lo soporta de forma madura.

---

## 7. Estado inicial

Los joints se inicializan aproximadamente en:

| Joint | Ángulo (rad) |
| --- | ---: |
| `WRJ1` | -0.1651433975 |
| `WRJ0` | -0.3197328657 |
| `FFJ3` | 0.1434051255 |
| `FFJ2` | 0.3202820833 |
| `FFJ1` | 0.7126053608 |
| `FFJ0` | 0.6705281001 |
| `MFJ3` | 0.0002464443 |
| `MFJ2` | 0.3152655251 |
| `MFJ1` | 0.7659800314 |
| `MFJ0` | 0.7323156897 |
| `RFJ3` | 0.0003852070 |
| `RFJ2` | 0.3674354620 |
| `RFJ1` | 0.7119514095 |
| `RFJ0` | 0.6699446328 |
| `LFJ4` | 0.0525442258 |
| `LFJ3` | -0.1361553472 |
| `LFJ2` | 0.3987203043 |
| `LFJ1` | 0.7415570010 |
| `LFJ0` | 0.7040963787 |
| `THJ4` | 0.0036738238 |
| `THJ3` | 0.5506291436 |
| `THJ2` | -0.0145151520 |
| `THJ1` | -0.0015229224 |
| `THJ0` | -0.7894883022 |

`v3` corrigió históricamente un problema en el que el estado inicial efectivo no coincidía con la documentación. El DWP deberá verificar el estado real después de `reset(seed=...)`.

---

## 8. Posiciones iniciales de las puntas

Valores documentados aproximados:

| Dedo | X (m) | Y (m) | Z (m) |
| --- | ---: | ---: | ---: |
| Índice | 0.99 | 0.80 | 0.150 |
| Medio | 1.02 | 0.80 | 0.150 |
| Anular | 1.04 | 0.81 | 0.155 |
| Meñique | 1.07 | 0.82 | 0.160 |
| Pulgar | 0.95 | 0.84 | 0.160 |

Son referencias nominales y deberán comprobarse con `achieved_goal` en runtime.

---

## 9. Generación del goal

La función `_sample_goal()` define dos regímenes.

### 9.1 Caso principal — ~90%

1. Se elige uniformemente uno entre índice, medio, anular o meñique.
2. El pulgar siempre participa.
3. Se define un punto de encuentro:

```text
meeting_pos = palm_xpos + [0.0, -0.09, 0.05]
```

4. Se añade ruido normal independiente por coordenada:

```text
Normal(0, 0.005 m)
```

5. El goal parte de las posiciones iniciales de las cinco puntas.
6. Solo el dedo elegido y el pulgar cambian de target.
7. Para evitar solaparlos exactamente, cada target se coloca a `0.005 m` del punto de encuentro en la dirección de su posición inicial.

Los tres dedos no seleccionados mantienen como objetivo su posición inicial.

### 9.2 Caso especial — ~10%

Con probabilidad `0.1`, el goal completo se reemplaza por `initial_goal`.

El objetivo es que las cinco puntas permanezcan en su configuración inicial. Este caso evita que el pulgar aprenda a quedarse permanentemente cerca de la región de encuentro.

### 9.3 Fuentes de aleatoriedad

- dedo no pulgar seleccionado;
- ruido Gaussiano de 5 mm sobre el meeting point;
- decisión del caso especial del 10%.

Las seeds de entrenamiento, tuning y evaluación final deberán permanecer separadas.

---

## 10. Recompensa

### 10.1 Reward Dense

Sea:

```text
a = achieved_goal ∈ R^15
g = desired_goal ∈ R^15
d = ||a-g||₂
```

La recompensa por step es:

```text
reward = -d
```

Propiedades:

- valor máximo instantáneo `0.0`;
- cuanto menor el error, menos negativa la recompensa;
- existe feedback en todos los steps;
- no hay bonus adicional por éxito;
- no hay penalización explícita por energía, velocidad o magnitud de acción;
- el error agrega las cinco puntas en una sola norma sobre 15 coordenadas.

### 10.2 Recompensa acumulada

```text
R_episode = Σ reward_t
```

Con horizonte de 50 pasos, llegar rápido y permanecer cerca del goal produce retorno mayor, es decir, menos negativo.

### 10.3 Variante sparse de referencia

La variante sparse devuelve `0` si la distancia global es `<0.01 m` y `-1` en otro caso. El reto exige la variante Dense y no debe sustituirse.

### 10.4 Implicación geométrica

El threshold se aplica sobre la **norma conjunta de las 15 coordenadas**, no sobre cada punta de manera independiente.

Si, solo como intuición, las cinco puntas tuvieran el mismo error escalar `e`, entonces:

```text
sqrt(5) * e < 0.01
⇒ e < ~0.00447 m
```

El cálculo real es 15-dimensional, por lo que esta aproximación no sustituye la métrica oficial.

---

## 11. Criterio de éxito

### 11.1 Success oficial por step

```text
is_success = ||achieved_goal - desired_goal||₂ < 0.01
```

La comparación es estricta.

### 11.2 `info["is_success"]`

Cada `step()` expone la señal de éxito como valor numérico compatible con `0.0/1.0`.

### 11.3 Tarea continua

Al alcanzar el objetivo:

```text
terminated = False
```

El episodio continúa. Una política puede alcanzar el umbral y volver a salir de él antes del TimeLimit.

### 11.4 Telemetría de éxito recomendada

El DWP deberá definir la semántica principal antes de la evaluación final. Conviene registrar simultáneamente:

- `success_any`;
- `success_final`;
- `success_fraction`;
- `time_to_first_success`;
- `goal_distance`;
- `min_goal_distance`;
- `final_goal_distance`;
- distancia de cada fingertip a su target;
- máximo error por fingertip.

Para una tarea de mantener posiciones, `success_final` y `success_fraction` son especialmente informativos.

---

## 12. Ciclo temporal y fin del episodio

### 12.1 API

```python
obs, reward, terminated, truncated, info = env.step(action)
```

### 12.2 Horizonte

El registro oficial esperado para `HandReachDense-v3` es:

```text
max_episode_steps = 50
```

### 12.3 Continuing task

El environment no termina por éxito. `TimeLimit` es quien establece `truncated=True` al completar el horizonte.

### 12.4 Frecuencia

El XML usa `timestep=0.002 s` y Reach configura `n_substeps=20`:

```text
control_dt = 0.002 * 20 = 0.04 s
control_frequency = 25 Hz
50 * 0.04 = 2.0 s simulados
```

---

## 13. Rendering y cámaras

### 13.1 Modos soportados

```text
human
rgb_array
```

### 13.2 FPS

```text
render_fps = 25
```

### 13.3 Cámara base

```text
distance = 0.5
azimuth = 55°
elevation = -25°
lookat = [1.0, 0.96, 0.14]
```

### 13.4 Sites visuales

`reach.xml` define:

- `target0..target4` para los targets;
- `finger0..finger4` para representar las posiciones actuales.

Los targets se colorean de forma diferenciada: rojo, verde, azul, amarillo y magenta. El callback de render actualiza sus posiciones a partir del goal y achieved_goal.

### 13.5 Regla de entrenamiento

No grabar/renderizar todo el entrenamiento largo frame a frame. El render se reserva para smoke y evidencias cortas de video.

---

## 14. Parámetros internos relevantes

| Parámetro | Valor / comportamiento |
| --- | --- |
| `distance_threshold` | `0.01` |
| `n_substeps` | `20` |
| `relative_control` | `False` |
| `reward_type` | `dense` para el ID del reto |
| action dimensions | `20` |
| goal dimensions | `15` |
| meeting offset | `[0,-0.09,0.05] m` |
| meeting noise std | `0.005 m` |
| initial-goal probability | `0.10` |
| anti-overlap target offset | `0.005 m` |
| MuJoCo timestep | `0.002 s` |

Estos son parámetros de la definición del environment, no hiperparámetros del agente. El baseline no deberá alterarlos.

---

## 15. Variables configurables del proyecto

La configuración del notebook/DWP deberá centralizar como mínimo:

| Variable | Baseline / estado |
| --- | --- |
| `ENV_ID` | `HandReachDense-v3` |
| `REWARD_TYPE` | `dense` |
| `ACTION_DIM` | `20` |
| `ACTION_LOW/HIGH` | `-1 / 1` |
| `OBS_VECTOR_DIM` | `63` |
| `ACHIEVED_GOAL_DIM` | `15` |
| `DESIRED_GOAL_DIM` | `15` |
| `SUCCESS_THRESHOLD` | `0.01 m` |
| `MAX_EPISODE_STEPS` | `50` esperado |
| `N_SUBSTEPS` | `20` |
| `MUJOCO_TIMESTEP` | `0.002 s` esperado |
| `CONTROL_DT` | `0.04 s` |
| `CONTROL_HZ` | `25` |
| `RELATIVE_CONTROL` | `False` |
| `MEETING_OFFSET` | `[0,-0.09,0.05] m` |
| `MEETING_NOISE_STD` | `0.005 m` |
| `INITIAL_GOAL_PROB` | `0.10` |
| `ALGORITHM` | hipótesis inicial `PPO` |
| `POLICY` | por validar para Dict |
| `SEED` | por definir DWP |
| `TOTAL_TIMESTEPS` | por definir DWP |
| `LEARNING_RATE` | por definir DWP |
| `GAMMA` | por definir DWP |
| `EVAL_EPISODES` | mínimo `10` |
| `DEVICE` | cpu/cuda/mps según gate |
| `RENDER_MODE_TRAIN` | `None` |
| `RENDER_MODE_VIDEO` | `rgb_array` |
| `PROJECT_ROOT` | única raíz |
| `MODEL_DIR` | derivada |
| `VIDEO_DIR` | derivada |
| `RESULTS_DIR` | derivada |

No se deberán repetir estas constantes como números mágicos en varias celdas.

---

## 16. Restricciones del problema

### 16.1 Restricciones duras

1. Action space continuo de 20 dimensiones.
2. Acciones son objetivos angulares escalados, no torques.
3. Control absoluto por defecto.
4. Existen 24 joints, pero solo 20 actuadores.
5. Cuatro joints distales están acoplados.
6. Observation space es `Dict`.
7. Reward Dense = norma 2 negativa del error global 15D.
8. Success requiere norma global `<0.01 m`.
9. Éxito no termina el episodio.
10. Horizonte esperado de 50 steps.
11. Goal cambia entre coordinación pulgar-dedo y mantenimiento inicial.
12. No existe objeto externo manipulable.
13. No alterar XML, reward ni threshold en baseline.
14. Frecuencia de control 25 Hz.
15. Los goals pertenecen a las cinco puntas, no a joints individuales.

### 16.2 Restricciones académicas/transversales

1. Debe utilizarse `HandReachDense-v3` salvo excepción aprobada/documentada.
2. Debe conservarse reward Dense.
3. Algoritmo perteneciente a la lista permitida.
4. El reto global debe usar mínimo tres algoritmos distintos.
5. Hipótesis transversal inicial: PPO.
6. No HER, reward shaping adicional ni curriculum en el primer baseline.
7. Entrenamiento y evaluación separados.
8. Evaluación final mínimo 10 episodios.
9. Baseline aleatorio comparable.
10. 3 métricas principales y 3 gráficas principales.
11. `training_process.mp4` y `trained.mp4`.
12. Un único artefacto canónico final.
13. DWP obligatorio para desarrollo sustantivo.
14. Ningún cambio directo sobre `main`.
15. Notebook final reproducible en Colab limpio.
16. Simplicidad, SOLID/DRY proporcional y notebook-first.

---

## 17. Telemetría recomendada

| Variable | Uso |
| --- | --- |
| `episode_return` | objetivo RL acumulado |
| `episode_length` | horizonte real |
| `goal_distance` | error global 15D |
| `min_goal_distance` | mejor aproximación |
| `final_goal_distance` | estabilidad final |
| `is_success` | éxito oficial |
| `success_any` | alcance transitorio |
| `success_final` | éxito al final |
| `success_fraction` | permanencia dentro del threshold |
| `time_to_first_success` | rapidez |
| `finger_distance_ff/mf/rf/lf/th` | error individual por punta |
| `max_fingertip_error` | dedo rezagado |
| `action_mean_abs` | intensidad de control proxy |
| `action_saturation_fraction` | saturación |
| `training_timesteps` | presupuesto |
| `wall_clock_training_time` | costo real |
| `eval_reward_mean/std` | rendimiento y estabilidad |
| `success_rate` | capacidad funcional |

Estas variables son telemetría; no deben incorporarse automáticamente al estado del agente.

---

## 18. Contrato de validación previo al entrenamiento

### 18.1 Aserciones mínimas

```python
assert env.spec.id == "HandReachDense-v3"
assert env.action_space.shape == (20,)
assert env.action_space.low.min() == -1.0
assert env.action_space.high.max() == 1.0

obs, info = env.reset(seed=SEED)
assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
assert obs["observation"].shape == (63,)
assert obs["achieved_goal"].shape == (15,)
assert obs["desired_goal"].shape == (15,)
```

### 18.2 Datos que deben imprimirse/registrarse

```text
gymnasium.__version__
gymnasium_robotics.__version__
mujoco.__version__
env.spec
env.spec.max_episode_steps
env.action_space
env.observation_space
env.unwrapped.n_substeps
env.unwrapped.model.opt.timestep
env.unwrapped.dt
env.unwrapped.distance_threshold
env.unwrapped.reward_type
env.unwrapped.relative_control
env.unwrapped.model.actuator_ctrlrange
```

### 18.3 Reward probe

```python
expected = -np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
assert np.isclose(reward, expected)
```

### 18.4 Goal sampling probe

Sobre múltiples resets con seeds distintas:

- confirmar targets tipo pulgar + dedo seleccionado;
- confirmar existencia del caso `initial_goal`;
- no exigir exactamente 10% en una muestra pequeña;
- repetir una seed y verificar `desired_goal` reproducible;
- usar seeds distintas y verificar variación del goal.

### 18.5 Smoke mínimo

1. `gym.make("HandReachDense-v3")`;
2. `reset(seed=...)`;
3. validar Dict y shapes;
4. varios `step()`;
5. reward finita y coherente;
6. `info["is_success"]` presente;
7. `terminated=False`;
8. truncation al horizonte efectivo;
9. `rgb_array` funcional;
10. `close()` correcto;
11. reproducibilidad de seed;
12. guardar caracterización runtime.

---

## 19. Implicaciones para el diseño del agente

### 19.1 Control continuo de alta dimensión

20 acciones simultáneas hacen el problema más complejo que Fetch Reach. La policy debe coordinar muñeca, dedos y joints acoplados indirectamente.

### 19.2 Observation Dict

La policy debe consumir `observation`, `achieved_goal` y `desired_goal`, o una representación equivalente documentada. El DWP debe validar la policy concreta antes del smoke training.

### 19.3 PPO como hipótesis inicial

Los lineamientos proponen PPO como tercer algoritmo para:

- cumplir diversidad global del reto;
- usar un método on-policy estable;
- abordar control continuo de alta dimensionalidad.

Es una hipótesis inicial, no una conclusión. El DWP deberá justificar y confirmar la decisión con compatibilidad/runtime.

### 19.4 Reward densa

La señal ya proporciona progreso geométrico en cada step. El baseline no necesita shaping adicional.

### 19.5 Success estricto

El threshold global de 1 cm en 15 dimensiones puede generar baja tasa de éxito aun cuando la recompensa mejore. Por eso deben registrarse distancias globales y por fingertip sin reemplazar el success oficial.

### 19.6 Generalización multi-goal

La policy debe resolver:

- cuatro posibilidades de dedo emparejado con pulgar;
- ruido del meeting point;
- episodios de mantenimiento inicial.

Una única seed no demuestra aprendizaje suficiente.

---

## 20. Configuración baseline del environment propuesta

```yaml
environment:
  id: HandReachDense-v3
  reward: dense
  max_episode_steps_expected: 50
  relative_control: false

  action:
    shape: [20]
    low: -1.0
    high: 1.0
    semantics: absolute_scaled_joint_targets

  observation:
    type: Dict
    observation_shape: [63]
    achieved_goal_shape: [15]
    desired_goal_shape: [15]

  success:
    distance_threshold_m: 0.01
    distance_type: l2_over_15d_goal_vector

  goal_sampling:
    thumb_plus_random_finger_probability: 0.90
    initial_goal_probability: 0.10
    meeting_offset_m: [0.0, -0.09, 0.05]
    meeting_noise_std_m: 0.005
    anti_overlap_offset_m: 0.005

  simulation:
    mujoco_timestep_s_expected: 0.002
    n_substeps_expected: 20
    control_dt_s_expected: 0.04
    control_frequency_hz_expected: 25

  rendering:
    training: null
    video: rgb_array
    fps_expected: 25
```

**Regla:** cualquier diferencia frente al runtime debe explicarse antes del entrenamiento largo.

---

## 21. Decisiones pendientes para el DWP de HandReachDense

1. pins exactos de dependencias;
2. compatibilidad local/Colab;
3. confirmación runtime de v3, 50 pasos y 25 Hz;
4. algoritmo final — hipótesis PPO;
5. policy compatible con Dict;
6. arquitectura actor/critic;
7. normalización si fuese necesaria;
8. seed de entrenamiento;
9. seeds de tuning;
10. seeds finales;
11. presupuesto de timesteps;
12. learning rate;
13. gamma;
14. GAE lambda si PPO;
15. clip range;
16. entropy coefficient;
17. value-function coefficient;
18. batch size;
19. rollout length / `n_steps`;
20. epochs por rollout;
21. gradient clipping;
22. device;
23. frecuencia de evaluación/checkpoints;
24. semántica principal del success rate;
25. objetivo cuantitativo interno;
26. criterio de selección del único modelo final;
27. baseline aleatorio;
28. estrategia de videos;
29. Mac mini vs Colab;
30. tuning máximo permitido;
31. criterio para `dwp-refine`;
32. tratamiento de mejora de reward sin alcanzar success.

---

## 22. Checklist de caracterización

- [x] Environment y variante Dense identificados.
- [x] 24 joints / 20 actuadores documentados.
- [x] 20 acciones y rangos documentados.
- [x] Control absoluto/escalado documentado.
- [x] Joints acoplados identificados.
- [x] Observation Dict documentado.
- [x] `observation (63,)`, `achieved_goal (15,)`, `desired_goal (15,)` documentados.
- [x] Reward Dense y success documentados.
- [x] Goal sampling 90%/10% documentado.
- [x] Meeting point, ruido y anti-overlap documentados.
- [x] Estado inicial documentado.
- [x] Horizonte/frecuencia documentados.
- [x] Render/cámara/sites documentados.
- [x] Restricciones académicas documentadas.
- [x] Telemetría propuesta.
- [x] Contrato smoke/runtime definido.
- [x] Implicaciones para PPO identificadas.
- [ ] Versiones finales fijadas — pendiente del DWP/runtime.
- [ ] Algoritmo/policy definitivos — pendiente del DWP.
- [ ] Hiperparámetros — pendiente del DWP.
- [ ] Success rate principal — pendiente del DWP.

---

## 23. Riesgos técnicos para el DWP

### A. Alta dimensionalidad de acciones
20 actuadores pueden elevar el costo y la inestabilidad del aprendizaje.

### B. Success muy estricto
La recompensa puede mejorar materialmente sin cruzar a menudo el threshold global de 1 cm.

### C. Goals heterogéneos
Se deben resolver cuatro configuraciones dedo-pulgar y el caso de mantenimiento inicial.

### D. Joints acoplados
La policy controla 20 DoF pero observa 24 joints; la interpretación de acciones/estado debe respetar ese contrato.

### E. Costo on-policy
Si PPO se confirma, el costo de interacción puede ser mayor que en algoritmos off-policy. Debe medirse wall-clock y sample efficiency.

### F. GPU no garantiza speedup
MuJoCo puede estar limitado por simulación CPU; `cuda`/`mps` deberán adoptarse solo con evidencia.

### G. Drift de dependencias
Las versiones deberán fijarse una vez validadas.

---

## 24. Referencias técnicas

- `enunciado-reto.md`.
- `lineamientos-transversales.md`.
- `Gymnasium_Robotics.ipynb`.
- Gymnasium Robotics — Shadow Dexterous Hand Reach.
- `gymnasium_robotics/envs/shadow_dexterous_hand/reach.py`.
- `gymnasium_robotics/envs/shadow_dexterous_hand/hand_env.py`.
- `gymnasium_robotics/envs/robot_env.py`.
- `gymnasium_robotics/__init__.py`.
- `gymnasium_robotics/envs/assets/hand/reach.xml`.
- Farama Gymnasium Robotics upstream.

---

## 25. Regla de mantenimiento

Esta ficha es contexto técnico durable para el DWP, no un sustituto de la inspección runtime.

Si el environment instalado contradice una variable aquí documentada:

1. detener entrenamiento largo;
2. registrar versiones exactas;
3. reproducir la diferencia con un smoke mínimo;
4. contrastar con source de la versión instalada;
5. identificar si cambia documentación, versión o configuración;
6. actualizar esta ficha si corresponde;
7. refinar el DWP si afecta acciones, observaciones, reward, success, goal sampling, horizonte o reproducibilidad.

No se modificará el environment para “hacerlo coincidir” con esta ficha sin una decisión técnica explícita.