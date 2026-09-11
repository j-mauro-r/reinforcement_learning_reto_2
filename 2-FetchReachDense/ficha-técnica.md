# Ficha técnica — FetchReachDense

> **Ambiente objetivo:** `FetchReachDense-v4`  
> **Proyecto:** Reto II — Reinforcement Learning / Gymnasium Robotics  
> **Propósito:** caracterizar por completo el ambiente antes de definir el DWP, la política, los hiperparámetros y el entrenamiento del modelo `2-FetchReachDense`. Este documento establece el contrato técnico conocido del environment y las variables que deberán verificarse en runtime.

---

## 1. Alcance y fuentes

El segundo problema del reto utiliza el robot **Fetch Mobile Manipulator** en la tarea **Reach**: mover el efector final del brazo a una posición objetivo aleatoria dentro de su workspace. El enunciado recomienda explícitamente la variante `FetchReachDense-v4`, con recompensa densa, y el objetivo académico común es maximizar la recompensa acumulada considerando estabilidad y tiempo de entrenamiento.

Fuentes técnicas utilizadas:

1. `enunciado-reto.md` del repositorio.
2. `lineamientos-transversales.md` del repositorio.
3. `Gymnasium_Robotics.ipynb` como referencia técnica de instalación, creación y render del environment.
4. Gymnasium Robotics — documentación oficial de Fetch Reach.
5. Código fuente oficial `gymnasium_robotics/envs/fetch/reach.py`.
6. Código base oficial `gymnasium_robotics/envs/fetch/fetch_env.py`.
7. Código base oficial `gymnasium_robotics/envs/robot_env.py`.
8. Registro oficial de environments en `gymnasium_robotics/__init__.py`.
9. Modelo MuJoCo oficial `fetch/reach.xml` y `fetch/robot.xml`.

### 1.1 Regla de versión

El reto fija el **ID del environment** como `FetchReachDense-v4`, pero no fija todavía una versión concreta del paquete `gymnasium-robotics`.

La rama `main` upstream consultada durante esta caracterización reporta `gymnasium_robotics.__version__ == 1.4.2`. Ese dato es una **referencia de caracterización**, no el pin definitivo del proyecto.

La versión final de Gymnasium Robotics, Gymnasium, MuJoCo y Stable-Baselines3 deberá fijarse en el DWP después de validar una combinación funcional en ejecución local y Google Colab.

**Regla:** cualquier diferencia entre esta ficha y el environment realmente instalado debe bloquear un entrenamiento largo hasta identificar, reproducir y documentar la causa.

---

## 2. Resumen ejecutivo del environment

| Propiedad | Contrato conocido |
| --- | --- |
| Environment ID del reto | `FetchReachDense-v4` |
| Familia | Gymnasium Robotics — Fetch |
| Tarea | Reach |
| Motor físico | MuJoCo |
| Robot | Fetch Mobile Manipulator |
| Brazo físico | 7 DoF |
| Efector final | Pinza paralela de dos dedos |
| Control del agente | Desplazamientos cartesianos del gripper |
| Cinemática inversa | Resuelta internamente por MuJoCo mediante mocap/control del modelo |
| Action space | `Box(-1.0, 1.0, shape=(4,), dtype=float32)` |
| Dimensiones efectivas de movimiento | 3 (`dx`, `dy`, `dz`) |
| Cuarta acción | Control de gripper, pero **sin efecto** en Reach porque `block_gripper=True` |
| Observation space | `Dict` goal-aware |
| `observation` | `(10,)`, `float64` |
| `achieved_goal` | `(3,)`, `float64` |
| `desired_goal` | `(3,)`, `float64` |
| Reward | Dense: negativo de la distancia euclídea al objetivo |
| Valor óptimo instantáneo de reward | `0.0` |
| Success threshold | distancia euclídea `< 0.05 m` |
| Señal de éxito | `info["is_success"]` |
| Horizonte por defecto | 50 pasos |
| `terminated` | `False` |
| `truncated` | `True` al alcanzar el `TimeLimit` de 50 pasos |
| Frecuencia de control | 25 Hz |
| MuJoCo timestep | `0.002 s` |
| Substeps por acción | 20 |
| Tiempo por decisión | `0.04 s` |
| Duración simulada de 50 pasos | `2.0 s` |
| Render modes | `human`, `rgb_array` |
| Render FPS | 25 |
| Tamaño de render por defecto | `480 x 480` |
| Aleatoriedad principal | objetivo XYZ aleatorio en cada reset |
| Objetos manipulables | ninguno |

---

## 3. Objetivo físico de la tarea

El agente debe aprender a controlar el efector final para que alcance una coordenada objetivo tridimensional y permanezca cerca de ella.

La secuencia conceptual es:

1. observar la posición actual del gripper;
2. observar la posición objetivo;
3. calcular implícitamente una dirección de desplazamiento;
4. aplicar pequeños movimientos cartesianos `dx`, `dy`, `dz`;
5. reducir progresivamente la distancia al objetivo;
6. entrar en la región de éxito de radio `0.05 m`;
7. mantener el efector dentro de esa región durante la tarea continua.

No existe un objeto que deba agarrarse, empujarse o levantarse. Por eso la apertura/cierre de la pinza no participa funcionalmente en la resolución de Reach.

---

## 4. Sistema físico simulado

### 4.1 Robot Fetch

Fetch es un manipulador móvil con un brazo de 7 grados de libertad y una pinza paralela de dos dedos. En este environment la base permanece fija y el agente **no controla directamente los siete joints del brazo**: ordena desplazamientos cartesianos del efector final y el modelo MuJoCo resuelve internamente la configuración articular necesaria.

Los siete joints principales del brazo físico son:

| Joint | Tipo / rango XML |
| --- | --- |
| `robot0:shoulder_pan_joint` | hinge, `[-1.6056, 1.6056] rad` |
| `robot0:shoulder_lift_joint` | hinge, `[-1.221, 1.518] rad` |
| `robot0:upperarm_roll_joint` | hinge, sin límite explícito |
| `robot0:elbow_flex_joint` | hinge, `[-2.251, 2.251] rad` |
| `robot0:forearm_roll_joint` | hinge, sin límite explícito |
| `robot0:wrist_flex_joint` | hinge, `[-2.16, 2.16] rad` |
| `robot0:wrist_roll_joint` | hinge, sin límite explícito |

Estos límites pertenecen a la física interna. **No son hiperparámetros RL ni componentes directos del action space.**

### 4.2 Base y estado inicial

La base del robot permanece fija aproximadamente en:

```text
(x, y, z) = (0.405, 0.48, 0.0) m
```

El código inicializa los slides de la base con:

```text
robot0:slide0 = 0.4049
robot0:slide1 = 0.48
robot0:slide2 = 0.0
```

La documentación oficial de Reach establece que, después del setup, el gripper queda aproximadamente en:

```text
(x, y, z) = (1.3419, 0.7491, 0.5550) m
```

con orientación fija de mocap expresada por el quaternion usado por el environment:

```text
[1.0, 0.0, 1.0, 0.0]
```

La configuración articular necesaria para posicionar el efector se resuelve internamente durante el setup.

### 4.3 Gripper bloqueado

Reach define:

```text
block_gripper = True
has_object = False
```

Los joints de los dedos existen físicamente y tienen rango aproximadamente `[0, 0.05] m`, pero el callback del environment fuerza sus posiciones a `0.0` durante la simulación.

Consecuencia:

- el action space conserva 4 dimensiones por compatibilidad con la familia Fetch;
- la acción 3 asociada al gripper no produce control funcional;
- las posiciones/velocidades de los dedos que aparecen en la observación deberían permanecer prácticamente constantes.

**Regla baseline:** no eliminar la cuarta acción mediante un wrapper sin justificarlo en el DWP. El contrato original del environment es de cuatro acciones.

---

## 5. Espacio de acciones

### 5.1 Contrato general

```text
Box(low=-1.0, high=1.0, shape=(4,), dtype=float32)
```

El environment recorta cualquier acción fuera de rango antes de aplicarla:

```python
action = np.clip(action, action_space.low, action_space.high)
```

### 5.2 Mapa completo de acciones

| idx | Acción | Dominio entregado por la política | Transformación interna | Efecto |
| ---: | --- | --- | --- | --- |
| 0 | desplazamiento `dx` | `[-1, 1]` | `dx * 0.05` | objetivo mocap X |
| 1 | desplazamiento `dy` | `[-1, 1]` | `dy * 0.05` | objetivo mocap Y |
| 2 | desplazamiento `dz` | `[-1, 1]` | `dz * 0.05` | objetivo mocap Z |
| 3 | apertura/cierre gripper | `[-1, 1]` | duplicado a dos dedos | anulado porque `block_gripper=True` |

### 5.3 Escalado cartesiano

El código aplica:

```python
pos_ctrl = action[:3]
pos_ctrl *= 0.05
```

Por tanto, cada decisión puede solicitar un cambio máximo de objetivo mocap de:

```text
±0.05 m = ±5 cm
```

por cada eje cartesiano.

Esto **no significa** que el efector físico recorra necesariamente 5 cm por paso: es la magnitud máxima de la orden cartesiana enviada al mocap/control; las dinámicas y restricciones articulares determinan el movimiento resultante.

### 5.4 Orientación del efector

La orientación no es una acción. El environment fija:

```text
rot_ctrl = [1.0, 0.0, 1.0, 0.0]
```

Por tanto, la política resuelve Reach mediante traslación XYZ; no aprende orientación de muñeca ni pose del gripper.

---

## 6. Espacio de observaciones

### 6.1 Contrato general

Fetch Reach usa la API multi-goal de Gymnasium Robotics:

```text
Dict(
  observation:  Box(-inf, inf, shape=(10,), dtype=float64),
  achieved_goal: Box(-inf, inf, shape=(3,), dtype=float64),
  desired_goal:  Box(-inf, inf, shape=(3,), dtype=float64)
)
```

No es un vector plano. Esta característica es relevante para la política/arquitectura del algoritmo elegido.

### 6.2 `observation` — 10 variables

| idx | Variable | Fuente | Unidad / interpretación |
| ---: | --- | --- | --- |
| 0 | gripper X | site `robot0:grip` | posición global m |
| 1 | gripper Y | site `robot0:grip` | posición global m |
| 2 | gripper Z | site `robot0:grip` | posición global m |
| 3 | joint del dedo derecho | `r_gripper_finger_joint` | posición, normalmente bloqueada |
| 4 | joint del dedo izquierdo | `l_gripper_finger_joint` | posición, normalmente bloqueada |
| 5 | velocidad lineal gripper X | site `robot0:grip` | valor escalado por `dt` |
| 6 | velocidad lineal gripper Y | site `robot0:grip` | valor escalado por `dt` |
| 7 | velocidad lineal gripper Z | site `robot0:grip` | valor escalado por `dt` |
| 8 | velocidad dedo derecho | joint | valor escalado por `dt`, normalmente ~0 |
| 9 | velocidad dedo izquierdo | joint | valor escalado por `dt`, normalmente ~0 |

### 6.3 Nota importante sobre las velocidades

El código obtiene las velocidades físicas y las multiplica por:

```text
dt = n_substeps * mujoco_timestep
   = 20 * 0.002
   = 0.04 s
```

Por eso las componentes de velocidad entregadas en la observación son **valores escalados por el tiempo de un step**, no deben interpretarse sin más como la velocidad bruta de MuJoCo.

El DWP deberá conservar el contrato tal como lo entrega el environment; no se debe “corregir” o reescalar silenciosamente.

### 6.4 `achieved_goal`

```text
shape = (3,)
```

Contiene exactamente la posición cartesiana actual del efector final:

```text
[x_gripper, y_gripper, z_gripper]
```

Como `has_object=False`, el goal alcanzado es la posición del propio gripper.

### 6.5 `desired_goal`

```text
shape = (3,)
```

Contiene:

```text
[x_target, y_target, z_target]
```

correspondiente a la meta aleatoria del episodio.

### 6.6 Redundancia útil

Las coordenadas XYZ del gripper aparecen simultáneamente en:

- `observation[0:3]`;
- `achieved_goal`.

Esto es parte del contrato GoalEnv y no debe eliminarse sin una decisión explícita.

### 6.7 Implicación para Stable-Baselines3

El environment entrega un `spaces.Dict`. Si se utiliza TD3 con Stable-Baselines3 —hipótesis transversal inicial— la implementación debe decidir explícitamente entre:

- una política compatible con observaciones diccionario, como `MultiInputPolicy`; o
- un wrapper de flattening documentado y validado.

No se debe asumir que `MlpPolicy` puede consumir directamente el `Dict` original.

El DWP deberá fijar esta decisión antes del smoke training.

---

## 7. Objetivo y distribución del goal

### 7.1 Muestreo

Como `has_object=False`, cada reset ejecuta conceptualmente:

```python
goal = initial_gripper_position + Uniform(-target_range, target_range, size=3)
```

con:

```text
target_range = 0.15 m
```

Por tanto, cada coordenada del goal se desplaza independientemente hasta ±15 cm alrededor de la posición inicial del gripper.

### 7.2 Rango cartesiano esperado del target

Partiendo de `[1.3419, 0.7491, 0.5550] m`, el cubo de muestreo esperado es aproximadamente:

| Eje | Mínimo | Máximo |
| --- | ---: | ---: |
| X | `1.1919 m` | `1.4919 m` |
| Y | `0.5991 m` | `0.8991 m` |
| Z | `0.4050 m` | `0.7050 m` |

La distancia máxima entre la posición inicial nominal y un target dentro de las esquinas del cubo es:

```text
sqrt(0.15² + 0.15² + 0.15²) ≈ 0.2598 m
```

Este valor caracteriza únicamente la distribución inicial de targets. No es un límite global de distancia durante un episodio porque la política puede mover el gripper lejos de la meta.

### 7.3 Target visual

El sitio MuJoCo `target0` se reposiciona en render para visualizar la meta. El archivo `reach.xml` define el target como un site esférico rojo.

---

## 8. Recompensa

## 8.1 Reward dense — `FetchReachDense-v4`

Definamos:

```text
p = achieved_goal = posición actual del gripper
g = desired_goal  = posición objetivo
d = ||p - g||₂
```

La reward por paso es:

```text
r = -d
```

Es decir:

```text
reward = -EuclideanDistance(achieved_goal, desired_goal)
```

### 8.2 Propiedades del reward dense

- mejor valor posible: `0.0`;
- cuanto más lejos esté el gripper, más negativa es la reward;
- cada reducción de distancia mejora inmediatamente la señal;
- no existe bonus discontinuo adicional al alcanzar el success threshold;
- no existe penalización explícita por magnitud de acción, velocidad o tiempo más allá del efecto acumulado de la distancia por paso.

Esto hace que la reward esté alineada directamente con el objetivo geométrico Reach.

### 8.3 Reward y reward acumulado

El objetivo académico usa recompensa acumulada:

```text
R_episode = Σ r_t
```

Con horizonte de 50 pasos, una política que llegue rápidamente y permanezca cerca del target tendrá un retorno menos negativo —más cercano a cero— que una política que permanezca alejada.

Por tanto, para Fetch Reach:

```text
mayor retorno = mejor
0 es el techo teórico por step
```

### 8.4 Variante sparse de referencia

El environment hermano `FetchReach-v4` usa:

```text
reward = 0   si d < 0.05 m
reward = -1  en otro caso
```

El reto selecciona explícitamente la variante **Dense**, por lo que `FetchReach-v4` sparse solo sirve como referencia conceptual y no debe sustituir al environment requerido.

### 8.5 API vectorizable

`compute_reward(achieved_goal, desired_goal, info)` soporta el contrato GoalEnv y puede operar sobre goals individuales o arrays compatibles. Esto es relevante para técnicas goal-conditioned como HER.

Sin embargo, los lineamientos del proyecto prohíben añadir HER al primer baseline. HER solo podría considerarse posteriormente mediante evidencia y refinamiento del DWP.

---

## 9. Criterio de éxito

### 9.1 Success oficial por step

La implementación usa:

```text
distance_threshold = 0.05 m
```

Y define:

```python
is_success = distance(achieved_goal, desired_goal) < 0.05
```

La comparación es **estrictamente menor que 5 cm**.

### 9.2 `info["is_success"]`

En cada `step()` el environment devuelve:

```python
info = {
    "is_success": ...
}
```

La implementación actual produce un valor numérico compatible con `0.0/1.0` (`np.float32`) aunque conceptualmente representa un booleano.

Para análisis robusto se puede interpretar como:

```python
success = bool(info["is_success"] > 0.5)
```

### 9.3 Tarea continua: éxito no implica terminar

Al alcanzar el goal:

```text
terminated = False
```

El episodio continúa. Por ello una política puede:

1. entrar en la región de éxito;
2. salir nuevamente;
3. volver a entrar antes del TimeLimit.

Esto es coherente con la descripción de Reach como tarea continua: no basta conceptualmente con tocar el target; interesa permanecer cerca.

### 9.4 Métricas de éxito recomendadas para diagnóstico

El DWP deberá decidir cuál será la métrica principal de éxito por episodio y mantenerla constante. Para evitar ocultar éxitos transitorios o inestabilidad se recomienda registrar simultáneamente:

- `success_any`: alcanzó la región al menos una vez;
- `success_final`: estaba dentro de la región al último step;
- `success_fraction`: fracción de steps del episodio dentro de `< 0.05 m`;
- `time_to_first_success`: primer timestep exitoso, si existe;
- `min_distance`: menor distancia observada en el episodio.

La **tasa de éxito oficial reportada** deberá quedar definida de forma explícita en el DWP antes de la evaluación final. No se debe cambiar su semántica después de observar los resultados.

---

## 10. Estado inicial y aleatoriedad

### 10.1 Qué se reinicia

`reset(seed=...)`:

1. reinicia los buffers y estado dinámico de MuJoCo;
2. restaura `qpos` y `qvel` iniciales;
3. restaura el estado base del robot;
4. no randomiza objetos porque `has_object=False`;
5. genera un nuevo `desired_goal` XYZ usando el PRNG del environment;
6. devuelve la observación inicial y un `info` vacío.

### 10.2 Aleatoriedad principal

La fuente de variación relevante para Reach es la posición del target:

```text
Uniform(-0.15, +0.15) por eje
```

La posición inicial nominal del gripper permanece fija después del reset correcto de v4.

### 10.3 Seeds

El desarrollo deberá separar como mínimo:

- seed principal de entrenamiento;
- seeds de validación/tuning;
- seeds de evaluación final;
- seed del environment;
- seed del action space cuando aplique;
- seed del framework RL.

Las seeds finales no deben utilizarse para seleccionar hiperparámetros.

---

## 11. Ciclo temporal y fin del episodio

### 11.1 Step API

```python
obs, reward, terminated, truncated, info = env.step(action)
```

### 11.2 Continuing task

La clase base define:

```text
terminated = False
truncated  = False
```

por lógica interna de la tarea.

El wrapper `TimeLimit` registrado por Gymnasium es quien establece `truncated=True` al llegar al límite temporal.

### 11.3 Horizonte

El registro oficial de `FetchReachDense-v4` usa:

```text
max_episode_steps = 50
```

El DWP deberá comprobarlo en runtime mediante:

```python
print(env.spec.max_episode_steps)
```

### 11.4 Frecuencia de control

El XML define:

```text
mujoco timestep = 0.002 s
```

Reach usa:

```text
n_substeps = 20
```

Entonces:

```text
dt por step = 0.002 * 20 = 0.04 s
control frequency = 1 / 0.04 = 25 Hz
```

Un episodio de 50 decisiones representa aproximadamente:

```text
50 * 0.04 = 2.0 s simulados
```

---

## 12. Rendering y cámaras

### 12.1 Modos soportados

La clase base declara:

```text
human
rgb_array
```

No se debe asumir `depth_array` para este environment sin verificar la versión runtime.

### 12.2 Render FPS

```text
render_fps = 25
```

coherente con la frecuencia de control.

### 12.3 Resolución por defecto

La clase base usa:

```text
width = 480
height = 480
```

salvo override en creación del environment.

### 12.4 Cámara Fetch por defecto

La configuración base Fetch incluye aproximadamente:

```text
distance = 2.5
azimuth = 132°
elevation = -14°
lookat = [1.3, 0.75, 0.55]
```

Para video se priorizará `rgb_array` y no se renderizará cada step durante el entrenamiento completo si esto degrada significativamente el rendimiento.

---

## 13. Parámetros internos del environment

`MujocoFetchReachEnv` inicializa la clase Fetch base con:

| Parámetro | Valor Reach v4 | Función |
| --- | --- | --- |
| `has_object` | `False` | no existe objeto manipulable |
| `block_gripper` | `True` | anula apertura/cierre de dedos |
| `n_substeps` | `20` | pasos MuJoCo por acción |
| `gripper_extra_height` | `0.2` | setup inicial del gripper |
| `target_in_the_air` | `True` | parámetro familiar; con `has_object=False`, el goal se muestrea directamente en XYZ |
| `target_offset` | `0.0` | offset adicional del target |
| `obj_range` | `0.15` | no relevante sin objeto |
| `target_range` | `0.15` | rango de goal por eje |
| `distance_threshold` | `0.05` | umbral de éxito |
| `reward_type` | `dense` para el ID del reto | fórmula de reward |
| `initial_qpos.slide0` | `0.4049` | posición base X |
| `initial_qpos.slide1` | `0.48` | posición base Y |
| `initial_qpos.slide2` | `0.0` | posición base Z |

Estos valores son **parámetros de la definición del environment**, no hiperparámetros del agente. El baseline no debe modificarlos.

---

## 14. Variables configurables del proyecto

La configuración del notebook/DWP deberá centralizar como mínimo:

| Variable | Tipo | Valor baseline / dominio | Estado |
| --- | --- | --- | --- |
| `ENV_ID` | str | `FetchReachDense-v4` | fijado por reto |
| `REWARD_TYPE` | str | `dense` | fijado por ID |
| `MAX_EPISODE_STEPS` | int | `50` esperado | verificar runtime |
| `ACTION_DIM` | int | `4` | fijo |
| `EFFECTIVE_CARTESIAN_DIMS` | int | `3` | fijo |
| `ACTION_LOW` | float | `-1.0` | fijo |
| `ACTION_HIGH` | float | `1.0` | fijo |
| `ACTION_POSITION_SCALE` | float | `0.05 m` | fijo environment |
| `OBS_VECTOR_DIM` | int | `10` | fijo |
| `ACHIEVED_GOAL_DIM` | int | `3` | fijo |
| `DESIRED_GOAL_DIM` | int | `3` | fijo |
| `SUCCESS_THRESHOLD` | float | `0.05 m` | fijo environment |
| `TARGET_RANGE` | float | `0.15 m` por eje | fijo environment |
| `N_SUBSTEPS` | int | `20` | fijo environment |
| `MUJOCO_TIMESTEP` | float | `0.002 s` esperado | verificar runtime |
| `CONTROL_DT` | float | `0.04 s` | derivado |
| `CONTROL_FREQUENCY` | float | `25 Hz` | derivado |
| `RENDER_MODE_TRAIN` | None/str | `None` | configuración proyecto |
| `RENDER_MODE_VIDEO` | str | `rgb_array` | configuración proyecto |
| `SEED` | int | por definir en DWP | hiperparámetro experimental |
| `ALGORITHM` | str | hipótesis inicial `TD3` | por confirmar DWP |
| `POLICY` | str | por definir según Dict/wrapper | pendiente DWP |
| `TOTAL_TIMESTEPS` | int | por definir | pendiente DWP |
| `LEARNING_RATE` | float | por definir | pendiente DWP |
| `GAMMA` | float | por definir | pendiente DWP |
| `EVAL_EPISODES` | int | mínimo `10` final | regla transversal |
| `DEVICE` | str | `cpu`, `cuda`, eventualmente `mps` si se valida | pendiente runtime |
| `PROJECT_ROOT` | path | única raíz | obligatorio |
| `MODEL_DIR` | path | derivada | obligatorio |
| `VIDEO_DIR` | path | derivada | obligatorio |
| `RESULTS_DIR` | path | derivada | obligatorio |

No se deben duplicar estas constantes en celdas distintas.

---

## 15. Restricciones del problema

### 15.1 Restricciones duras del environment

1. Action space de cuatro componentes en `[-1,1]`.
2. Acciones fuera de rango son recortadas.
3. Solo los primeros tres componentes producen desplazamiento cartesiano efectivo.
4. El cuarto componente no mueve la pinza porque el gripper está bloqueado.
5. Cada componente XYZ se escala por `0.05` antes de aplicar el control.
6. La orientación del gripper es fija desde el punto de vista de la política.
7. Los límites articulares del robot restringen físicamente el movimiento aunque no sean acciones directas.
8. No existe objeto manipulable.
9. El target cambia en cada reset dentro del cubo ±0.15 m por eje.
10. Success por step requiere distancia estrictamente `<0.05 m`.
11. Alcanzar success no termina el episodio.
12. El episodio se trunca por `TimeLimit` a 50 pasos por defecto.
13. Reward dense es `-distance`, sin bonus ni shaping adicional.
14. Observation space es `Dict`, no un vector plano.

### 15.2 Restricciones académicas/transversales

1. Debe utilizarse `FetchReachDense-v4` salvo excepción aprobada/documentada.
2. Debe conservarse recompensa densa.
3. El algoritmo debe pertenecer a la lista permitida por el enunciado.
4. El reto global debe usar mínimo tres algoritmos distintos.
5. Hipótesis inicial transversal: TD3 para Fetch Reach.
6. No HER, reward shaping adicional ni curriculum en el primer baseline.
7. Entrenamiento y evaluación separados.
8. Evaluación final de mínimo 10 episodios.
9. Debe existir baseline aleatorio comparable.
10. Deben producirse 3 métricas principales, 3 gráficas y 2 videos.
11. Debe existir un único artefacto canónico final.
12. El desarrollo sustantivo deberá ejecutarse mediante DWP Full/plan conforme a las reglas del repositorio.
13. Ningún cambio directo sobre `main`.

---

## 16. Telemetría recomendada

Además de las métricas finales obligatorias, para entender el aprendizaje conviene registrar:

| Variable | Uso |
| --- | --- |
| `episode_return` | objetivo RL acumulado |
| `episode_length` | horizonte real |
| `distance_to_goal` | progreso geométrico principal |
| `min_distance` | mejor aproximación del episodio |
| `final_distance` | estabilidad al final |
| `is_success` por step | success oficial del environment |
| `success_any` | alcanzó target al menos una vez |
| `success_final` | terminó dentro del target |
| `success_fraction` | estabilidad dentro del umbral |
| `time_to_first_success` | rapidez para alcanzar la meta |
| `action_xyz_norm` | intensidad del control cartesiano |
| `action_gripper` | verificar dimensión nula / saturación innecesaria |
| `training_timesteps` | presupuesto de interacción |
| `wall_clock_training_time` | costo real |
| `eval_reward_mean/std` | rendimiento y estabilidad |
| `success_rate` | capacidad funcional |

Estas variables son telemetría; no deben añadirse automáticamente al observation space del agente.

---

## 17. Contrato de validación previo al entrenamiento

Antes de fijar hiperparámetros finales o ejecutar entrenamiento largo, el DWP/notebook deberá validar programáticamente el runtime real.

### 17.1 Aserciones mínimas

```python
assert env.spec.id == "FetchReachDense-v4"
assert env.action_space.shape == (4,)
assert env.action_space.low.min() == -1.0
assert env.action_space.high.max() == 1.0

obs, info = env.reset(seed=SEED)
assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
assert obs["observation"].shape == (10,)
assert obs["achieved_goal"].shape == (3,)
assert obs["desired_goal"].shape == (3,)
```

### 17.2 Datos que deben imprimirse/registrarse

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
env.unwrapped.target_range
env.unwrapped.block_gripper
env.unwrapped.has_object
env.unwrapped.reward_type
```

### 17.3 Smoke test mínimo del environment

1. `gym.make("FetchReachDense-v4")` sin error;
2. `reset(seed=...)` sin error;
3. validar Dict y shapes;
4. ejecutar varios `step()`;
5. comprobar reward finita y `<= 0` dentro de condiciones normales;
6. comprobar presencia de `info["is_success"]`;
7. comprobar que la acción 4 no cambia el gripper de manera funcional;
8. verificar `terminated=False` durante la tarea;
9. verificar `truncated=True` al horizonte efectivo;
10. render `rgb_array` válido;
11. `close()` correcto;
12. repetir reset con la misma seed y validar reproducibilidad del target;
13. verificar que seeds distintas producen goals distintos;
14. guardar caracterización runtime junto al experimento.

---

## 18. Implicaciones para el diseño del agente

Esta sección no fija aún el DWP; traduce la física y la API a decisiones que deberán tomarse antes del entrenamiento.

### 18.1 Control continuo de baja dimensión efectiva

El action space formal tiene dimensión 4, pero solo 3 componentes afectan la tarea. Esto favorece algoritmos de control continuo como TD3, SAC o DDPG.

Los lineamientos establecen **TD3** como hipótesis inicial para Fetch Reach por su control continuo preciso y reducción del sesgo de sobreestimación.

### 18.2 Observation space tipo Dict

Este es probablemente el punto de integración más importante para el DWP.

La política debe consumir:

```text
observation + achieved_goal + desired_goal
```

o una representación equivalente documentada.

Antes del entrenamiento se deberá probar la compatibilidad real de TD3 + Stable-Baselines3 con la política elegida (`MultiInputPolicy` o wrapper de flattening).

### 18.3 Reward bien alineada

A diferencia de AdroitHandDoor, no existe una anomalía conocida del signo en la reward de Reach Dense. La señal `-distance` entrega información en cada step y apunta directamente a reducir el error cartesiano.

Por ello el baseline **no necesita reward shaping adicional**.

### 18.4 Cuarta acción irrelevante

Una política puede desperdiciar capacidad aprendiendo valores para la cuarta componente que el environment ignora.

El baseline deberá conservar el action space original. Si posteriormente se considera un wrapper de tres acciones, será una modificación del contrato y requerirá evidencia, validación y refinamiento del DWP.

### 18.5 Tarea continua

Un episodio exitoso no debería analizarse únicamente como “alguna vez estuvo a menos de 5 cm”. Para demostrar comportamiento aprendido conviene observar también permanencia cerca del target (`success_fraction`) y distancia final.

### 18.6 Target aleatorio

La policy debe generalizar a objetivos distintos dentro del cubo de ±15 cm. Evaluar siempre una única seed no es suficiente.

---

## 19. Configuración baseline del environment propuesta

Esta configuración caracteriza el primer smoke del environment. **No fija todavía los hiperparámetros TD3.**

```yaml
environment:
  id: FetchReachDense-v4
  reward: dense
  max_episode_steps_expected: 50

  action:
    shape: [4]
    low: -1.0
    high: 1.0
    cartesian_dimensions: 3
    position_scale_m: 0.05
    gripper_dimension_effective: false

  observation:
    type: Dict
    observation_shape: [10]
    achieved_goal_shape: [3]
    desired_goal_shape: [3]

  goal:
    target_range_m: 0.15
    success_distance_m: 0.05

  simulation:
    mujoco_timestep_s_expected: 0.002
    n_substeps_expected: 20
    control_dt_s_expected: 0.04
    control_frequency_hz_expected: 25

  robot:
    has_object: false
    block_gripper: true

  rendering:
    training: null
    video: rgb_array
    fps_expected: 25
```

**Regla:** cualquier diferencia observada frente a `env.spec` o `env.unwrapped` se debe explicar antes del entrenamiento largo.

---

## 20. Decisiones pendientes para el DWP de FetchReachDense

La caracterización deja identificadas las decisiones que deberá cerrar el siguiente DWP:

1. pin exacto de `gymnasium-robotics`, Gymnasium y MuJoCo;
2. compatibilidad reproducible local/Colab;
3. confirmación runtime de los 50 pasos, 25 Hz y parámetros del environment;
4. algoritmo final dentro de la estrategia global —hipótesis actual TD3—;
5. `MultiInputPolicy` vs flattening para el `Dict` observation space;
6. confirmar si se conserva la acción 4 inútil sin wrapper —baseline recomendado: sí—;
7. arquitectura de policy/Q-networks;
8. normalización de observaciones/goals si fuese necesaria;
9. seed de entrenamiento y conjuntos separados de tuning/final evaluation;
10. presupuesto de timesteps;
11. learning rate, gamma, tau, batch size y replay buffer;
12. policy delay, target policy noise y noise clip propios de TD3;
13. exploration/action noise;
14. learning starts y frecuencia de actualización;
15. frecuencia de evaluación/checkpoints;
16. semántica oficial de success por episodio (`success_final`, `success_any` u otra) y métricas auxiliares;
17. objetivo cuantitativo interno de desempeño;
18. criterio para seleccionar el único modelo final;
19. estrategia de baseline aleatorio comparable;
20. estrategia de video `Training Process` y `Trained Agent`;
21. condiciones de ejecución local Mac mini vs Google Colab;
22. criterios para habilitar tuning;
23. umbral a partir del cual un cambio avanzado exigiría `dwp-refine`.

---

## 21. Checklist de caracterización

- [x] Problema físico identificado.
- [x] Environment ID correcto identificado.
- [x] Variante dense confirmada.
- [x] Robot y estrategia de control descritos.
- [x] Action space completo documentado.
- [x] Escalado cartesiano documentado.
- [x] Acción de gripper sin efecto identificada.
- [x] Observation space tipo Dict documentado.
- [x] 10 variables de `observation` documentadas.
- [x] `achieved_goal` y `desired_goal` documentados.
- [x] Reward dense documentada con fórmula.
- [x] Sparse documentada como referencia.
- [x] Success threshold identificado.
- [x] Naturaleza continua del task documentada.
- [x] Distribución del target documentada.
- [x] Estado inicial nominal documentado.
- [x] Horizonte documentado.
- [x] Frecuencia de control y substeps documentados.
- [x] Render documentado.
- [x] Restricciones físicas y académicas documentadas.
- [x] Telemetría diagnóstica propuesta.
- [x] Contrato de smoke/runtime definido.
- [x] Implicación del Dict para TD3/SB3 identificada.
- [ ] Versiones finales fijadas — pendiente del DWP/runtime.
- [ ] Algoritmo/policy definitivos — pendiente del DWP.
- [ ] Hiperparámetros fijados — pendiente del DWP.
- [ ] Semántica final de success rate — pendiente del DWP.

---

## 22. Referencias técnicas

- Enunciado del Reto II: `enunciado-reto.md`.
- Lineamientos: `lineamientos-transversales.md`.
- Notebook del profesor: `Gymnasium_Robotics.ipynb`.
- Gymnasium Robotics — Fetch environments: https://robotics.farama.org/envs/fetch/
- Gymnasium Robotics — Fetch Reach: https://robotics.farama.org/envs/fetch/reach/
- Fuente Reach: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/reach.py
- Base Fetch: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/fetch_env.py
- Base RobotEnv: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/robot_env.py
- Registro de environments: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/__init__.py
- MuJoCo Reach XML: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/assets/fetch/reach.xml
- Fetch robot XML: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/assets/fetch/robot.xml

---

## 23. Regla de mantenimiento

Esta ficha es contexto técnico para el DWP, no un sustituto de la inspección runtime.

Si el environment instalado contradice una variable aquí documentada:

1. detener el entrenamiento largo;
2. registrar versiones exactas;
3. reproducir la diferencia con un smoke mínimo;
4. contrastarla con la fuente upstream de esa versión;
5. decidir si el problema es documentación, versión o configuración;
6. actualizar esta ficha si corresponde;
7. refinar el DWP antes de continuar si la diferencia afecta observaciones, acciones, reward, success, horizonte o reproducibilidad.

No se modificará el environment para “hacerlo coincidir” con esta ficha sin una decisión técnica explícita.