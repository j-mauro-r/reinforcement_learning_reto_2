# Ficha técnica — AdroitHandDoor

> **Ambiente objetivo:** `AdroitHandDoor-v1`  
> **Proyecto:** Reto II — Reinforcement Learning / Gymnasium Robotics  
> **Propósito:** caracterizar el ambiente antes de definir arquitectura del agente, hiperparámetros y plan de entrenamiento. Este documento establece el contrato técnico que deberá respetar la configuración del ambiente y sirve como fuente de verdad para el desarrollo del modelo `1-AdroitHandDoor`.

---

## 1. Alcance y fuentes

Esta ficha caracteriza el ambiente solicitado en `enunciado-reto.md`: una mano Adroit debe manipular la manija, liberar el pestillo y abrir la puerta. El entregable académico exige maximizar la recompensa acumulada considerando también estabilidad y tiempo de entrenamiento.

Fuentes técnicas utilizadas:

1. `enunciado-reto.md` del repositorio.
2. `Gymnasium_Robotics.ipynb` como referencia de creación/render del ambiente.
3. Gymnasium Robotics — documentación oficial de **Adroit Door**.
4. Código fuente oficial `gymnasium_robotics/envs/adroit_hand/adroit_door.py`.
5. Modelo MuJoCo oficial `adroit_door.xml`, `adroit_model.xml` y `adroit_assets.xml`.
6. Para el comportamiento exacto de `AdroitHandDoor-v1`, código del tag `Gymnasium-Robotics v1.3.0`.

### 1.1 Regla de versión

La ficha separa deliberadamente:

- **Contrato del ambiente `AdroitHandDoor-v1`**, que es el solicitado por el reto.
- **Cambios posteriores del upstream**, que no deben incorporarse silenciosamente.

Existe una diferencia crítica entre `v1` y versiones posteriores de Adroit Door: el término de recompensa asociado a la distancia mano–manija tenía el signo contrario en `v1`; upstream lo corrigió posteriormente. Esta diferencia se documenta en la sección de recompensas y debe verificarse en tiempo de ejecución antes de iniciar un entrenamiento costoso.

---

## 2. Resumen ejecutivo del entorno

| Propiedad | Valor / contrato |
| --- | --- |
| Environment ID del reto | `AdroitHandDoor-v1` |
| Familia | Gymnasium Robotics — Adroit Hand |
| Motor físico | MuJoCo |
| Robot | Adroit: ShadowHand + brazo libre |
| Grados de libertad controlados | 28 actuadores: 4 brazo + 24 mano/muñeca |
| Action space | `Box(-1.0, 1.0, shape=(28,), dtype=float32)` |
| Observation space | `Box(-inf, inf, shape=(39,), dtype=float64)` |
| Reward por defecto | Dense |
| Variante sparse | `AdroitHandDoorSparse-v1` |
| Criterio de éxito | ángulo de la bisagra `door_hinge >= 1.35 rad` |
| Apertura física máxima de la puerta | `1.57 rad` |
| Límite de episodio registrado para `v1` | 200 pasos |
| `terminated` por lógica de tarea | `False` |
| `truncated` | lo aplica `TimeLimit` al llegar a `max_episode_steps` |
| Render modes | `human`, `rgb_array`, `depth_array` |
| Render FPS | 100 |
| `frame_skip` | 5 |
| Aleatoriedad principal al reset | posición XYZ del marco de la puerta |
| Señal de éxito en `info` | `info["success"]` booleano |

### 2.1 Objetivo físico

El agente debe aprender una secuencia coordinada de manipulación:

1. desplazar/orientar brazo y muñeca hacia la manija;
2. posicionar la palma y los dedos alrededor de la manija;
3. actuar sobre un pestillo con fricción alta;
4. superar la resistencia de la puerta;
5. girar la bisagra hasta alcanzar al menos `1.35 rad`.

La tarea no entrega al agente una variable explícita del tipo “pestillo liberado”. El estado del pestillo y la dinámica de contacto deben ser inferidos a partir de observaciones y consecuencias de las acciones.

---

## 3. Sistema físico simulado

### 3.1 Robot

El sistema Adroit expone **28 controles continuos**:

- 4 controles del brazo libre;
- 2 controles de muñeca;
- 4 controles para el índice;
- 4 controles para el dedo medio;
- 4 controles para el anular;
- 5 controles para el meñique;
- 5 controles para el pulgar.

El estado MuJoCo incluye además articulaciones no controladas directamente por el agente, entre ellas la bisagra de la puerta y el pestillo. Por eso el vector interno `qpos` tiene 30 posiciones aunque el agente produzca 28 acciones.

### 3.2 Puerta

| Variable física | Valor |
| --- | ---: |
| Joint | `door_hinge` |
| Tipo | hinge |
| Eje | Z |
| Rango | `[0.0, 1.57] rad` |
| Damping | `1` |
| Friction loss | `2` |
| Fricción del geom principal | `1 1 1` |
| Umbral bonus 1 | `> 0.2 rad` |
| Umbral bonus 2 | `> 1.0 rad` |
| Umbral de éxito | `>= 1.35 rad` |
| Objetivo utilizado por reward cuadrático | `1.57 rad` |

### 3.3 Pestillo / manija

| Variable física | Valor |
| --- | ---: |
| Joint | `latch` |
| Tipo | hinge |
| Eje | Y |
| Rango | `[0.0, 1.8] rad` |
| Friction loss | `5` |
| Site de la manija | `S_handle` |
| Site de referencia de la palma | `S_grasp` |

El pestillo tiene una resistencia considerable respecto a otras articulaciones. Esto hace que “llegar a la manija” no sea equivalente a resolver la tarea: la política debe generar una interacción de contacto capaz de liberar el mecanismo y luego mover la puerta.

### 3.4 Contactos y fricción

El modelo MuJoCo utiliza, como defaults generales:

- `joint damping = 0.05` para articulaciones que no sobrescriben el valor;
- `joint frictionloss = 0.001` por defecto;
- `geom friction = (1, 0.5, 0.01)` por defecto;
- contactos específicos entre partes de los dedos para estabilizar la simulación;
- articulaciones del brazo con `damping = 20`;
- bisagra y pestillo con parámetros propios descritos arriba.

Estos parámetros forman parte de la física del ambiente y **no son hiperparámetros del algoritmo RL**. No deben modificarse en el baseline salvo que exista una decisión experimental explícita y documentada.

---

## 4. Espacio de acciones

### 4.1 Contrato general

```text
Box(low=-1.0, high=1.0, shape=(28,), dtype=float32)
```

Cada componente de acción es continuo. Antes de ejecutar la simulación, el entorno realiza:

```python
a = clip(a, -1.0, 1.0)
actual_control = act_mean + a * act_rng
```

donde:

```text
act_mean = (control_min + control_max) / 2
act_rng  = (control_max - control_min) / 2
```

Por tanto:

- `-1` representa el extremo inferior del rango del actuador;
- `0` representa el centro del rango del actuador;
- `+1` representa el extremo superior;
- valores fuera de `[-1, 1]` son recortados por el ambiente.

### 4.2 Mapa completo de acciones

| idx | Actuador / joint | Movimiento | Acción normalizada | Rango físico de referencia |
| ---: | --- | --- | --- | --- |
| 0 | `A_ARTz` / `ARTz` | Traslación lineal del brazo | `[-1,1]` | `[-0.3, 0.5] m` |
| 1 | `A_ARRx` / `ARRx` | Rotación vertical del brazo | `[-1,1]` | XML v1: `[-0.75, 0.75] rad` |
| 2 | `A_ARRy` / `ARRy` | Rotación lateral del brazo | `[-1,1]` | XML v1: `[-0.75, 0.75] rad` |
| 3 | `A_ARRz` / `ARRz` | Roll del brazo | `[-1,1]` | `[-1.0, 2.0] rad` |
| 4 | `A_WRJ1` / `WRJ1` | Desviación radial/ulnar de muñeca | `[-1,1]` | `[-0.524, 0.175] rad` |
| 5 | `A_WRJ0` / `WRJ0` | Flexión/extensión de muñeca | `[-1,1]` | `[-0.785, 0.611] rad` aprox. |
| 6 | `A_FFJ3` / `FFJ3` | Índice MCP abducción/aducción | `[-1,1]` | `[-0.436, 0.436] rad` |
| 7 | `A_FFJ2` / `FFJ2` | Índice MCP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 8 | `A_FFJ1` / `FFJ1` | Índice PIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 9 | `A_FFJ0` / `FFJ0` | Índice DIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 10 | `A_MFJ3` / `MFJ3` | Medio MCP abducción/aducción | `[-1,1]` | `[-0.436, 0.436] rad` |
| 11 | `A_MFJ2` / `MFJ2` | Medio MCP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 12 | `A_MFJ1` / `MFJ1` | Medio PIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 13 | `A_MFJ0` / `MFJ0` | Medio DIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 14 | `A_RFJ3` / `RFJ3` | Anular MCP abducción/aducción | `[-1,1]` | `[-0.436, 0.436] rad` |
| 15 | `A_RFJ2` / `RFJ2` | Anular MCP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 16 | `A_RFJ1` / `RFJ1` | Anular PIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 17 | `A_RFJ0` / `RFJ0` | Anular DIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 18 | `A_LFJ4` / `LFJ4` | Meñique CMC | `[-1,1]` | `[0, 0.698] rad` |
| 19 | `A_LFJ3` / `LFJ3` | Meñique MCP abducción/aducción | `[-1,1]` | `[-0.436, 0.436] rad` |
| 20 | `A_LFJ2` / `LFJ2` | Meñique MCP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 21 | `A_LFJ1` / `LFJ1` | Meñique PIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 22 | `A_LFJ0` / `LFJ0` | Meñique DIP flex/ext | `[-1,1]` | `[0, 1.571] rad` |
| 23 | `A_THJ4` / `THJ4` | Pulgar CMC horizontal | `[-1,1]` | `[-1.047, 1.047] rad` |
| 24 | `A_THJ3` / `THJ3` | Pulgar CMC vertical | `[-1,1]` | `[0, 1.3] rad` |
| 25 | `A_THJ2` / `THJ2` | Pulgar MCP abducción/aducción | `[-1,1]` | `[-0.26, 0.26] rad` aprox. |
| 26 | `A_THJ1` / `THJ1` | Pulgar MCP flex/ext | `[-1,1]` | `[-0.52, 0.52] rad` aprox. |
| 27 | `A_THJ0` / `THJ0` | Pulgar IP flex/ext | `[-1,1]` | `[-1.571, 0] rad` |

### 4.3 Rangos articulares y rangos de control

La tabla textual de la documentación histórica de Adroit Door presenta para `ARRx` y `ARRy` rangos más estrechos que el XML de `v1`. Sin embargo, el código de ejecución escala la acción usando `model.actuator_ctrlrange`, y el XML de `v1.3.0` define ambos actuadores en `[-0.75, 0.75]`.

**Verificación local (Gymnasium Robotics 1.3.0 / MuJoCo 3.1.6):** la columna «Rango físico de referencia» de §4.2 contiene principalmente límites articulares (`model.jnt_range`), que no siempre coinciden con `model.actuator_ctrlrange`. Por ejemplo, `THJ4` tiene límite articular `[-1.047, 1.047]`, pero su actuador `A_THJ4` usa `[-1.0, 1.0]`; `FFJ2` tiene límite articular `[0, 1.571]`, pero `A_FFJ2` usa `[0, 1.6]`. No son cambios de física ni de versión: ambos valores coexisten en los XML upstream. La transformación de acciones usa exclusivamente el rango de control. El registro completo por joint/actuador queda en `results/runtime_contract.json`.

Fuentes: [límites articulares](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.0/gymnasium_robotics/envs/assets/adroit_hand/adroit_model.xml) y [actuadores y defaults](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.0/gymnasium_robotics/envs/assets/adroit_hand/adroit_assets.xml).

**Regla para implementación:** antes del entrenamiento se debe inspeccionar el ambiente realmente instalado y registrar `env.unwrapped.model.actuator_ctrlrange`. El valor observado en runtime es el contrato efectivo de esa instalación.

---

## 5. Espacio de observaciones

### 5.1 Contrato general

```text
Box(low=-inf, high=inf, shape=(39,), dtype=float64)
```

La observación es un vector plano, no un `Dict`. Contiene:

- 27 posiciones articulares del brazo/mano seleccionadas del `qpos`;
- estado del pestillo;
- posición angular de la puerta;
- posición XYZ de la palma;
- posición XYZ de la manija;
- diferencia XYZ palma–manija;
- indicador binario de puerta abierta.

### 5.2 Mapa completo de observaciones

| idx | Variable observada | Fuente | Unidad / dominio |
| ---: | --- | --- | --- |
| 0 | `ARRx` | `qpos` | rad |
| 1 | `ARRy` | `qpos` | rad |
| 2 | `ARRz` | `qpos` | rad |
| 3 | `WRJ1` | `qpos` | rad |
| 4 | `WRJ0` | `qpos` | rad |
| 5 | `FFJ3` | `qpos` | rad |
| 6 | `FFJ2` | `qpos` | rad |
| 7 | `FFJ1` | `qpos` | rad |
| 8 | `FFJ0` | `qpos` | rad |
| 9 | `MFJ3` | `qpos` | rad |
| 10 | `MFJ2` | `qpos` | rad |
| 11 | `MFJ1` | `qpos` | rad |
| 12 | `MFJ0` | `qpos` | rad |
| 13 | `RFJ3` | `qpos` | rad |
| 14 | `RFJ2` | `qpos` | rad |
| 15 | `RFJ1` | `qpos` | rad |
| 16 | `RFJ0` | `qpos` | rad |
| 17 | `LFJ4` | `qpos` | rad |
| 18 | `LFJ3` | `qpos` | rad |
| 19 | `LFJ2` | `qpos` | rad |
| 20 | `LFJ1` | `qpos` | rad |
| 21 | `LFJ0` | `qpos` | rad |
| 22 | `THJ4` | `qpos` | rad |
| 23 | `THJ3` | `qpos` | rad |
| 24 | `THJ2` | `qpos` | rad |
| 25 | `THJ1` | `qpos` | rad |
| 26 | `THJ0` | `qpos` | rad |
| 27 | posición del pestillo `latch` | `qpos[-1]` | rad |
| 28 | posición angular puerta `door_hinge` | `qpos[door_hinge_addr]` | rad |
| 29 | palma X | site `S_grasp` | m |
| 30 | palma Y | site `S_grasp` | m |
| 31 | palma Z | site `S_grasp` | m |
| 32 | manija X | site `S_handle` | m |
| 33 | manija Y | site `S_handle` | m |
| 34 | manija Z | site `S_handle` | m |
| 35 | palma X − manija X | derivada | m |
| 36 | palma Y − manija Y | derivada | m |
| 37 | palma Z − manija Z | derivada | m |
| 38 | `door_open` | derivada de `door_hinge` | `-1.0` o `1.0` |

### 5.3 Indicador `door_open`

```text
if door_hinge > 1.0 rad:
    door_open = 1.0
else:
    door_open = -1.0
```

Este indicador **no es el criterio de éxito**. Existe una zona intermedia:

```text
1.0 < door_hinge < 1.35
```

en la que `door_open == 1`, pero `info["success"] == False`.

### 5.4 Observación omitida relevante

El vector `_get_obs()` usa `qpos[1:-2]`, por lo que la primera coordenada interna de posición (`ARTz`, traslación del brazo) no aparece directamente como una entrada separada del vector. Su efecto sí se refleja indirectamente en la posición cartesiana de la palma y en la distancia a la manija.

### 5.5 Nota sobre una etiqueta histórica

Documentación histórica ha etiquetado la observación 28 como “angular velocity”. El código fuente de `v1` obtiene ese valor de `data.qpos`, por lo que técnicamente corresponde a **posición angular de la bisagra**, no a `qvel`.

---

## 6. Recompensa

## 6.1 Variante dense — `AdroitHandDoor-v1`

El ID del reto usa recompensa densa. Para el código `v1.3.0`, la recompensa por paso es:

```text
θ = door_hinge
p = palm_position
h = handle_position
v = vector completo de qvel

r = +0.1 * ||p - h||₂
    -0.1 * (θ - 1.57)²
    -0.00001 * Σ(vᵢ²)
    + bonuses(θ)
```

Bonificaciones:

```text
if θ > 0.2:  +2
if θ > 1.0:  +8
if θ > 1.35: +10
```

Los bonus son acumulativos. Por ejemplo, al superar `1.35 rad`, el paso obtiene `2 + 8 + 10 = 20` puntos adicionales antes de sumar los términos continuos.

### 6.2 Componentes de reward

| Componente | Fórmula `v1` | Objetivo conceptual |
| --- | --- | --- |
| Distancia palma–manija | `+0.1 * ‖p-h‖₂` | históricamente pretendía acercar la mano |
| Apertura | `-0.1 * (θ-1.57)^2` | aproximar puerta a apertura máxima |
| Penalización de velocidad | `-1e-5 * Σ(qvel²)` | limitar movimiento excesivo |
| Bonus parcial 1 | `+2` si `θ > 0.2` | premiar inicio de apertura |
| Bonus parcial 2 | `+8` si `θ > 1.0` | premiar apertura avanzada |
| Bonus éxito | `+10` si `θ > 1.35` | premiar resolución |

### 6.3 Advertencia crítica: bug conocido de `v1`

El código de `AdroitHandDoor-v1` en Gymnasium Robotics `v1.3.0` usa:

```python
reward = 0.1 * np.linalg.norm(palm_pos - handle_pos)
```

Eso **aumenta la recompensa al incrementar la distancia palma–manija**, lo contrario de la intención descrita por la documentación conceptual. Upstream corrigió posteriormente este término cambiándolo a signo negativo.

Consecuencias para este reto:

- no debemos asumir que “dense” implica necesariamente que todos sus términos empujen en la dirección deseada;
- el ambiente exacto instalado debe caracterizarse antes del entrenamiento;
- no se modificará manualmente la reward del environment sin decisión académica explícita, porque hacerlo cambiaría el problema evaluado;
- la versión de `gymnasium-robotics` usada para producir el modelo final deberá quedar registrada en notebook, metadata y reporte.

### 6.4 Variante sparse de referencia

`AdroitHandDoorSparse-v1` usa:

```text
+10.0  si door_hinge >= 1.35
-0.1   en otro caso
```

No es la variante base seleccionada para este desarrollo, pero es útil como referencia para entender el criterio puro de éxito.

---

## 7. Criterios de éxito y progreso

### 7.1 Éxito oficial

```text
goal_achieved = door_hinge >= 1.35
info["success"] = goal_achieved
```

### 7.2 Hitos físicos útiles para análisis

| Estado | Condición | Interpretación |
| --- | --- | --- |
| Sin apertura relevante | `θ <= 0.2` | no se ha conseguido progreso de puerta significativo |
| Apertura inicial | `0.2 < θ <= 1.0` | puerta empieza a abrirse |
| Apertura avanzada | `1.0 < θ < 1.35` | door_open ya puede ser `1`, pero aún no hay éxito |
| Éxito | `θ >= 1.35` | `info["success"] == True` |
| Máximo físico aproximado | `θ -> 1.57` | puerta contra el extremo del rango |

Estos hitos son adecuados para análisis y visualización, pero solo `>=1.35` debe utilizarse como success rate oficial del environment.

---

## 8. Estado inicial y aleatoriedad

En cada `reset()` estándar:

- la configuración articular parte del estado inicial del modelo;
- el marco de la puerta cambia de posición;
- la aleatoriedad usa el generador del ambiente y, por tanto, puede controlarse con seed.

### 8.1 Distribución de posición de la puerta

| Coordenada | Distribución |
| --- | --- |
| X | `Uniform(-0.30, -0.20)` m |
| Y | `Uniform(0.25, 0.35)` m |
| Z | `Uniform(0.252, 0.35)` m |

Esto obliga a la política a generalizar dentro de una pequeña región espacial en vez de memorizar una única posición de la manija.

### 8.2 Estado serializable

El ambiente expone un estado reproducible mediante:

```python
{
    "qpos": np.ndarray(shape=(30,)),
    "qvel": np.ndarray(shape=(30,)),
    "door_body_pos": np.ndarray(shape=(3,)),
}
```

Puede obtenerse con:

```python
env.unwrapped.get_env_state()
```

y establecerse mediante `set_env_state(...)` o `reset(options={"initial_state_dict": ...})` según la API disponible en la versión instalada.

### 8.3 Seeds

La configuración de entrenamiento deberá separar al menos:

- seed de entrenamiento;
- seeds de evaluación;
- seed del environment/reset;
- seed del framework RL cuando aplique.

Las evaluaciones finales no deben depender de un único reset favorable.

---

## 9. Ciclo de episodio

### 9.1 Step API

Contrato Gymnasium:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

La implementación interna de Adroit Door retorna:

```text
terminated = False
truncated  = False
```

pero el wrapper Gymnasium `TimeLimit` marca `truncated=True` cuando se alcanza `max_episode_steps`.

### 9.2 Horizonte

El registro de `AdroitHandDoor-v1` utilizado por los datasets oficiales de Farama reporta:

```text
max_episode_steps = 200
```

La documentación web contiene actualmente una inconsistencia textual: una sección indica 200 y otra menciona 50. Para este proyecto, el valor efectivo debe obtenerse en runtime con:

```python
print(env.spec.max_episode_steps)
```

**Baseline esperado para `v1`: 200 pasos.**

### 9.3 El éxito no termina automáticamente el episodio

Al llegar a `door_hinge >= 1.35`, el environment establece `info["success"] = True`, pero no retorna `terminated=True`. Esto significa que la política puede seguir recibiendo reward después de resolver la tarea hasta que el wrapper alcance el límite temporal.

Implicación: durante evaluación debemos definir claramente si medimos:

- retorno completo de 200 pasos; y
- éxito alcanzado alguna vez durante el episodio.

El success rate debe considerar `any(info["success"] during episode)` para no perder un éxito transitorio.

---

## 10. Rendering y video

| Parámetro | Valor |
| --- | --- |
| `render_mode="human"` | ventana interactiva |
| `render_mode="rgb_array"` | frame RGB para video |
| `render_mode="depth_array"` | mapa de profundidad |
| `render_fps` | 100 |
| Cámara por defecto | distance `1.5`, azimuth `90°` |
| Cámara XML `fixed` | disponible en el modelo |
| Cámara XML `vil_camera` | disponible en el modelo |

Para la entrega se utilizará `rgb_array`, evitando renderizar cada frame durante el entrenamiento completo. Los videos se generarán en ejecuciones controladas de evaluación/checkpoint conforme a `lineamientos-transversales.md`.

---

## 11. Parámetros configurables del ambiente

Variables que deben centralizarse desde el inicio:

| Variable | Tipo | Valor baseline | Valores / dominio | Observación |
| --- | --- | --- | --- | --- |
| `ENV_ID` | str | `AdroitHandDoor-v1` | IDs registrados | fijado por el reto |
| `REWARD_TYPE` | str | `dense` implícito | `dense`, `sparse` | sparse usa otro ID/kwargs según versión |
| `MAX_EPISODE_STEPS` | int | `200` esperado | entero positivo | validar `env.spec` |
| `RENDER_MODE_TRAIN` | str/None | `None` | `None`, `human`, `rgb_array`, `depth_array` | `None` para velocidad |
| `RENDER_MODE_VIDEO` | str | `rgb_array` | modos soportados | para captura |
| `SEED` | int | a definir en plan | entero | seed principal |
| `FRAME_SKIP` | int | `5` | definido por environment | no modificar baseline |
| `SUCCESS_THRESHOLD` | float | `1.35` | contrato environment | rad |
| `DOOR_OPEN_FLAG_THRESHOLD` | float | `1.0` | contrato environment | rad |
| `DOOR_TARGET_ANGLE` | float | `1.57` | contrato reward | rad |
| `ACTION_LOW` | float | `-1.0` | fijo | límite de Box |
| `ACTION_HIGH` | float | `1.0` | fijo | límite de Box |
| `ACTION_DIM` | int | `28` | fijo | número de actuadores |
| `OBS_DIM` | int | `39` | fijo | observación plana |

No se deben duplicar estos valores en distintas celdas/módulos cuando el desarrollo del notebook empiece.

---

## 12. Restricciones del problema

### 12.1 Restricciones duras del environment

1. Acciones continuas limitadas a `[-1,1]^28`.
2. Las acciones fuera de rango son recortadas.
3. Límites físicos articulares impuestos por MuJoCo/XML.
4. Puerta limitada aproximadamente a `[0,1.57] rad`.
5. Pestillo limitado aproximadamente a `[0,1.8] rad`.
6. Fricción significativa en pestillo y puerta.
7. Posición de la puerta cambia en cada reset.
8. El éxito depende del ángulo de puerta, no de la distancia de la mano ni de `door_open` por sí solos.
9. No hay finalización automática por éxito.
10. El horizonte está limitado por `TimeLimit`.

### 12.2 Restricciones académicas/transversales

1. Debe conservarse `AdroitHandDoor-v1` salvo decisión académica explícita.
2. El algoritmo asignado a este ambiente debe pertenecer al conjunto permitido por el reto.
3. El reto completo debe emplear al menos tres algoritmos de la lista permitida.
4. Entrenamiento y evaluación deben permanecer separados.
5. Evaluación final: mínimo 10 episodios.
6. Debe producirse modelo entrenado, métricas, gráficas, videos y reporte.
7. El desarrollo sustantivo de este modelo deberá ejecutarse con un DWP **Full**.

---

## 13. Variables que deben observarse durante entrenamiento

Aunque no todas forman parte del observation space entregado a la política, se recomienda registrar para diagnóstico:

| Variable | Uso |
| --- | --- |
| `episode_return` | objetivo RL principal |
| `episode_length` | eficiencia temporal |
| `success` | métrica funcional |
| `door_hinge` | progreso físico principal |
| `latch` | diagnóstico de manipulación de manija |
| `palm_handle_distance` | aproximación/manipulación |
| `action_norm` | saturación/esfuerzo de política |
| `qvel_norm` | detectar movimientos violentos |
| `training_timesteps` | presupuesto de interacción |
| `wall_clock_training_time` | costo real de entrenamiento |

No se propone añadir estas variables a la observación del agente; son telemetría para análisis.

---

## 14. Contrato de validación previo al entrenamiento

Antes de seleccionar hiperparámetros definitivos o lanzar entrenamiento largo, el notebook deberá verificar programáticamente:

```python
assert env.spec.id == "AdroitHandDoor-v1"
assert env.action_space.shape == (28,)
assert env.observation_space.shape == (39,)
assert env.action_space.low.min() == -1.0
assert env.action_space.high.max() == 1.0
```

Además deberá imprimir/registrar:

```text
gymnasium.__version__
gymnasium_robotics.__version__
mujoco.__version__
env.spec
env.spec.max_episode_steps
env.action_space
env.observation_space
env.unwrapped.frame_skip
env.unwrapped.model.actuator_ctrlrange
```

Smoke test mínimo:

1. `reset(seed=...)` sin errores;
2. verificar `obs.shape == (39,)`;
3. ejecutar al menos un `step()`;
4. verificar reward finito;
5. verificar claves de `info`, especialmente `success`;
6. verificar render `rgb_array`;
7. verificar truncation al horizonte efectivo;
8. guardar la caracterización runtime junto al experimento.

---

## 15. Implicaciones para el diseño del agente

Esta sección no fija todavía el algoritmo ni sus hiperparámetros; traduce la caracterización del ambiente a requisitos para la siguiente fase.

### 15.1 Espacio de control

La política debe producir un vector continuo de dimensión 28. Por tanto, el algoritmo elegido debe soportar naturalmente espacios `Box` continuos y control de alta dimensión.

### 15.2 Coordinación temporal

La tarea no se resuelve mediante una única acción. Requiere una secuencia coordinada: aproximación → contacto → manipulación del pestillo → apertura. La estabilidad de la política y la exploración son especialmente relevantes.

### 15.3 Reward con discontinuidades

Aunque la reward es densa, contiene saltos de `+2`, `+8` y `+10`, además de términos continuos. Las curvas de reward deben interpretarse conjuntamente con `success_rate` y `door_hinge`; un aumento de retorno por sí solo no prueba que el agente aprendió a abrir la puerta.

### 15.4 Aleatoriedad del reset

La posición aleatoria de la puerta exige generalización espacial. Evaluar siempre sobre la misma seed produciría una estimación pobre del desempeño.

### 15.5 Advertencia sobre reward v1

El signo histórico de la distancia palma–manija puede generar un gradiente de aprendizaje contradictorio. Esta condición debe considerarse al analizar estabilidad, convergencia y comportamiento aprendido, pero no debe “corregirse” silenciosamente en nuestro código.

---

## 16. Configuración baseline de ambiente propuesta

Esta es la configuración inicial que debe usar el primer smoke test; no es todavía la configuración definitiva del algoritmo:

```yaml
environment:
  id: AdroitHandDoor-v1
  reward: dense
  max_episode_steps_expected: 200
  action_dim: 28
  observation_dim: 39
  action_low: -1.0
  action_high: 1.0
  success_threshold_rad: 1.35
  door_open_flag_threshold_rad: 1.0
  door_target_rad: 1.57
  render_mode_training: null
  render_mode_video: rgb_array
  frame_skip_expected: 5
```

**Regla:** cualquier diferencia observada entre esta ficha y `env.spec`/runtime debe bloquear el entrenamiento largo hasta quedar explicada y documentada.

---

## 17. Decisiones pendientes para la fase de planeación

Esta ficha deja caracterizado el environment. La siguiente fase deberá resolver, mediante DWP Full, como mínimo:

1. versión exacta y pin de `gymnasium-robotics`, Gymnasium y MuJoCo;
2. confirmación runtime del reward efectivo de `AdroitHandDoor-v1`;
3. algoritmo asignado a Adroit Hand dentro de la estrategia de tres algoritmos del reto;
4. política/red y normalización de observaciones/acciones;
5. presupuesto de timesteps;
6. frecuencia de evaluación;
7. conjunto de seeds;
8. criterios de checkpoint y selección del modelo final;
9. hiperparámetros controlables;
10. instrumentación mínima de métricas;
11. estrategia de video de entrenamiento y agente entrenado;
12. criterios cuantitativos para declarar que el comportamiento es satisfactorio.

---

## 18. Checklist de caracterización cerrada

- [x] Objetivo físico identificado.
- [x] ID del environment identificado.
- [x] Action space documentado.
- [x] 28 acciones mapeadas.
- [x] Observation space documentado.
- [x] 39 observaciones mapeadas.
- [x] Reward dense documentada con fórmula.
- [x] Reward sparse documentada como referencia.
- [x] Bug conocido de reward `v1` identificado.
- [x] Success threshold identificado.
- [x] Hitos de apertura identificados.
- [x] Estado inicial y aleatoriedad documentados.
- [x] Estado serializable documentado.
- [x] Restricciones físicas principales documentadas.
- [x] Horizonte y semántica `terminated/truncated` documentados.
- [x] Render modes documentados.
- [x] Variables runtime que deben verificarse definidas.
- [x] Smoke gate previo a entrenamiento definido.
- [ ] Versión final de dependencias fijada — pendiente de planeación.
- [ ] Algoritmo e hiperparámetros fijados — pendiente de planeación.

---

## 19. Referencias técnicas

- Gymnasium Robotics — Adroit Hand: https://robotics.farama.org/envs/adroit_hand/
- Gymnasium Robotics — Adroit Door: https://robotics.farama.org/envs/adroit_hand/adroit_door/
- Fuente oficial: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.0/gymnasium_robotics/envs/adroit_hand/adroit_door.py
- XML de puerta: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.0/gymnasium_robotics/envs/assets/adroit_hand/adroit_door.xml
- Modelo Adroit: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.0/gymnasium_robotics/envs/assets/adroit_hand/adroit_model.xml

---

## 20. Regla de mantenimiento

Este documento describe el contrato conocido de `AdroitHandDoor-v1`. Si durante la implementación el runtime contradice una variable aquí documentada, no se cambiará el código para forzar esta ficha. Primero se debe:

1. identificar versión instalada;
2. reproducir la diferencia;
3. contrastarla contra la fuente upstream de esa versión;
4. documentar la decisión;
5. actualizar esta ficha y el DWP antes de continuar con el entrenamiento largo.
