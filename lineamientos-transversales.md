# Lineamientos transversales — Reto II de Reinforcement Learning

> **Estado:** obligatorio para los tres modelos del Reto II.  
> **Repositorio:** `j-mauro-r/reinforcement_learning_reto_2`  
> **Objetivo:** establecer una única fuente de reglas de ingeniería, experimentación, entrenamiento, evaluación y entrega para que los tres desarrollos sean reproducibles, simples, trazables y alineados con el enunciado académico.

---

## 1. Principios rectores

Estos lineamientos aplican transversalmente a los tres problemas:

1. `AdroitHandDoor-v1`
2. `FetchReachDense-v4`
3. `HandReachDense-v3`

El objetivo académico común es **maximizar la recompensa acumulada**, considerando también **estabilidad** y **tiempo de entrenamiento**. El enunciado exige utilizar al menos **tres algoritmos** de la lista permitida: A3C/A2C, SAC, TRPO, PPO, DDPG y TD3.

Las decisiones de implementación deberán seguir, en este orden, estas prioridades:

1. **Cumplir el enunciado y los criterios de evaluación.**
2. **Lograr un agente que resuelva efectivamente la tarea.**
3. **Garantizar reproducibilidad y trazabilidad.**
4. **Mantener el código simple, limpio y ejecutable en Google Colab.**
5. **Optimizar estabilidad y tiempo de entrenamiento.**
6. **Evitar sobreingeniería y componentes que no aporten evidencia al reto.**

### 1.1 Fuentes de verdad y precedencia

En caso de conflicto, se seguirá esta precedencia:

1. `enunciado-reto.md`.
2. Este archivo `lineamientos-transversales.md`.
3. La especificación vigente de Deep Work Plan (DWP): <https://deepworkplan.com/es/spec/>.
4. El Deep Work Plan específico del modelo que se esté desarrollando.
5. `Gymnasium_Robotics.ipynb` como referencia técnica de instalación, creación de entornos y render; **no** como arquitectura obligatoria.
6. Decisiones experimentales documentadas y soportadas por evidencia.

No se deberá modificar una regla transversal de forma implícita dentro de un notebook. Cualquier excepción deberá quedar justificada y aprobada mediante un cambio explícito a estos lineamientos o al plan DWP correspondiente.

---

## 2. Alcance de los tres modelos

Cada problema tendrá su propia carpeta en la raíz del repositorio:

```text
1-AdroitHandDoor/
2-FetchReachDense/
3-HandReachDense/
```

Cada carpeta será autónoma para efectos de ejecución, entrenamiento, evaluación y entrega, pero deberá conservar la misma organización conceptual.

### 2.1 Entornos obligatorios

| Modelo | Entorno base | Recompensa |
|---|---|---|
| 1 | `AdroitHandDoor-v1` | La definida por el entorno, salvo justificación documentada |
| 2 | `FetchReachDense-v4` | Densa |
| 3 | `HandReachDense-v3` | Densa |

No se cambiará la versión del entorno sin una razón técnica documentada, una validación de compatibilidad y una actualización del DWP del modelo.

### 2.2 Algoritmos

El reto completo deberá emplear **mínimo tres algoritmos distintos** de la lista permitida por el enunciado.

Como hipótesis inicial de trabajo se recomienda:

| Modelo | Algoritmo inicial | Razón de partida |
|---|---|---|
| AdroitHandDoor | SAC | Control continuo de alta dimensionalidad y buena exploración mediante política estocástica |
| FetchReachDense | TD3 | Control continuo preciso y reducción del sesgo de sobreestimación |
| HandReachDense | PPO | Alternativa estable para control continuo de alta dimensionalidad y cumplimiento del requisito de diversidad |

Esta asignación es una **hipótesis inicial**, no una conclusión. Puede cambiarse únicamente si el análisis técnico o la evidencia experimental lo justifican. El cambio deberá registrarse en el DWP y conservar el requisito global de mínimo tres algoritmos distintos.

Cuando exista una implementación madura en una librería estándar, se preferirá utilizarla en lugar de reimplementar el algoritmo desde cero. Por defecto se priorizará **Stable-Baselines3** y, si fuera estrictamente necesario, `sb3-contrib`. La implementación manual solo se justificará si aporta valor académico explícito o si la librería no cubre un requisito.

No se añadirá HER, reward shaping adicional, curriculum learning u otra técnica avanzada al primer baseline. Estas técnicas solo se incorporarán si existe evidencia de que el baseline no aprende adecuadamente y si el incremento de complejidad está justificado mediante DWP.

---

## 3. Deep Work Plan (DWP) obligatorio

Desde el inicio del desarrollo de cada modelo se utilizará DWP conforme a la especificación vigente.

### 3.1 Nivel de rigor

El desarrollo completo de cada modelo se tratará, por defecto, como trabajo **standard**, porque incluye múltiples etapas: entendimiento del entorno, baseline, entrenamiento, evaluación, videos, métricas y reporte.

Cambios realmente atómicos y de una sola preocupación podrán tratarse como **micro**, respetando el principio de rigor proporcional de DWP.

### 3.2 Reglas DWP

Antes de implementar entrenamiento de un modelo:

- El repositorio deberá haber sido incorporado a DWP.
- Deberá existir un plan específico para el modelo bajo `.dwp/plans/PLAN_<slug>/`.
- Cada tarea del plan deberá incluir las nueve secciones normativas de DWP:
  1. Goal.
  2. Context.
  3. Steps.
  4. Acceptance criteria.
  5. Validation.
  6. Files.
  7. Dependencies.
  8. Risks.
  9. Completion & Log.
- Cada tarea deberá tener criterios de aceptación verificables.
- Cada tarea deberá tener al menos un gate de validación objetivo.
- Solo se ejecutará la siguiente tarea pendiente; no se agruparán tareas sin necesidad.
- Antes de continuar un plan interrumpido se aplicará el protocolo de reanudación de DWP.
- Si el alcance cambia de forma material, se deberá **refinar el plan antes de implementar**.
- `.dwp/` será un workspace operativo y estará incluido en `.gitignore`, conforme a la especificación DWP.
- Las decisiones permanentes del proyecto no deberán existir únicamente en `.dwp/`; deberán reflejarse también en documentación versionada cuando corresponda.

### 3.3 DWP no justifica sobreingeniería

DWP es un mecanismo de control y trazabilidad, no una excusa para crear capas innecesarias.

Un task DWP deberá producir la solución mínima que satisfaga:

- el objetivo;
- los criterios de aceptación;
- los gates de validación;
- y los requisitos académicos.

---

## 4. Estrategia de Git y gobierno del repositorio

### 4.1 Regla absoluta sobre `main`

**Ningún cambio será implementado directamente sobre `main`.**

Todo cambio deberá seguir:

```text
main
  ↓
feature/... | experiment/... | fix/... | docs/...
  ↓
commit(s)
  ↓
Pull Request
  ↓
revisión
  ↓
merge a main
```

### 4.2 Convención de ramas

Ejemplos:

```text
docs/lineamientos-transversales
feature/adroit-sac-baseline
experiment/adroit-sac-tuning
feature/fetch-td3-baseline
feature/hand-ppo-baseline
fix/fetch-video-overlay
```

Se evitarán ramas que mezclen cambios independientes de los tres modelos.

### 4.3 Pull Requests

Cada PR deberá:

- tener un objetivo único y entendible;
- explicar qué cambia y por qué;
- listar validaciones ejecutadas;
- identificar riesgos o limitaciones;
- no incluir notebooks con celdas abandonadas o código muerto;
- no mezclar refactorizaciones no relacionadas;
- no hacer merge automático sin revisión.

### 4.4 Archivos generados y versionados

Se versionarán:

- notebooks finales;
- configuraciones;
- código reutilizable realmente necesario;
- pruebas si se crean;
- artefacto final del modelo, siempre que su tamaño y la política de GitHub lo permitan;
- reporte, gráficas y videos requeridos para la entrega cuando sea viable.

Se ignorarán por defecto:

- checkpoints temporales;
- caches;
- logs voluminosos;
- `mlruns/`;
- ejecuciones temporales;
- ambientes virtuales;
- `.dwp/`;
- `tmp/`.

**No se deberán ignorar directorios de código, pruebas o documentación solamente para reducir ruido.**

---

## 5. Estructura mínima por modelo

Cada carpeta deberá mantener una estructura compacta. Se crearán únicamente los componentes que el modelo necesite.

Estructura de referencia:

```text
<modelo>/
├── <modelo>.ipynb              # Notebook principal y entregable
├── configs/
│   └── config.yaml             # Perillas e hiperparámetros, si se usa config externa
├── models/
│   ├── <modelo>_<algoritmo>.<ext>
│   └── <modelo>_<algoritmo>.metadata.json
├── videos/
│   ├── training_process.mp4
│   └── trained.mp4
├── results/
│   ├── evaluation.csv
│   └── figures/
└── src/                        # Solo si existe lógica reutilizable que justifique extraerla
```

No se crearán `src/`, clases, factories, managers, adapters, wrappers o módulos auxiliares si una función pequeña y clara dentro del notebook resuelve el problema correctamente.

---

## 6. Buenas prácticas de desarrollo obligatorias

### 6.1 Simplicidad

Se seguirá el principio:

> La solución más simple que cumpla el reto, sea reproducible y permita evidenciar el aprendizaje es preferible a una solución más sofisticada.

Se prohíbe:

- código muerto;
- celdas duplicadas;
- funciones sin uso;
- clases de una sola utilidad sin justificación;
- abstracciones anticipadas;
- patrones de diseño aplicados sin necesidad;
- duplicación de pipelines equivalentes;
- múltiples mecanismos para resolver el mismo problema.

### 6.2 SOLID de forma proporcional

Se aplicarán principios SOLID cuando existan componentes reutilizables, sin forzar orientación a objetos.

Ejemplos:

- una función de evaluación no debe entrenar;
- una función de video no debe modificar los pesos del agente;
- la configuración no debe estar dispersa por el notebook;
- una función de creación de entorno debe concentrar las opciones del entorno.

### 6.3 DRY

No se repetirán:

- IDs del entorno;
- rutas;
- seeds;
- hiperparámetros;
- número de episodios de evaluación;
- nombres de artefactos;
- configuraciones de render;
- lógica de evaluación;
- lógica de guardado/carga.

Toda constante transversal deberá tener una única fuente de verdad.

### 6.4 Documentación

Funciones y clases reutilizables deberán usar **docstrings estilo Google**.

Cuando aporte claridad se utilizarán:

- type hints;
- nombres descriptivos;
- comentarios que expliquen el **porqué**, no lo obvio.

No se llenará el notebook de comentarios narrando línea por línea el código.

---

## 7. Configuración, “perillas” y rutas

### 7.1 Configuración centralizada

Todos los hiperparámetros y variables ajustables deberán estar expuestos como “perillas” en una única sección o archivo de configuración.

Como mínimo:

- `ENV_ID`
- `ALGORITHM`
- `SEED`
- `TOTAL_TIMESTEPS`
- `LEARNING_RATE`
- `GAMMA`
- parámetros específicos del algoritmo;
- episodios de evaluación;
- configuración de video;
- dispositivo;
- rutas de artefactos.

No se permitirán hiperparámetros relevantes escondidos como números mágicos dentro de funciones.

### 7.2 Rutas

Las rutas deberán derivarse de una única raíz:

```python
PROJECT_ROOT
MODEL_DIR
VIDEO_DIR
RESULTS_DIR
```

No se dispersarán cadenas como `/content/...`, `./models/...` o `../results/...` por múltiples celdas.

Si Google Colab requiere `/content`, la ruta deberá existir en **una sola variable bootstrap** y el resto se derivará de ella.

### 7.3 Perfiles de ejecución

Cada modelo podrá tener dos perfiles simples:

- `smoke`: valida instalación, entorno, entrenamiento corto, guardado/carga y pipeline.
- `full`: entrenamiento real.

No se crearán múltiples perfiles adicionales sin necesidad.

---

## 8. Reproducibilidad

Cada modelo deberá establecer y registrar semillas para:

- `random`;
- NumPy;
- el framework de entrenamiento utilizado;
- el entorno Gymnasium;
- espacios de acción/observación cuando aplique.

La evaluación deberá usar semillas controladas y distintas de las del entrenamiento.

La reproducibilidad en RL no implica exigir resultados bit a bit idénticos; implica poder reconstruir las condiciones experimentales y obtener comportamiento estadísticamente comparable.

Cada ejecución final deberá registrar como mínimo:

- `run_id`;
- fecha/hora;
- entorno y versión;
- algoritmo;
- seed;
- hiperparámetros;
- librerías y versiones;
- hardware/dispositivo;
- timesteps;
- tiempo de entrenamiento;
- ruta del modelo final;
- resultados de evaluación.

---

## 9. Dependencias y Google Colab

El notebook final deberá poder ejecutarse desde un **runtime limpio de Google Colab**.

### 9.1 Instalación

La primera sección deberá:

1. instalar dependencias necesarias;
2. configurar MuJoCo/Gymnasium Robotics;
3. configurar render headless si aplica;
4. importar librerías;
5. mostrar versiones relevantes;
6. ejecutar un smoke test mínimo del entorno.

Se evitarán instalaciones duplicadas.

Las versiones finales deberán fijarse después de validar una combinación funcional en Colab.

### 9.2 Regla de compatibilidad

Antes de un entrenamiento largo deberá probarse:

```text
instalar → importar → crear env → reset → step → render → cerrar env
```

Después:

```text
entrenamiento corto → guardar → cargar → evaluar → generar video
```

Solo si esos gates pasan se habilitará el entrenamiento completo.

---

## 10. Notebook principal: capítulos obligatorios

El notebook principal de **cada modelo** deberá contener, como mínimo, los siguientes grandes capítulos y en este orden.

### Capítulo 1 — Librerías, bootstrap y variables transversales

Debe incluir:

- instalación;
- imports;
- versiones;
- detección de CPU/GPU;
- seed;
- configuración;
- rutas;
- creación de carpetas;
- helpers mínimos;
- smoke test.

### Capítulo 2 — Definición del agente

Debe explicar y definir explícitamente:

- problema;
- entorno;
- observación/estado;
- espacio de acciones;
- recompensa;
- política;
- algoritmo;
- principales hiperparámetros;
- criterio de éxito;
- seeds de entrenamiento/evaluación.

Se deberán imprimir o inspeccionar `observation_space` y `action_space` antes de entrenar.

### Capítulo 3 — Construcción y ejecución del entrenamiento

Debe incluir:

- construcción del modelo;
- callbacks estrictamente necesarios;
- logging;
- inicio/fin del cronómetro;
- entrenamiento;
- guardado del modelo;
- validación de que el modelo se puede recargar.

El entrenamiento completo no deberá depender de ejecutar manualmente celdas fuera de orden.

### Capítulo 4 — Métricas y gráficas

Cada notebook deberá calcular **exactamente un conjunto mínimo común de 3 métricas principales** y mostrar **3 gráficas principales**, sin perjuicio de métricas auxiliares.

#### Métricas principales

1. **Tiempo total de entrenamiento** en segundos/minutos.
2. **Recompensa acumulada promedio de evaluación** sobre **mínimo 10 episodios**, acompañada de desviación estándar.
3. **Tasa de éxito de la tarea** sobre los mismos episodios, usando `is_success` cuando el entorno lo exponga o un criterio explícito y documentado cuando no exista.

#### Gráficas principales

1. **Evolución de recompensa durante entrenamiento**: retorno por episodio o ventana equivalente + media móvil.
2. **Desempeño en explotación/evaluación**: recompensa acumulada por episodio para el agente entrenado, mínimo 10 episodios.
3. **Comparación contra baseline aleatorio**: distribución o barras de recompensa promedio `Random vs Trained`, usando condiciones comparables.

Cuando sea útil, la tercera gráfica podrá incorporar una variable física de éxito:
- distancia al objetivo para tareas Reach;
- apertura/ángulo de puerta para Adroit.

Pero nunca reemplazará la comparación cuantitativa principal si esta es necesaria para demostrar mejora.

TensorBoard puede utilizarse como apoyo, pero las tres gráficas obligatorias deberán quedar visibles dentro del notebook final.

### Capítulo 5 — Videos

Se deberán producir dos videos distintos.

#### Video 1 — `Training Process`

Objetivo: evidenciar que el agente se encuentra **en proceso de entrenamiento**, no presentar como entrenamiento una reproducción generada únicamente con el modelo final.

Requisitos:

- archivo: `training_process.mp4`;
- título visible: `Training Process`;
- mostrar el render del entorno;
- mostrar en overlay el reward/score acumulado y, cuando sea útil, episodio o timestep;
- generarse con una política correspondiente a una etapa real del entrenamiento;
- mantener duración corta para no penalizar significativamente el entrenamiento;
- no modificar la dinámica de aprendizaje por el simple hecho de grabar.

Una forma válida y eficiente es registrar una evaluación corta de un **checkpoint intermedio** claramente identificado como parte del proceso de entrenamiento.

#### Video 2 — `Trained`

Objetivo: demostrar el comportamiento aprendido por el modelo final.

Requisitos:

- archivo: `trained.mp4`;
- título visible: `Trained Agent`;
- cargar explícitamente el artefacto final entrenado;
- ejecutar en modo explotación/determinista cuando el algoritmo lo permita;
- mostrar reward/score acumulado;
- no ejecutar actualizaciones de aprendizaje;
- usar el mismo contrato de entorno/preprocesamiento del entrenamiento.

> **Resolución de ambigüedad:** la instrucción original repite como ejemplo el título `Training Process` para el segundo video, aunque lo denomina “Video 2 Trained”. Para distinguir inequívocamente las dos evidencias se adopta `Trained Agent` como título del segundo video. Si el equipo académico exige literalmente otra etiqueta, se cambia únicamente el texto del overlay.

### Capítulo 6 — Informe técnico

El reporte estará integrado dentro del notebook y deberá contener:

1. selección y justificación del algoritmo;
2. condiciones de ejecución;
3. hiperparámetros;
4. librerías y versiones;
5. hardware;
6. tiempo de entrenamiento;
7. evaluación de mínimo 10 episodios;
8. análisis de las 3 métricas;
9. análisis de las 3 gráficas;
10. descripción del comportamiento aprendido;
11. comparación con política aleatoria;
12. relación entre comportamiento, algoritmo e hiperparámetros;
13. limitaciones;
14. conclusión general.

Las afirmaciones del informe deberán estar respaldadas por los resultados obtenidos en el mismo notebook.

---

## 11. Entrenamiento

### 11.1 Baseline primero

Cada modelo seguirá:

```text
entorno validado
→ política aleatoria de referencia
→ baseline entrenable mínimo
→ smoke training
→ evaluación
→ entrenamiento completo
→ tuning solo si es necesario
```

No se iniciará con tuning exhaustivo.

### 11.2 Tuning controlado

Cuando sea necesario ajustar hiperparámetros:

- se partirá de un baseline funcional;
- se cambiará un conjunto pequeño y explícito de perillas;
- se registrará cada experimento relevante;
- se comparará con el baseline;
- no se harán búsquedas masivas sin justificación;
- se priorizarán parámetros de alto impacto del algoritmo seleccionado.

No se elegirá el “mejor” modelo por un único episodio afortunado.

### 11.3 Checkpoints

Se usarán checkpoints únicamente cuando aporten:

- recuperación ante interrupción;
- evidencia de proceso de entrenamiento;
- selección de modelo;
- o trazabilidad.

Los checkpoints temporales no deben confundirse con el modelo final entregable.

---

## 12. Evaluación

### 12.1 Separación estricta entrenamiento/evaluación

Durante evaluación:

- no se actualizarán pesos;
- no se actualizará replay buffer salvo que la librería lo haga de forma inevitable y sin entrenamiento;
- no habrá exploración artificial salvo que sea parte explícita de la política evaluada;
- se usará modo determinista cuando aplique;
- se usarán mínimo 10 episodios;
- se conservarán rewards por episodio;
- se calcularán media, desviación estándar y tasa de éxito.

### 12.2 Baseline aleatorio

Cada modelo deberá ejecutar una política aleatoria sobre condiciones comparables para responder:

> ¿El agente entrenado mejora materialmente frente a no haber aprendido?

La comparación deberá usar el mismo entorno, horizonte y número de episodios o una configuración explícitamente equivalente.

### 12.3 Criterio de “resolver” la tarea

No se declarará que un agente “resuelve” el problema únicamente porque su reward aumentó.

La conclusión deberá apoyarse en:

- recompensa;
- tasa de éxito;
- evidencia visual;
- y una variable física o comportamiento observable cuando aplique.

---

## 13. Artefacto del modelo y trazabilidad

Cada modelo tendrá **un único artefacto canónico de entrega** claramente identificado.

Ejemplo:

```text
models/adroit_hand_door_sac.zip
models/fetch_reach_td3.zip
models/hand_reach_ppo.zip
```

Se evitarán nombres ambiguos como:

```text
best.pth
best_single.pth
final2.pth
final_ok.pth
model_new.pt
```

### 13.1 Metadatos mínimos

Junto al modelo final deberá generarse un pequeño `metadata.json` con:

- `run_id`;
- `environment_id`;
- `algorithm`;
- `seed`;
- `total_timesteps`;
- hiperparámetros esenciales;
- versiones;
- tiempo de entrenamiento;
- recompensa media de evaluación;
- desviación estándar;
- tasa de éxito;
- checksum SHA-256 del artefacto, cuando sea viable.

### 13.2 Correspondencia obligatoria

El modelo entregado deberá corresponder al entrenamiento documentado en el notebook.

No se aceptará:

- entrenar un modelo fuera del notebook y entregar otro sin trazabilidad;
- evaluar pesos distintos a los guardados;
- modificar configuración de observación/acción entre entrenamiento y explotación;
- declarar métricas de una ejecución diferente sin identificarla.

---

## 14. Videos: reglas técnicas

La captura de video deberá reutilizar una función simple y común por modelo, parametrizada por:

- título;
- número de episodios;
- ruta de salida;
- política/modelo;
- modo training/intermediate/final.

El overlay deberá ser legible e incluir como mínimo:

```text
Training Process | Reward: ...
```

o

```text
Trained Agent | Reward: ...
```

Se evitará grabar todo el entrenamiento completo frame a frame si esto afecta significativamente el rendimiento. La evidencia debe ser suficiente, no costosa.

---

## 15. Logging y resultados

Se mantendrá logging suficiente para responder el enunciado, sin construir una plataforma MLOps.

Mínimo:

- reward de entrenamiento;
- duración/steps por episodio cuando esté disponible;
- timestamps de inicio/fin;
- métricas de evaluación;
- tasa de éxito.

TensorBoard es recomendado cuando se integre de forma sencilla.

MLflow no será obligatorio. Solo se incorporará si aporta un beneficio real sin aumentar desproporcionadamente la complejidad.

---

## 16. Lecciones aprendidas del Reto I

La revisión del repositorio `reinforcement_learning_reto_1` deja prácticas útiles y problemas a evitar.

### 16.1 Prácticas a conservar

#### A. Centralización de configuración

En LunarLander, Assault y BattleZone se separaron parámetros del entorno, red, entrenamiento y evaluación. Esto facilita tuning, repetibilidad y revisión.

**Regla Reto II:** todas las perillas estarán centralizadas.

#### B. Perfiles `smoke` y `full`

Assault y BattleZone distinguen validaciones cortas de entrenamiento real.

**Regla Reto II:** conservar solo estos dos perfiles cuando sean necesarios.

#### C. Seeds y evaluación separada

El Reto I incluye seeds controladas y episodios de evaluación explícitos.

**Regla Reto II:** entrenamiento y evaluación tendrán seeds controladas y diferenciadas.

#### D. Checkpoints recuperables

Assault y BattleZone implementaron persistencia de checkpoints para proteger entrenamientos largos.

**Regla Reto II:** conservar checkpointing cuando el costo de reentrenar lo justifique.

#### E. Artefacto de inferencia separado del estado completo de entrenamiento

En Assault se creó un artefacto de inferencia compacto, eliminando optimizer, replay buffer y otros datos que no son necesarios para ejecutar la política.

**Regla Reto II:** diferenciar modelo entregable de checkpoints de entrenamiento.

#### F. Trazabilidad del artefacto

El Reto I incluyó metadatos, checksum, configuración y referencias de la ejecución.

**Regla Reto II:** mantener una versión ligera de esa trazabilidad mediante `metadata.json`.

### 16.2 Prácticas a mejorar o evitar

#### A. Sobreingeniería

Assault y BattleZone terminaron con numerosos módulos especializados para bootstrap, checkpointing, delivery, experiment tracking, reporting y artefactos.

Esto mejora robustez en un sistema de producción, pero eleva el costo cognitivo en un reto académico centrado en RL.

**Regla Reto II:** no extraer módulos hasta que exista duplicación o complejidad real que lo justifique.

#### B. Duplicación de responsabilidades

En el Reto I aparecieron varios mecanismos relacionados con persistencia, entrega y bootstrap.

**Regla Reto II:** un solo responsable por tarea:
- una configuración;
- una función de entorno;
- una función de evaluación;
- una función de video;
- un modelo final.

#### C. Inconsistencia en la cantidad de episodios de evaluación

En Assault existe una configuración global de 10 episodios, mientras un perfil `full` contiene 5.

**Regla Reto II:** la evaluación final será siempre de **mínimo 10 episodios**, sin overrides que reduzcan este requisito.

#### D. Demasiadas variantes de modelos finales

LunarLander conserva artefactos como `best`, `best_single` y `final`, lo que puede generar ambigüedad sobre cuál corresponde a la entrega.

**Regla Reto II:** un único artefacto canónico de entrega; los checkpoints no se nombran como “final”.

#### E. `.gitignore` demasiado agresivo

En Reto I se ignoraron algunas carpetas de tests y docs en Assault/BattleZone.

**Regla Reto II:** ignorar solo artefactos generados; pruebas, documentación y código fuente deben permanecer versionados si existen.

#### F. Rutas y bootstrap replicados por problema

Reto I contiene lógica específica de resolución de rutas y bootstrap en varios módulos.

**Regla Reto II:** cada modelo tendrá una única raíz y un único bloque de rutas.

#### G. Complejidad MLOps no esencial

TensorBoard, MLflow, manifiestos, hashes y múltiples capas de delivery pueden ser valiosos, pero juntos pueden distraer del objetivo académico.

**Regla Reto II:** se mantendrán solamente:
- métricas requeridas;
- logging suficiente;
- metadata ligera;
- checksum del modelo;
- TensorBoard opcional.

---

## 17. Validaciones obligatorias antes de entrenamiento largo

No se ejecutará un entrenamiento costoso si falla alguno de estos gates:

### Gate 1 — Dependencias

- instalación limpia;
- imports correctos;
- versiones visibles.

### Gate 2 — Entorno

- `gym.make(...)`;
- `reset(seed=...)`;
- al menos varios `step(...)`;
- render funcional;
- `close()` correcto;
- observaciones/acciones con dimensiones esperadas.

### Gate 3 — Modelo

- construcción correcta;
- acción compatible con `action_space`;
- entrenamiento smoke sin excepciones.

### Gate 4 — Persistencia

- guardar modelo;
- crear nueva instancia o contexto;
- cargar modelo;
- ejecutar inferencia.

### Gate 5 — Evaluación

- evaluación sin aprendizaje;
- rewards recuperables;
- cálculo de métricas.

### Gate 6 — Video

- MP4 reproducible;
- overlay visible;
- reward/score visible;
- título correcto.

---

## 18. Definition of Done de cada modelo

Un modelo está terminado únicamente cuando:

- [ ] Existe un DWP completado para el desarrollo relevante.
- [ ] El notebook corre en un runtime limpio de Colab.
- [ ] El entorno correcto está documentado.
- [ ] El algoritmo está justificado.
- [ ] La configuración y rutas tienen una única fuente de verdad.
- [ ] Las seeds están controladas.
- [ ] El baseline aleatorio está medido.
- [ ] El entrenamiento completo está ejecutado.
- [ ] El modelo final está guardado y se puede recargar.
- [ ] El modelo final corresponde al notebook.
- [ ] Existen mínimo 10 episodios de evaluación.
- [ ] Se reportan las 3 métricas principales.
- [ ] Se muestran las 3 gráficas principales.
- [ ] Existe `training_process.mp4`.
- [ ] Existe `trained.mp4`.
- [ ] Ambos videos muestran reward/score.
- [ ] El informe técnico está completo.
- [ ] Se registran versiones y hardware.
- [ ] Se registra tiempo de entrenamiento.
- [ ] Se documentan limitaciones y conclusión.
- [ ] No existe código muerto ni duplicado evidente.
- [ ] El PR correspondiente está revisado antes de merge.

---

## 19. Checklist transversal del Reto II

Antes de la entrega global:

- [ ] Están completos `1-AdroitHandDoor/`, `2-FetchReachDense/` y `3-HandReachDense/`.
- [ ] Se utilizaron mínimo tres algoritmos distintos de la lista permitida.
- [ ] Los dos Reach usan las variantes Dense definidas por el enunciado.
- [ ] Los tres modelos pueden cargarse y ejecutarse.
- [ ] Cada modelo tiene evaluación de mínimo 10 episodios.
- [ ] Cada notebook contiene 3 métricas y 3 gráficas orientadas a evaluación.
- [ ] Cada modelo tiene video de proceso de entrenamiento y video entrenado.
- [ ] Los videos evidencian reward/score.
- [ ] Cada notebook contiene el reporte técnico requerido.
- [ ] Las librerías y versiones son reproducibles en Colab.
- [ ] Los artefactos entregados corresponden a las ejecuciones reportadas.
- [ ] No hay cambios directos en `main`.
- [ ] Todos los cambios llegaron mediante Pull Request.

---

## 20. Regla final

Ante una decisión entre una solución más compleja y una más simple, se elegirá la solución simple **siempre que**:

1. cumpla el enunciado;
2. preserve reproducibilidad;
3. permita entrenar, evaluar y cargar el agente correctamente;
4. produzca evidencia cuantitativa y visual suficiente;
5. no comprometa la calidad técnica.

El propósito de la ingeniería del Reto II es **hacer visible el aprendizaje por refuerzo**, no ocultarlo detrás de infraestructura innecesaria.

---

## Referencias

- Enunciado del Reto II: `enunciado-reto.md` (fuente del proyecto).
- Notebook base: `Gymnasium_Robotics.ipynb` (fuente del proyecto).
- Deep Work Plan, especificación: <https://deepworkplan.com/es/spec/>.
- Reto I — repositorio de referencia: <https://github.com/j-mauro-r/reinforcement_learning_reto_1>.
- Gymnasium Robotics: <https://robotics.farama.org/>.
