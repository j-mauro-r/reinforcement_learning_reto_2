# RETO II

Para el segundo reto del curso vamos a trabajar dentro del entorno de gymnasiumrobotics (
https://robotics.farama.org
) que plantea diferentes ambientes para resolver problemas relacionados con robots de distintos tipos. En particular, para este reto vamos a utilizar tres ambientes específicos para la manipulación de objetos con una mano robótica.

### El primer ambiente es el ambiente de Adroit Hand (
https://robotics.farama.org/envs/adroit_hand/
). Este ambiente consiste en la simulación de una mano robótica (de Shadow Robotics) pegada a un brazo libre. La mano tiene 30 grados de libertad de movimiento (en respuesta a sus articulaciones). Dentro de este ambiente vamos a resolver una tarea particular, la tarea de abrir una puerta (
https://robotics.farama.org/envs/adroit_hand/adroit_door/
). Dentro del ambiente la mano se encuentra frente a una puerta, donde debe tomar la manija de la puerta y realizar los movimientos para abrirla.

### El segundo ambiente es el ambiente de Fetch 
donde resolveremos una tarea específica, la tarea Reach (
https://robotics.farama.org/envs/fetch/reach/
). Este ambiente consiste en un brazo mecánico manipulador que debe mover su efector final a una posición objetivo sobre una mesa o en el aire. El brazo mecánico es un manipulador móvil de 7 grados de libertad con una pinza de dos dedos. El robot se controla mediante pequeños desplazamientos de la pinza en coordenadas cartesianas, y la cinemática inversa se calcula internamente mediante el framework de MuJoCo. La tarea es continua, lo que significa que el robot debe mantener el efector en la posición objetivo durante un tiempo indefinido.

### El tercer ambiente es el ambiente de Shadow Dexterous Hand, 
el cual consiste en una mano antropomórfica con 20 grados de libertad que permiten el movimiento de la muñeca y cada una de las articulaciones de los dedos. Dentro de este ambiente resolveremos una tarea particular, Reach (
https://robotics.farama.org/envs/shadow_dexterous_hand/reach/
)  donde se busca que los dedos de la mano lleguen a una posición particular. 

Su tarea consiste en resolver los problemas propuestos en cada uno de los ambientes suministrados. Para ello, podrán utilizar cualquiera de los métodos vistos en clase. No obstante, es obligatorio emplear al menos tres (3) de los siguientes algoritmos disponibles:

- A3C / A2C
- SAC
- TRPO
- PPO
- DDPG
- TD3

El objetivo en todos los casos es maximizar la recompensa acumulada durante la tarea, por lo que, al momento de seleccionar los algoritmos, también deben considerar el tiempo de entrenamiento y la estabilidad de cada método.

Se recomienda trabajar con un sistema de recompensas continuo (recompensas densas), ya que este proporciona retroalimentación frecuente sobre el progreso del agente y facilita el proceso de aprendizaje. En particular:

Para el problema reach en el ambiente Shadow Dexterous Hand, se recomienda utilizar la variante HandReachDense-v3.
Para el problema reach en el ambiente Fetch, se recomienda utilizar la variante FetchReachDense-v4.
Estas versiones manejan recompensas densas, lo que permite un entrenamiento más estable y eficiente.

Cada uno de los problemas cuenta con un notebook asociado donde se muestra la instalación del entorno y la ejecución básica de una política aleatoria durante un episodio, el cual puede tomarse como punto de partida para el desarrollo del reto: @Gymnasium_Robotics.ipynb

## Entrega

 La entrega del reto para cada problema debe incluir los siguientes elementos obligatorios:

1. Archivos a entregar

Un notebook en Google Colab con el desarrollo completo del agente. Al entregar el notebook, este debe poder ser ejecutado sin errores ni problemas de dependencias por el equipo académico.  Los notebooks que no ejecuten correctamente tendrán una penalización.
El modelo entrenado del agente, el cual será utilizado para verificar la correcta ejecución y el comportamiento aprendido.
Un video corto en el que se demuestre el funcionamiento de los agentes desarrollados (es decir el render del ambiente, no necesitan grabarse o colocar audio). En el video debe evidenciarse:
El proceso de entrenamiento del agente.
El comportamiento aprendido por el agente una vez entrenado.
2. Reporte técnico (incluido dentro del notebook)

El notebook debe contener un reporte técnico con presentación profesional, en el cual se explique y justifique el proceso de desarrollo de los agentes. Este reporte debe incluir, como mínimo, los siguientes puntos:

- La selección de los algoritmos de aprendizaje por refuerzo utilizados, junto con una breve justificación de su elección para cada problema.
- Las condiciones de ejecución del agente, incluyendo:
- Hiperparámetros de aprendizaje utilizados.
- Librerías y versiones empleadas.
- Características del hardware utilizado durante el entrenamiento.
- Un análisis de los resultados obtenidos para cada uno de los problemas resueltos. Este análisis debe estar acompañado de evidencias cuantitativas del entrenamiento y desempeño del agente, tales como:
  * Tiempo de entrenamiento.
  * Puntaje promedio obtenido por el agente en al menos 10 episodios de evaluación. Para juzgar el desempeño de su agente, especialmente en los juegos de Atari, puede compararlo con el comportamiento de una política completamente aleatoria.  
  * Gráficas de la evolución de la recompensa durante entrenamiento y explotación (se recomienda el uso de TensorBoard para la generación de estas gráficas).  
  Una descripción del comportamiento aprendido por el agente, relacionando dicho comportamiento con el algoritmo utilizado y los hiperparámetros seleccionados.
  * Finalmente, una conclusión general, basada en los resultados obtenidos, sobre las capacidades de los agentes implementados (es decir, de los métodos de aprendizaje por refuerzo utilizados) para resolver el problema propuesto.

## Evaluación

La evaluación del reto tendrá en cuenta:

- La completitud de la entrega. La entrega debe estar completa para cada problema.
- La implementación del agente. El notebook de su implementación de su agente debe seguir buenos estándares de programación, estar organizada, debe ser funcional, y debe dar evidencia de un desarrollo dentro del contexto de los métodos utilizados en el curso.
- El modelo del agente entrenado debe corresponder al entrenamiento del agente en el notebook.
Para cada uno de los problemas el agente debe ser capaz de resolver el juego efectivamente.
- El video debe dar evidencia del proceso de entrenamiento del agente y su ejecución
- El reporte debe presentar formalmente el trabajo realizado debidamente justificado.
