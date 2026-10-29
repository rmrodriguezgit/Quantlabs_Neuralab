# Arquitectura y decisiones

```text
React / TypeScript
  │ window.lab.run | importFiles | save
  ▼
preload de Electron (contextBridge)
  │ canales explícitos, frame principal validado
  ▼
Proceso principal Node.js / TypeScript
  ├── Diálogo nativo → importación / exportación
  └── spawn(Python, engine.py), sin shell
          │ JSON stdin → stdout; logs stderr
          ▼
       engine.py → models.py / neural.py / features.py → bibliotecas científicas
```

No se abre un servidor HTTP local. El renderer no tiene acceso a Node ni al sistema de archivos; los únicos métodos expuestos son los del preload. Context isolation, sandbox, CSP, bloqueo de nuevas ventanas y navegación. Los nombres de modelos se validan contra una lista cerrada. No se aceptan nombres de scripts ni comandos desde React. Las rutas de archivos provienen de diálogos del proceso principal. El motor importa los modelos pesados de manera diferida.

Cada trabajo utiliza un proceso nuevo y termina al concluir. Evita mezclar estado de entrenamiento entre datasets y permite descargar memoria al terminar una red. Hay timeout configurable de 1–240 minutos (60 por defecto; análisis general 120) y límite de salida; al cerrar Electron se terminan los procesos que controla. No hay cola persistente, trabajo en segundo plano ni cancelación manual en esta versión; solo una tarea UI a la vez.

## Contrato de datos

Un dataset contiene `id`, `name`, `kind`, `frequency`, `columns` y `rows`. Cada fila contiene `date` ISO y valores numéricos. Identificadores UUID distinguen archivos de igual nombre. Un resultado contiene `ranking`, `reports`, `failures`, `test`, `future`, `decomposition`, `residualHistogram` y `anova`. El test y el futuro se omiten si no se obtiene un ganador. Valores no finitos de estadísticas descriptivas se serializan como `null`.

## Separación entre exploración, elección y test

La descomposición y ANOVA usan la serie completa, pero sus valores nunca se utilizan como entradas de los modelos. El ranking usa prefijos de entrenamiento y ventanas posteriores de validación; no mezcla filas aleatoriamente. Las redes calculan su escalado dentro de cada llamada de entrenamiento. El test final está excluido del ranking. Se agrega una referencia de persistencia para comparar la utilidad de modelos complejos. El reentrenamiento futuro sí usa toda la serie, después de obtener el error de test.

No hay normalización global entre archivos. La tabla de sesión permite contrastar resultados, pero no selecciona un ganador entre magnitudes incompatibles. R² es indefinido en una serie de prueba constante y se devuelve `null`.

## Extensiones previstas

Runtime autónomo de Python por sistema y arquitectura; calendario oficial de bolsas; backtesting de estrategias con comisiones; calibración de incertidumbre; selección anidada de hiperparámetros; persistencia de proyectos; cancelación y progreso granular del motor. Estas extensiones no se presentan como funcionalidades implementadas.

## Laboratorio neuronal 1.1

`NetworkStudio` mantiene configuraciones por método en almacenamiento local. `engine.py` valida y normaliza las configuraciones seleccionadas y las incorpora a `experiment` para exportación. `features.py` construye indicadores incrementalmente, sin retrorelleno; `neural.py` genera modelos Keras a partir de una lista validada de capas, activaciones y optimizadores. No se evalúa código ni expresiones proporcionadas por el usuario.

La entrada declara la forma y una proyección Dense aplica su activación configurable. RNN/LSTM conservan la dimensión temporal hasta su última capa recurrente. CNN utiliza convoluciones causales y resumen temporal global. DQN aplana las ventanas de características y devuelve tres valores Q lineales. El cGAN combina contexto aplanado y ruido en G; D evalúa contexto y candidato. Ambos optimizadores se actualizan de forma alternada con pérdidas explícitas.

El resultado final incluye forma, número de parámetros, lista de capas, características, muestras, configuración y curvas de pérdidas cuando corresponda. Solo las características del entrenamiento ajustan los escaladores; en pasos futuros se actualizan con valores predichos. Se limita la matriz de entrenamiento a 20 millones de elementos, además de los límites previos de archivos y filas.

## Taller de infografías 1.2

Los canales IPC explícitos `unit_lab` y `demo_ohlc` pasan por el mismo origen validado y proceso Python acotado. `financial_lab.py` prepara precios OHLC, once indicadores seleccionables y etiquetas del siguiente intradía; `unit_lab.py` despacha a módulos separados según la tarea. Los modelos de reconstrucción, generación y trading no pasan por el ranking de precios del espacio general.

Las configuraciones de cada unidad se guardan en `forecast-lab-units-v1` de almacenamiento local. Resultados y datos de sesión permanecen en memoria y se exportan mediante el diálogo JSON nativo. Las imágenes originales se sirven desde recursos locales de Vite y se empaquetan dentro de dist; el manifest conserva nombres de origen y hashes SHA-256.

La selección supervisada excluye referencias Ridge/logística, aunque se reportan ambas métricas de validación/test. Los checkpoints se restauran en memoria antes de evaluación. U5 usa folds internos anteriores a validación externa, U9 purga ventanas completas y mantiene condición previa, U10 implementa Double DQN y costes con posición anterior durante gap. Las señales supervisadas operan únicamente el intradía siguiente.

## Banco de variantes y paquetes 1.3

`training_control.py` separa selección del mejor checkpoint y decisión de parada por mejora mínima/paciencia/calor. `emulator.py` construye variantes y retiene ganadores por validación; el test se evalúa solo después. `model_exports.py` clona modelos sin optimizador, guarda pesos NPZ y metadatos, y genera ZIP con contexto e inferencia. `bot_infer.py` verifica hashes y calcula salidas paper sin órdenes ni credenciales.

El proceso principal fija el directorio de artefactos en datos de la app, sin aceptar rutas del renderer. Los identificadores de descarga se validan como UUID hex de 32 caracteres; los diálogos nativos eligen destino. El progreso sale por stderr con prefijo cerrado y se reenvía por un canal IPC explícito sin exponer el evento Electron.
