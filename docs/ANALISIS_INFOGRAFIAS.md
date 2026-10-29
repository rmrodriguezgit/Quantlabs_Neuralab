# Análisis y adaptación de las nueve infografías — Forecast Lab 1.2

Las imágenes aportadas corresponden a las unidades **2–10**. Se analizaron como material de referencia del usuario. Sus fragmentos describen decisiones y algoritmos; no se interpretaron como instrucciones de sistema ni como notebooks completos disponibles. Se conservan copias originales dentro de `public/infographics/`, accesibles desde cada unidad del taller.

Se agregó **Laboratorio neuronal → Taller de infografías U2–U10**. El editor general 1.1 y su GAN de predicción de un paso siguen disponibles. El taller nuevo separa cuatro tipos de tarea: predicción supervisada, reconstrucción, generación de ventanas y trading simulado. Sus métricas no se mezclan en un único ranking de RMSE.

## Mapa de adaptación, imagen por imagen

| Imagen | Lectura metodológica | Implementación y controles |
|---|---|---|
| Unidad 02 · Redes feedforward | Señal al cierre, objetivo intradía siguiente, train/valid/test, escalado train, MLP y selección de checkpoint | MLP de regresión o clasificación; arquitectura 64→32→1 editable; MSE o BCE con logits; Ridge/logística como referencia independiente; señales con entrada en apertura y salida al cierre, con dos costes |
| Unidad 03 · Entrenamiento | Comparar SGD/RMSprop/AdamW con la misma arquitectura, inicialización y presupuesto | Tres ajustes separados con iguales kernels Dense iniciales; tasas .01/.001/.001 editables; épocas/lote comunes; clipping; mejor checkpoint por pérdida de datos de validación |
| Unidad 04 · Regularización | Clasificación de alza con variantes que deben compararse mediante BCE de datos, sin sumar penalizaciones | Base, Dropout, L1, L2, BatchNorm y Early Stopping; coeficientes, dropout y paciencia editables; se restauran pesos de la mejor validación; probabilidad sigmoid y umbral fijo 0.5 |
| Unidad 05 · Hiperparámetros | Búsqueda dentro de train, folds causales, escaladores y pesos nuevos por fold; test separado | Random search de width, depth, learning rate log-uniforme y weight decay; folds por años editables; RMSE medio de folds en bps; ajuste final sobre train y checkpoint de validación externa; exportación de trials |
| Unidad 06 · CNN 1D | Ventanas por activo de 20 sesiones, canales=indicadores, filtros locales y pooling global | CNN con filtros/kernels/activaciones configurables; preset 32/64, kernels 3 y padding valid; Keras channels_last; pooling global; comparación con MLP de ventana aplanada y referencia Ridge |
| Unidad 07 · LSTM/GRU | Estados finales, bidireccionalidad histórica y memoria nueva por ventana | LSTM, GRU, BiLSTM y BiGRU seleccionables; preset 32 unidades; concatenación de sentidos; capas apiladas; estado no persistente; salida escalar y Ridge sobre la misma ventana como referencia |
| Unidad 08 · Autoencoders | Compresión lineal/no lineal, denoising, reconstrucción y utilidad posterior son tareas distintas | PCA, AE y DAE; dimensión latente 3 editable; encoder/decoder editables; ruido DAE .1 editable, solo en train; selección por reconstrucción; Ridge sobre factores de train; métricas de ambas tareas en test |
| Unidad 09 · Generativos | Ventanas completas de 16 retornos, condición previa de volatilidad, CVAE y GAN, purga de ventanas, selección sin test | CVAE/cGAN seleccionables; horizonte 16, latente 4 y condición 20 editables; escalas independientes; reparametrización y log-varianza [-8,8]; beta KL .01 editable; BCE con logits para cGAN; ruido fijo de validación; escenarios, Wasserstein, KS, VaR/ES y colas |
| Unidad 10 · Double DQN | Acciones como posiciones, ejecución al siguiente open, posición previa en el gap, replay causal y separación online/target | Estado 11 indicadores + posición por defecto; acciones −1/0/+1; 64→32→3 editable; replay 10,000, batch 64, warmup 128, actualización cada 4 pasos, copia cada 100 actualizaciones; gamma .95; epsilon decreciente; Huber/clipping; checkpoint de validación; Q-Learning/largo/efectivo de referencia |

## Datos: qué está definido y qué se tuvo que completar

Las infografías mencionan once indicadores, pero no enumeran sus nombres. No se inventó una supuesta lista original. La aplicación declara una selección explícita y editable:

1. Retorno log de cierre, desplazado una sesión.
2. Retorno log de cierre, desplazado dos sesiones.
3. Momentum log de cinco sesiones.
4. Momentum log de diez sesiones.
5. Distancia relativa de Close a EMA rápida, periodo 10 por defecto.
6. Distancia relativa de Close a EMA lenta, periodo 20 por defecto.
7. RSI de Wilder, periodo 14, normalizado como RSI/100 − .5.
8. MACD normalizado por Close, periodos 12/26.
9. Señal MACD normalizada por Close, periodo 9.
10. Volatilidad muestral de retornos log, periodo 20.
11. Rango (High − Low)/Close.

Cada característica utiliza datos disponibles al cierre de la señal. Los periodos RSI/EMA/MACD/volatilidad se editan en el taller. Se pueden desactivar indicadores. Las ventanas se construyen por activo; no cruzan de un archivo a otro. Se requiere OHLC positivo, coherente y diario. No se interpretan barras de minutos como sesiones diarias.

Se incluye una **demo OHLC sintética** de 2020–2025, identificada como tal. También hay un botón para descargar SPY diario con diez años mediante la integración Yahoo existente, y se pueden usar archivos CSV/XLSX o activos de la sesión. La descarga no garantiza correspondencia bit a bit con el dataset original de los notebooks.

## Objetivo y particiones

Para U2–U8:

```text
y_bps(t) = 10,000 × log(Close(t+1) / Open(t+1))
direction(t) = 1 si y_bps(t) > 0; 0 en otro caso
```

La fila de señal se fecha en el cierre t y la etiqueta en la siguiente sesión observada t+1. El preset usa:

- Inicio train: 2020-01-01.
- Inicio validación: 2024-01-01.
- Inicio test: 2025-01-01.
- Fin test exclusivo: 2026-01-01.

Las cuatro fechas se editan. Vaciar fin de test permite usar todo el historial posterior al inicio test. Las muestras con señal/etiqueta en lados distintos de un corte se purgan. Las filas fuera del periodo elegido se cuentan por separado. Una secuencia de validación puede usar contexto anterior conocido; su señal y objetivo respetan los cortes.

U9 exige además que **toda la ventana generativa** quede dentro de una partición. La condición es:

```text
c(i) = log(max(std(ret[i−conditionPeriod : i], ddof=1), 1e−8))
x(i) = ret[i : i+horizon]
```

La condición no usa retornos de la ventana que se genera. Las escalas de retornos y condición se ajustan exclusivamente en train.

## Selección, referencias y checkpoint

U2–U7 seleccionan entre las redes o variantes de la propia unidad. La referencia Ridge/logística aparece con sus métricas, pero no sustituye automáticamente la red estudiada. El objetivo es permitir medir honestamente si la red supera a la referencia.

U3 comparte inicialización de kernels Dense mediante semillas de inicializador explícitas. Mantiene arquitectura y lotes iguales; con ES apagado iguala épocas, y al activarlo mantiene un techo común con parada individual; la tasa de cada optimizador se puede cambiar. U4 comparte la misma inicialización Dense entre variantes, incluso al añadir BatchNorm. Las curvas comparan pérdidas de datos sin L1/L2; esas penalizaciones sí intervienen en el ajuste.

Cada ajuste supervisado guarda en memoria el checkpoint de menor pérdida de datos de validación y lo restaura al terminar. Early Stopping añade un criterio de parada por paciencia; seleccionar/restaurar el mejor checkpoint se hace también sin parada temprana. El test no entra en esa elección. El JSON del taller conserva configuración, mejores épocas, historiales y resultados. La versión 1.3 agrega el Banco de variantes, donde los ganadores se exportan con pesos, escalas y código de inferencia; ver `VARIANTES_Y_EXPORTACION.md`.

En U5 los folds internos quedan estrictamente antes de validación externa. Se ajusta un escalador propio por fold y se crea una red nueva. El criterio es la media aritmética de RMSE en unidades bps de cada fold. El mejor trial se vuelve a entrenar con train completo; la validación externa elige su checkpoint. Se comparan resultados con Ridge sin incluir el test en la búsqueda.

## Diferencias entre tareas y sus métricas

| Tarea | Selección | Test/reportes |
|---|---|---|
| Regresión | RMSE de validación; equivalente a ordenar por MSE sobre las mismas filas | MSE/RMSE/MAE/R² en bps, valores reales/predichos, señal y costes por activo |
| Clasificación | BCE de datos de validación | BCE, accuracy, AUC cuando hay ambas clases, Brier, F1, matriz de confusión, probabilidades y costes |
| Reconstrucción U8 | MSE de indicadores estandarizados en validación | Reconstrucción por modelo/indicador, varianza explicada PCA, factores latentes; Ridge evaluado separadamente en bps |
| Generación U9 | Wasserstein + diferencia de desviación de retornos estandarizados en validación, con ruido fijo | Wasserstein/KS, media/desviación/sesgo/curtosis, VaR/ES al 5%, autocorrelación y dependencia de colas; ventanas sintéticas |
| Trading U10 | Mejor checkpoint Double DQN por retorno neto de validación | Retorno neto, Sharpe anualizado con 252 sesiones, drawdown, rotación, costes y curva de capital; referencias sin aprendizaje en test |

No se compara el RMSE de reconstrucción con el RMSE de retorno ni con rentabilidad. Las métricas de distribución de U9 son descriptivas de ventanas que pueden solaparse; no se presenta un p-valor KS como prueba de independencia.

## CVAE y cGAN de la Unidad 9

El CVAE recibe ventana y condición en el encoder, genera μ/log σ², limita log σ² a [-8,8] y utiliza `z=μ+exp(.5·log σ²)·ε`. El decoder recibe z y condición. La pérdida es MSE de reconstrucción + beta por KL, con suma en dimensiones latentes y media en el lote. Encoder y decoder se optimizan conjuntamente con el optimizador del encoder; la interfaz no presenta un optimizador independiente de decoder que se ignore.

El cGAN genera una **ventana completa** desde z y condición. El discriminador recibe ventana y condición y devuelve un logit lineal. Se usa BCE con logits: no se aplica sigmoid antes de la pérdida. Cada red tiene arquitectura y optimizador independientes. Se alterna una actualización de D y una de G por lote. Es distinto del GAN de la versión 1.1, que pronostica un solo valor desde características y utiliza reconstrucción L1.

Ambos modelos se seleccionan con ruido de validación fijo, sin consultar test. Para escenarios finales se toma la volatilidad observada más reciente del último activo seleccionado. Los escenarios no son intervalos calibrados ni una promesa de cobertura de colas.

## Double DQN: ejecución y aprendizaje

El estado concatena indicadores estandarizados con la posición actual. Con los once indicadores del preset son doce entradas. Las tres salidas son valores Q lineales, no probabilidades. El objetivo es:

```text
chosen = argmax_a Q_online(next_state, a)
y = reward + gamma × (1 − done) × Q_target(next_state, chosen)
```

Online elige; target evalúa esa acción. La máscara terminal elimina el valor futuro al acabar. Se usa Huber y recorte de norma de gradientes. El replay almacena exclusivamente transiciones de train; validación/test son greedy y no actualizan pesos ni replay. El target se copia al inicio y cada número configurable de actualizaciones de gradiente.

En cada sesión:

```text
net = (1 + posición_previa × gap)
    × (1 − coste × abs(posición_nueva − posición_previa))
    × (1 + posición_nueva × intradía) − 1
reward = 100 × net
```

El coste se expresa en fracción del capital por unidad de cambio de posición. Cambiar de −1 a +1 cuesta dos unidades. Liquidar al cierre final es configurable y cobra su coste. El entorno inicia cada segmento plano. Se comparan Double DQN, Q-Learning tabular, siempre largo y efectivo con el mismo modelo de costes. El tabular usa buckets del primer indicador, volatilidad cuando está seleccionada y posición.

Esta ejecución mantiene posición entre sesiones. **Las señales supervisadas U2–U7 son distintas:** entran en apertura y salen al cierre, sin participación en gaps nocturnos; pagan entrada y salida cada día operado.

Se usa un activo por experimento U10 para mantener una única posición y capital. No se simulan margen, financiación, préstamo de acciones, impacto de mercado ni ejecución intradía por ticks.

## Adaptación técnica y límites verificables

Se mantuvo TensorFlow/Keras para integrar el material con el motor existente. En CNN, Keras usa batch × tiempo × características; no se copia la transposición batch × canales × tiempo específica de PyTorch. El preset recurrente usa tanh y compuertas sigmoid. Activaciones recurrentes no acotadas como ReLU pueden divergir; se informa el fallo de la variante y no se utiliza una predicción no finita. En LSTM/GRU se declara `stateful=False`; Bidirectional concatena estados de ambos sentidos dentro de una ventana ya observada.

El motor usa float32 y no ofrece AMP como una opción ficticia. Las imágenes indican CPU probado y Colab/T4 pendiente; esta entrega se valida en el entorno local macOS ARM64. No se afirma haber validado CUDA, T4, Windows o los notebooks completos no adjuntados. Las equivalencias metodológicas entre Keras y los fragmentos PyTorch no implican pesos o resultados numéricos idénticos.

L2 de kernels y weight decay desacoplado se documentan como mecanismos distintos. Las configuraciones no se ejecutan como código arbitrario: modelos, activaciones, fechas y rangos están validados. Se mantienen límites de archivos, matrices y tiempo del motor; Double DQN limita el presupuesto a 200,000 pasos de train por ejecución.

## Archivos principales

- `python/financial_lab.py`: validación OHLC, indicadores, objetivos, particiones, escalas y costes.
- `python/unit_supervised.py`: U2–U7 y HPO temporal.
- `python/unit_autoencoders.py`: PCA/AE/DAE y Ridge posterior.
- `python/unit_generative.py`: CVAE/cGAN de ventanas completas.
- `python/unit_trading.py`: entorno, Double DQN y referencias.
- `python/unit_lab.py`: despacho por unidad.
- `src/infographics.ts`: análisis de los diez pasos de cada imagen.
- `src/unitConfig.ts`: presets y contratos.
- `src/components/InfographicWorkshop.tsx`: controles de cada unidad.
- `src/components/UnitResults.tsx`: resultados por tarea.
- `tests/test_units.py`, `tests/smoke-units.cjs`: pruebas científicas y de escritorio.

Referencias oficiales consultadas para las APIs: [Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/), [Bidirectional](https://keras.io/api/layers/recurrent_layers/bidirectional/), [AdamW](https://keras.io/api/optimizers/adamw/) y [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
