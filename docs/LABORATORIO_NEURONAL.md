# Laboratorio neuronal y GAN condicional

El apartado **Laboratorio neuronal** permite editar una configuración independiente para **RNN, CNN 1D, LSTM, DQN y GAN condicional**. Activa «Comparar este método» para incluirlo en la siguiente ejecución. La configuración se guarda automáticamente en este equipo y se incorpora al JSON del experimento.

## Entrada, capas y salida

- La forma de entrada se obtiene de **ventana temporal × características**. Se conserva el orden de observaciones.
- La capa `Input` de Keras declara la forma. A continuación se coloca una **proyección Dense de entrada** con neuronas y activación configurables. Esta distinción evita atribuir una activación a un simple marcador de forma.
- Entre **una y cinco capas intermedias** por red. RNN apila SimpleRNN, LSTM apila LSTM, CNN apila Conv1D con padding causal, DQN usa Dense. Generador y discriminador del GAN usan Dense independientes.
- Cada capa tiene **neuronas o filtros, activación y dropout**. CNN permite definir kernel por capa. LSTM permite elegir la activación de las compuertas; valores distintos a tanh/sigmoid pueden cambiar estabilidad y aceleración disponible.
- Activaciones: `linear`, `relu`, `tanh`, `sigmoid`, `elu`, `selu`, `softplus`, `swish`.
- Salida supervisada y generador: **Dense(1), linear**.
- DQN: **Dense(3), linear**, un valor Q por bajar/mantener/subir. No son tres probabilidades ni tres precios.
- Discriminador: **Dense(1), sigmoid**, clasificación real/generado.
- Optimizadores por red: **Adam, RMSprop, SGD, Adamax, Nadam, Adagrad**. Tasa de aprendizaje editable y recorte de norma de gradientes a 1.
- Ventana: 2–120 observaciones. Unidades/filtros: 1–256. Dropout: 0–0.5. Épocas: 1–5,000; lote: 8–256. DQN usa los episodios de RL del panel de análisis.

## Características causales

La serie objetivo se incluye como característica base. Se pueden añadir:

| Característica | Definición y parámetros |
|---|---|
| Lags | Valor de hace k observaciones respecto a la fila histórica de la característica. Hasta 20 lags. |
| EMA | Media exponencial con α=2/(p+1), sembrada en la primera observación. Hasta seis periodos. |
| SMA | Media simple de las últimas p observaciones. Hasta seis periodos. |
| RSI | Suavizado de Wilder; semilla de ganancias/pérdidas medias de los primeros p cambios. Si no hay cambios, RSI=50; solo subidas, 100; solo bajadas, 0. |
| MACD | EMA rápida − EMA lenta, señal EMA y su histograma. La rápida debe ser menor que la lenta. |
| Variación relativa | Δy / max(abs(y anterior), 10⁻⁸), utilizable con series empresariales de signo variable. |
| Volatilidad | Desviación estándar muestral de los últimos p cambios Δy, en unidades de la variable. No es la volatilidad de retornos de GARCH. |

Los periodos se miden en observaciones. Con datos mensuales, RSI(14) usa catorce cambios mensuales. EMA/SMA admiten periodos de 2–365; lags de 1–365.

Una fila de características de tiempo t contiene información hasta t; la ventana que predice y(t+1) termina en t. Los lags adicionales se miden desde t: el valor base ya representa y(t) y lag_1 agrega y(t−1). Una ventana temporal y un lag pueden aportar información redundante; el laboratorio permite esa elección explícita.

Las filas de calentamiento se descartan y no se rellenan con valores futuros. MACD espera hasta `slow + signal − 2` como índice inicial; RSI necesita p cambios. Se informa un mínimo de entrenamiento por configuración: calentamiento + ventana + ocho ejemplos. Si una ventana de evaluación no alcanza ese mínimo, el reporte muestra la causa.

Las características y los objetivos se escalan **dentro de cada ajuste de entrenamiento**. El escalador de características usa únicamente filas que forman parte de ventanas de entrenamiento. En test y futuro, los indicadores se actualizan recursivamente con valores predichos, nunca con valores reales todavía no observados. La normalización y la transformación inversa se conservan durante esa recursión.

Las características derivan de la **variable objetivo elegida**. Esta versión no agrega regresores externos de otras columnas ni valores futuros conocidos de esas columnas.

## GAN implementado: cGAN condicional con reconstrucción

No es TimeGAN ni un generador incondicional de secuencias completas. Produce el siguiente valor condicionado a una ventana histórica de características.

```text
Ventana causal → aplanar → concatenar ruido z
                               ↓
                  Proyección G + 1–5 Dense
                               ↓
                      valor G(c,z), lineal
                               ↓
          concatenar con el mismo contexto c
                               ↓
                  Proyección D + 1–5 Dense
                               ↓
                       probabilidad, sigmoid
```

Discriminador:

```text
L_D = 0.5 × [BCE(1, D(c,y)) + BCE(0, D(c, stop_gradient(G(c,z))))]
```

Generador:

```text
L_G = BCE(1, D(c,G(c,z))) + λ × mean(abs(y − G(c,z)))
```

Se alternan una actualización del discriminador y una del generador por lote cronológico. Cada red utiliza su propio optimizador y tasa de aprendizaje. Al actualizar D se congela la salida de G; al actualizar G se calculan gradientes a través de D sin actualizar sus pesos. El dropout de D se desactiva durante la actualización del generador.

Parámetros adicionales: **dimensión latente 1–64**, **8–256 muestras por paso** y **peso L1 de 0–100**. Poner λ=0 elimina el término de reconstrucción. Un λ alto puede reducir la diversidad generativa.

El pronóstico central es la media de las muestras por paso. Los percentiles 2.5 y 97.5 describen el generador condicionado al contexto actual; el contexto de pasos posteriores usa la media previa. **No son trayectorias Monte Carlo independientes ni intervalos calibrados para todo el horizonte**. El modelo puede sufrir colapso de modos. La calidad se evalúa con las mismas ventanas temporales y test final de los demás modelos.

El dashboard muestra la arquitectura realmente ejecutada, forma de entrada, parámetros, características, muestras de entrenamiento y curvas de pérdida de G, D y reconstrucción. La academia agrega una lección de GAN con cinco preguntas: la academia ampliada de la versión 1.2 contiene 24 lecciones / 120 preguntas, incluidas las nuevas familias del taller U2–U10.

## Reproducibilidad y pruebas

El reporte JSON incluye `experiment.options.networks`: configuraciones efectivas de los métodos seleccionados, semillas y ventanas. `future.network` describe la red final entrenada; en GAN incluye también `discriminator`. La pérdida de entrenamiento no sustituye las métricas de validación.

```bash
npm run build
.venv/bin/python -m unittest discover -s tests -v
npm run smoke:neural
```

La prueba de escritorio configura ambas redes, verifica el límite de cinco capas, activa indicadores, entrena un GAN real desde la interfaz y comprueba la configuración y arquitectura en un reporte exportado.

Referencias de implementación: [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) y [optimizadores Keras](https://keras.io/api/optimizers/).

## Validación y exportación 1.3

El editor general incorpora validación interna cronológica y Early Stopping opcional con paciencia, minDelta y calentamiento. Sus escalas excluyen la cola interna de validación. El banco compara variantes por familia y exporta paquetes de inferencia; consultar `VARIANTES_Y_EXPORTACION.md`.
