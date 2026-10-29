# Banco de variantes, Early Stopping y exportación — versión 1.3

## Dónde está cada función

- **Laboratorio neuronal → Banco de variantes y exportación**, o la entrada **Banco de variantes** de la barra lateral.
- **Taller U2–U10 → Datos y protocolo temporal**: fechas de train, validación y test.
- **Validación y Early Stopping**: interruptor, paciencia, mejora mínima y calentamiento.
- Después de ejecutar: comparación de validación, épocas reales/máximo, mejor época y motivo de parada. Las curvas distinguen entrenamiento y validación.

## Entrenamientos de 1,500 épocas

El máximo admitido ahora es **5,000 épocas**. El botón **Usar máximo 1500 + Early Stopping** cambia únicamente el presupuesto y la parada, conservando las capas y características editadas.

Presets nuevos: techo 1,500, Early Stopping activado, paciencia 30, mejora mínima 0.00001 y diez épocas de calentamiento. Los valores ya guardados por el usuario se conservan. U3/U4 mantienen un presupuesto inicial de 100 para sus comparaciones educativas; el usuario puede elevarlo.

- **Activado:** tras el calentamiento, se detiene cuando transcurren `patience` épocas sin una mejora acumulada mayor que `minDelta` en el criterio de validación.
- **Desactivado:** ejecuta todas las épocas solicitadas, salvo error o tiempo máximo.
- **Ambos estados:** restaura los pesos de la época con el mejor valor real de validación. La selección del mejor checkpoint es independiente del umbral mínimo que reinicia la paciencia.
- **Calentamiento:** impide parar prematuramente; aun durante esas épocas se conservan los mejores pesos observados.
- El test nunca determina la parada.

`minDelta` se interpreta en la escala del criterio: MSE normalizado para regresión, BCE para dirección, MSE de indicadores normalizados en AE/DAE y distribución estandarizada para generativos. No es un umbral universal expresado en bps.

U3 permite activar/desactivar ES: apagado iguala épocas; encendido mantiene un techo común, pero cada optimizador puede ejecutar un número distinto de épocas. U4 permite incluir/excluir su variante EarlyStopping; las otras variantes conservan el presupuesto completo para estudiar los mecanismos por separado. Ridge/PCA no tienen épocas. DQN/Double DQN usan episodios y selección del mejor episodio de validación, no parada por épocas.

El tiempo máximo de una ejecución del taller/banco es configurable entre 1 y 240 minutos, con 60 por defecto. El análisis general utiliza 120 minutos. Subir el máximo de épocas no elimina ese límite.

## Validación del editor general

RNN/CNN/LSTM y el GAN predictor del editor general ahora reservan una cola cronológica interna del ajuste (20% por defecto; configurable entre 10–40%). Los escaladores y los pesos aprenden exclusivamente del bloque anterior. La validación interna se encuentra **dentro del entrenamiento externo**: el test externo no interviene.

El modelo final conserva esa separación y restaura su mejor checkpoint interno; no se presenta una refit adicional con etiquetas de validación como si fuera el mismo checkpoint. Se informa cuántas muestras aprendieron los pesos y cuántas se reservaron. Las características futuras siguen calculándose causalmente con predicciones o historia observada, según el protocolo de cada espacio.

Para el GAN predictor, el criterio de parada es MSE interno con ruido fijo. Para CVAE/cGAN de ventanas U9, es similitud de distribución en validación con ruido fijo. Se restauran conjuntamente los pesos de las redes que forman el checkpoint, no solo una parte.

## Qué explora el banco

Se elige una tarea, uno o varios activos OHLC diarios, familias y variantes por familia. El espacio editable incluye anchos, profundidades, ventanas, dimensiones latentes, dropout, optimizadores y learning rate log-uniforme. Se utiliza una semilla reproducible y se informa progreso por variante.

Familias/pipelines disponibles:

| Tarea | Familias | Criterio principal |
|---|---|---|
| Pronóstico de retorno intradía siguiente | Ridge, MLP, CNN, RNN, LSTM, GRU, BiLSTM, BiGRU, AE+Ridge, DAE+Ridge, PCA+Ridge | MSE de validación menor; se reporta R² sobre las mismas etiquetas |
| Dirección | Logística, MLP, CNN, RNN, LSTM, GRU, BiLSTM, BiGRU | BCE de validación menor; AUC, Brier, accuracy y F1 reportados |
| Escenarios de ventanas | CVAE y cGAN | Wasserstein + diferencia de desviación en validación estandarizada |
| Trading simulado | DQN y Double DQN | Retorno neto de validación mayor, con costes |

Para regresión:

```text
R² = 1 − MSE / var(y)
```

Con las mismas filas/etiquetas, maximizar R² equivale a minimizar MSE cuando hay varianza. Por eso no se presenta una falsa optimización de dos objetivos independientes. El máximo lookback del espacio determina las filas comunes: una variante de ventana corta no obtiene una evaluación más fácil por usar otras fechas. Si hay empate exacto en el criterio, se prefiere menor número de parámetros.

AE/DAE/PCA+Ridge se comparan como pipelines predictivos: el extractor aprende representación y Ridge se ajusta con factores/etiquetas de train. Su error de predicción se distingue del de reconstrucción de U8. En clasificación no se fuerza R² sobre probabilidades. En escenarios y trading tampoco se utiliza MSE de retornos como sustituto de distribución o rentabilidad.

Una vez terminada toda la búsqueda, se evalúa **solo el ganador de cada familia en test**. Ese test no selecciona variantes ni la familia principal. El informe conserva todas las variantes, fallos, configuración, resultados de validación, escaladores y evaluación final de los ganadores. «Mejor» significa mejor entre las configuraciones ensayadas para la tarea y periodos elegidos; no demuestra rentabilidad futura.

La métrica se agrupa sobre los activos seleccionados; el paquete conserva el contexto individual de cada activo. Conviene revisar también el desempeño por activo y por periodo antes de una integración real.

## Descargas

- **Descargar resultado JSON:** configuración, variantes, criterios, métricas, ganadores, escalas e identificadores/hashes de paquetes.
- **JSON de familia:** resultado/configuración del ganador individual.
- **Modelo + pesos + inferencia:** ZIP autosuficiente de esa familia.
- **Descargar todos los paquetes:** ZIP con una carpeta por ganador y `selection_report.json` en la raíz.

Los ZIP de redes contienen:

```text
manifest.json                    # Configuración, escalas, cortes, métricas y hashes
predictor.keras / online.keras / generator.keras / encoder.keras
*.architecture.json              # Arquitectura Keras serializada
*.weights.npz                    # Arrays de pesos en orden descrito por el manifest
ridge.json / pca.json / logistic.json   # Cuando corresponde, sin pickle
context_0.csv, context_1.csv ...  # OHLC histórico para reproducir indicadores
bot_infer.py                     # Inferencia CLI
financial_lab.py, neural.py, features.py, training_control.py
requirements.txt                 # Versiones utilizadas
risk.json                        # Modo paper y requisitos básicos
README.md
```

Cada familia incluye únicamente los modelos necesarios: por ejemplo, Ridge/PCA pueden utilizar coeficientes numéricos JSON sin TensorFlow para inferir; CVAE incluye encoder y decoder; cGAN incluye generador/discriminador; los agentes incluyen online/target. Los modelos Keras se exportan sin estado de optimizador y con los pesos seleccionados. El paquete sirve para **inferencia**, no para reanudar exactamente el estado del optimizador en el checkpoint.

Los hashes SHA-256 detectan modificaciones/daños de archivos. La inferencia verifica el paquete antes de utilizarlo. No se utiliza pickle para Ridge/PCA/logística.

## Integración de inferencia en un bot

Descomprime un paquete individual o entra en una carpeta del ZIP global:

```bash
python -m venv .venv
# Activa el entorno según tu sistema
pip install -r requirements.txt
python bot_infer.py --bundle . --output signal.json
python bot_infer.py --bundle . --csv barras_nuevas.csv --output signal.json
```

El CSV debe tener `date`, `Open`, `High`, `Low`, `Close`, barras diarias cerradas y la misma convención de ajuste que el entrenamiento. El script combina las nuevas barras con el contexto exportado y vuelve a calcular los indicadores causales. `--asset 0` elige el primer activo; `--asof FECHA` corta el contexto para inferencia retrospectiva. No convierte este modelo ya seleccionado en un checkpoint histórico anterior al periodo de selección.

Salidas:

- Regresión: retorno intradía esperado en bps y señal experimental -1/0/+1 con umbral aproximado de costes.
- Clasificación: probabilidad de alza, clase y señal largo/efectivo con umbral 0.5.
- Trading: tres valores Q y posición objetivo, usando `--position -1/0/1` para informar posición actual.
- Generación: ventanas de retornos y escenarios de precios condicionados por volatilidad anterior; `--scenarios 32` configura su cantidad.

El ejemplo funciona en **modo paper**: no contiene credenciales ni envía órdenes. La conexión a broker, dimensionamiento de posiciones, calendario de bolsa, persistencia de órdenes, ejecución, financiación y controles operativos se integran por separado. Los escenarios no son intervalos calibrados.

## Ejemplo entrenado entregado

Se generó una demo OHLC **sintética** de 2020–2025. Se exploraron dos variantes de Ridge, MLP y LSTM, con techo de 1,500 épocas, paciencia 15, calentamiento 10 y `minDelta=0.0001`.

- Ganador LSTM: ejecutó 54 épocas.
- Ganador MLP: ejecutó 35 épocas.
- Ridge no utiliza épocas.

La familia principal por validación fue LSTM en esa ejecución. El resultado no constituye evidencia de una estrategia rentable con SPY: su finalidad es demostrar selección, exportación y recarga reproducible. Se entregan `seleccion-variantes-demo.json` y `modelos-demo-inferencia.zip` por separado del código fuente.

## Verificación

Las pruebas cubren interruptor activado/desactivado, paciencia, mejora mínima, calentamiento, techo 1,500, restauración de pesos, separación de escalas, búsqueda/selección por familia, test solo de ganadores, exportación/recarga de todas las familias, equivalencia de métricas del modelo cargado, paquetes globales y detección de archivos alterados.

La prueba de Electron descarga JSON y ZIP mediante los diálogos nativos, extrae el modelo descargado y ejecuta su script de inferencia. También verifica que ES activado detiene una prueba controlada en tres épocas y desactivado completa las cuatro solicitadas. Esa prueba controlada usa un `minDelta` grande deliberadamente para hacer determinista la parada; no es una recomendación de entrenamiento.
