# Validación de la entrega — 28 de octubre de 2026

## Comprobaciones realizadas

- Compilación de los procesos de Electron con TypeScript estricto, comprobación del frontend y bundle de Vite: correcta.
- **67 pruebas Python aprobadas**, sin omisiones, sobre Python 3.12 en macOS ARM64.
- Modelos estadísticos: persistencia, regresiones lineal/cuadrática, ARIMA, SARIMA, GARCH y Monte Carlo producen arrays finitos de longitud correcta en un caso sintético de prueba.
- Modelos avanzados: RNN, CNN, LSTM y Prophet producen cinco pronósticos finitos; Q-Learning y DQN ejecutan su entorno y producen cinco salidas. Redes probadas con una época y agentes con diez episodios: se comprueba funcionamiento, no calidad predictiva ni convergencia de aprendizaje.
- Identidad de descomposición TR + VA + TE; métricas con resultados conocidos; R² indefinido en serie constante; reproducibilidad de simulación por semilla.
- Comprobación mediante registro de longitudes de entrenamiento de que el test final no se utiliza en el ranking.
- Horizonte sin suficiente historial informa error y no asigna ganador.
- Exportación e importación CSV/XLSX, archivo inválido y respuesta JSON limpia por stdout.
- **Electron real**, automatizado con Playwright: carga de interfaz, llamada por preload a Python, simulación empresarial, ejecución de persistencia/regresión, dashboard y ranking, pregunta educativa con respuesta y retroalimentación. Sin errores JavaScript. Capturas en `inicio.png`, `dashboard.png`, `academia.png`.
- **Yahoo Finance real**: descarga MSFT a resolución diaria y rango de un año; se recibieron 252 filas con Close, High, Low, Open y Volume. No se verificaron todos los símbolos ni intervalos.
- `start_mac.sh`: validación de sintaxis con `bash -n`. El arrancador Windows se revisó estáticamente y se normalizó a CRLF; no se ejecutó en Windows.

## Reproducir

```bash
npm run build
.venv/bin/python -m unittest discover -s tests -v
npm run smoke
```

Windows: sustituir el ejecutable por `.venv\Scripts\python.exe`. Para verificar todos los métodos, instalar primero `python/requirements-advanced.txt`; las seis pruebas avanzadas se omiten explícitamente si falta la dependencia correspondiente. `smoke` requiere sesión gráfica y usa el ejecutable Electron local; acepta `FORECAST_PYTHON` para apuntar a otro entorno. Genera capturas en `docs/` y usa un perfil temporal de Electron.

## Ampliación 1.1: laboratorio neuronal y GAN

- Trece pruebas adicionales cubren invariancia causal ante cambios del futuro, lags/EMA/SMA/MACD con valores de referencia, RSI creciente/decreciente/plano, volatilidad móvil, coincidencia de cálculo incremental con cálculo por lote, escalas y ventanas de entrada, rechazo de configuraciones inválidas.
- Construcción e inferencia con una, tres y cinco capas para RNN, LSTM, CNN y DQN. Comprobación de proyección de entrada, salida lineal y optimizador RMSprop realmente aplicados.
- Un paso de entrenamiento con cada uno de seis optimizadores: Adam, RMSprop, SGD, Adamax, Nadam y Adagrad.
- Pronósticos reales con indicadores activados en redes RNN, LSTM y CNN; DQN con arquitectura personalizada.
- Entrenamiento cGAN con arquitecturas y optimizadores independientes. Se comprueban salida lineal del generador, salida sigmoide del discriminador, longitudes de escenarios, pérdidas finitas, ranking temporal y exportación de metadatos.
- **Prueba de interfaz 1.1 aprobada** con Electron real: edición de entrada, activaciones y optimizadores independientes de G/D; agregar hasta cinco capas y verificar el límite; activar lags, EMA, SMA, RSI, MACD, variación y volatilidad; entrenar GAN con un dataset mensual; mostrar pérdidas y métricas; exportar y comprobar JSON con las configuraciones y capas realmente ejecutadas. Sin errores JavaScript. Capturas: `redes-configurables.png` y `gan-dashboard.png`.
- Los seis tests avanzados originales y los trece nuevos se ejecutaron con TensorFlow/Gymnasium/Prophet instalados; no hubo omisiones.

## Ampliación 1.2: nueve infografías U2–U10

- **21 pruebas nuevas**, añadidas a las treinta anteriores: OHLC/objetivo intradía, causalidad de indicadores, purga señal/etiqueta y ventanas generativas completas, fechas de inicio/fin, separación de activos, escalas exclusivas de train y costes intradía frente a posiciones nocturnas.
- Ejecución real de las nueve unidades con datos sintéticos diarios 2020–2025: regresión y clasificación MLP, SGD/RMSprop/AdamW, seis variantes de regularización, HPO con folds internos, CNN y MLP de referencia, LSTM/GRU/BiLSTM/BiGRU, PCA/AE/DAE y Ridge, CVAE/cGAN, Double DQN y referencias de trading.
- Verificación de kernels Dense iniciales idénticos entre variantes con/sin BatchNorm; trials agregan RMSE de folds en bps; el test queda fuera de HPO y checkpoint.
- Prueba numérica específica del objetivo Double DQN: online elige y target evalúa esa acción; máscara terminal. Replay y pasos de aprendizaje exclusivamente de train.
- Entradas no acotadas pueden producir inestabilidad recurrente: la protección de valores no finitos rechaza esas variantes y las informa. Los presets recurrentes tanh/sigmoid y las cuatro familias pasaron.
- **Prueba de escritorio U2–U10 aprobada**, con carga y decodificación de las nueve imágenes originales, presets, demo OHLC, controles editables, ejecución Python real, selección de cada familia, métricas y nueve exportaciones JSON comprobadas. Sin errores JavaScript. Capturas de U7, U8, U9 y U10 en docs/.
- La selección U2–U7 estudia redes/variantes de cada unidad. Las referencias Ridge/logística se reportan en validación y test sin sustituir automáticamente el modelo neuronal seleccionado.
- Academia: **24 lecciones / 120 preguntas**. Se validó el render KaTeX de todas las fórmulas y cinco preguntas por lección.

Comandos adicionales: `npm run smoke:units` y `npm run units`. Para detalles del análisis de cada imagen y equivalencias PyTorch/Keras, ver `ANALISIS_INFOGRAFIAS.md`.

## Ampliación 1.3: épocas, parada y banco exportable

- **16 pruebas nuevas**, sobre las 51 anteriores, para un total de 67: techo de 1,500 y límite de 5,000, interruptor ES activado/desactivado, paciencia, mejora mínima, calentamiento y restauración del mejor checkpoint.
- Entrenamiento real con máximo 1,500 y criterio controlado: ES se detiene en tres épocas; apagado completa cuatro. Se verifica la pérdida del modelo restaurado, no solo las etiquetas del reporte.
- Validación interna del editor general excluida de las escalas y del entrenamiento; GAN predictor y redes utilizan el mismo controlador de parada.
- Banco: ganadores por familia, mismos objetivos/filas para MSE/R², test evaluado solo para ganadores después de toda la búsqueda. Se comprobaron todas las familias predictivas y la recarga de clasificación, generación y agentes.
- Exportación/recarga real de modelos .keras, pesos NPZ y parámetros numéricos JSON; el predictor recargado conserva sus métricas de test dentro de la tolerancia numérica. Paquetes globales e integridad SHA-256 verificados.
- **Prueba Electron aprobada**: controles de 1,500, ambos estados de ES, variantes y progreso, descarga JSON, ZIP individual/global, extracción del ZIP descargado y ejecución real del script de inferencia. Sin errores JavaScript.
- Ejemplo sintético con máximo 1,500 y paciencia 15: ganador LSTM ejecutó 54 épocas; MLP ejecutó 35. JSON y ZIP en `examples/`; no es un entrenamiento validado con SPY.
- El script exportado opera en paper, no contiene credenciales ni envía órdenes. No se validó una conexión con un broker ni ejecución real.

Comandos adicionales: `npm run smoke:emulator` y `npm run emulator`. Formato y metodología en `VARIANTES_Y_EXPORTACION.md`.

## Alcance pendiente

No se generaron ni firmaron instaladores DMG/NSIS; no se ejecutó la app en Windows. La configuración de electron-builder está incluida, pero su distribución y firma requieren una verificación adicional en cada sistema destino. Python no se embebe en un binario autónomo: el paquete solicita seleccionar una instalación con los requisitos.

No se certifica rentabilidad ni precisión general de modelos. La evaluación depende de los datos y del tamaño de cada ventana. No hay calendario de mercado, streaming, regresores externos libres, persistencia de proyectos ni cancelación manual. La búsqueda automática MLP por folds internos sí está disponible en U5; no es una búsqueda universal de todos los métodos. La interfaz y el README explican estas restricciones.

TensorFlow puede advertir sobre retracing durante las pruebas que construyen numerosas arquitecturas distintas. Las pruebas concluyen correctamente.

Vite advierte que el bundle principal supera 500 kB; la compilación concluye correctamente. Prophet advierte que Plotly no está instalado; sus predicciones funcionan y la aplicación usa Recharts, por lo que no depende de los gráficos de Prophet.

## Reingeniería 1.4

78 pruebas Python aprobadas (61 s), incluidas once nuevas del endpoint de comprobación previa. Build aprobado. Pruebas Electron de guía y emulador aprobadas: la ejecución requiere una revisión vigente, y las entradas inválidas o cambios posteriores la bloquean. Véase [recorrido guiado](REINGENIERIA_USO.md).

## Identidad 1.4.1

Build aprobado y prueba Electron `node tests/smoke-brand.cjs` aprobada: nombre, título, versión visible 1.4.1 e icono SVG cargado. Iconos PNG, ICNS e ICO incluidos; los instaladores de Windows no se ejecutaron en macOS.

## GAN temporal 1.5.0

- 87 pruebas Python aprobadas: 78 anteriores y nueve nuevas de condiciones, causalidad, escalas, volumen, regímenes, dinámica temporal, diversidad, ambas expansiones convolucionales, serialización e inferencia exportada.
- Después del ajuste final de activación de la proyección, las nueve pruebas cDCGAN se repitieron y aprobaron.
- Build de TypeScript/Vite aprobado.
- `npm run smoke:cdcgan` aprobado en Electron: U9 con cinco condiciones, cDCGAN, embeddings, sensibilidad al régimen con ruido fijo, diagnóstico, banco de variantes, exportación ZIP y ejecución del CLI desde el paquete extraído.
- Pruebas realizadas con datos sintéticos y presupuesto corto; verifican funcionamiento, no calidad predictiva ni calibración financiera.

## Infografía complementaria 1.5.1

Build aprobado y `npm run smoke:infographics` aprobado: ambas láminas de Unidad 9 se decodifican en Electron y sus hashes coinciden con el manifiesto. La imagen original no se modificó. Revisión visual de la nueva lámina realizada antes de integrarla.

## Perfiles 1.6.0

90 pruebas Python aprobadas en 67 s, incluidas comprobaciones de todos los perfiles contra el preflight real y conservación de metadatos de red. `npm run test:profiles` aprobado: 14 perfiles de unidades, cinco redes y 24 búsquedas. Build aprobado. Prueba Electron de perfiles aprobada: aplicación, edición libre, recuperación, persistencia y navegación entre unidades.

## Auditoría MSE/R² y Forex 1.7.0

- 105 pruebas Python aprobadas (70 s), incluidas precisión de MSE/R², invariancia de R² a escalado afín, rechazo de shapes inválidos, ajuste TR de Standard/Robust/MinMax/None, desescalado, deduplicación de ventanas, causalidad, barras cerradas, TE configurable y reconstrucción de cotizaciones desde diferencias/log-retornos.
- `npm run smoke:fx-audit` aprobado con precios sintéticos: preparación Forex RNN/LSTM/CNN, auditoría visible, fechas/scalers sin cruce de VA/TE, pips y exportación JSON.
- Prueba adicional con EURUSD=X diario real: los 14 métodos de Análisis y Simulación completaron tres folds externos de veinte pasos. TE se consultó solo para el ganador de VA. Informe y limitaciones en `REVISION_MSE_R2.md`; resumen reproducible en `examples/auditoria-eurusd-resumen-v1.7.json`.
- Comparación adicional RNN/LSTM de objetivo/scaler exclusivamente en VA; no se usó TE para retocar parámetros. Son resultados de un presupuesto acotado, no un óptimo demostrado.
- Build TypeScript/Vite y verificación de perfiles/precisión JavaScript aprobados.

## Estrategias por etapas 1.8.0

- **113 pruebas Python aprobadas** (76 s), incluidas ocho nuevas: cambiar TE no altera el ganador, mezcla elegida en búsqueda, TE solo para el ganador, respaldo a persistencia, fallo explícito de candidatos, reserva temporal y validación sin entrenar.
- Prueba Electron de estrategias aprobada: plan, bloqueo de listas incompletas, búsqueda real, semillas, TE, JSON y resultados conservados al navegar.
- Prueba Electron de guía previa aprobada: editar conserva el nuevo valor e invalida la revisión. Se corrigió una interferencia entre eventos nativos de invalidación y entradas controladas de React.
- Build TypeScript/Vite aprobado.
- Prueba real acotada en EURUSD=X: **54 ajustes**, 6 candidatos de historial/objetivo, 4 refinamientos, 3 semillas, 6 folds de 20 pasos y TE de 20. Se conservaron R² negativos de TE y todos los errores, sin retocar selección con TE. Informe de resultados y JSON entregados por separado; el repositorio contiene un resumen.
