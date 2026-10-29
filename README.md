# Quantlabs Neuralab · Forecast Lab 1.3

Aplicación de escritorio para explorar series empresariales y financieras, generar escenarios, comparar pronósticos y estudiar los métodos. Interfaz en español, Electron + React/TypeScript y motor Python mediante JSON por entrada/salida estándar.

**Nuevo 1.3:** [Banco de variantes, Early Stopping y exportación para inferencia](docs/VARIANTES_Y_EXPORTACION.md). Hasta 5,000 épocas, techo 1,500 con parada opcional, comparación por familia y descargas JSON/ZIP con arquitectura, pesos, escalas y script paper.

**Taller 1.2:** [análisis de las nueve infografías y taller U2–U10](docs/ANALISIS_INFOGRAFIAS.md). Incluye MLP, comparación de optimizadores, regularización, HPO temporal, CNN, LSTM/GRU bidireccionales, PCA/AE/DAE, CVAE/cGAN de ventanas y Double DQN con costes.

**Editor general 1.1:** [laboratorio neuronal y GAN condicional](docs/LABORATORIO_NEURONAL.md), con arquitecturas independientes para RNN, CNN, LSTM, DQN y GAN; entrada configurable, 1–5 capas intermedias, activaciones, optimizadores, indicadores y curvas de pérdidas.

## Repositorio

[rmrodriguezgit/Quantlabs_Neuralab](https://github.com/rmrodriguezgit/Quantlabs_Neuralab)

```bash
git clone https://github.com/rmrodriguezgit/Quantlabs_Neuralab.git
cd Quantlabs_Neuralab
# macOS: FORECAST_ADVANCED=1 ./start_mac.sh
# Windows: set FORECAST_ADVANCED=1 y ejecutar start_windows.bat
```

Ejemplos descargables: [selección de variantes JSON](examples/seleccion-variantes-demo.json) y [modelos de inferencia ZIP](examples/modelos-demo-inferencia.zip), entrenados con la demo sintética. La integración se describe en [VARIANTES_Y_EXPORTACION.md](docs/VARIANTES_Y_EXPORTACION.md).

## 1. Estructura del proyecto

```text
forecast-lab/
├── start_mac.sh                 # Verifica herramientas, instala y arranca
├── start_windows.bat
├── package.json
├── package-lock.json            # Resolución de npm verificada
├── tsconfig.json
├── tsconfig.electron.json
├── vite.config.ts
├── index.html                   # CSP y entrada del renderer
├── electron/
│   ├── main.ts                  # Ventanas, IPC, procesos y diálogos nativos
│   └── preload.ts               # Puente de tres operaciones concretas
├── python/
│   ├── engine.py                # Validación, datos, métricas y ranking temporal
│   ├── models.py                # Estadística, simulación y despacho de modelos
│   ├── neural.py                # Redes configurables, DQN y cGAN
│   ├── features.py              # Indicadores incrementales causales
│   ├── financial_lab.py         # OHLC, etiquetas, escalas y ejecución
│   ├── unit_lab.py              # Despacho U2–U10
│   ├── unit_supervised.py       # MLP, regularización, HPO, CNN, recurrentes
│   ├── unit_autoencoders.py     # PCA, AE, DAE y Ridge
│   ├── unit_generative.py       # CVAE/cGAN de ventanas
│   ├── unit_trading.py          # Double DQN y referencias
│   ├── requirements.txt         # Dependencias estadísticas y financieras
│   ├── requirements-advanced.txt
│   └── requirements-verified.txt
├── src/
│   ├── main.tsx
│   ├── App.tsx                  # Sesión, datasets y controles de evaluación
│   ├── networks.ts              # Configuración y persistencia del editor
│   ├── unitConfig.ts            # Configuraciones y contratos U2–U10
│   ├── infographics.ts          # Análisis de las nueve imágenes
│   ├── types.ts
│   ├── style.css
│   ├── lessons.json             # 24 lecciones, cinco preguntas cada una
│   └── components/
│       ├── DataPanel.tsx
│       ├── Dashboard.tsx
│       ├── Report.tsx
│       ├── Education.tsx
│       ├── NetworkStudio.tsx
│       ├── NetworkResult.tsx
│       ├── InfographicWorkshop.tsx
│       └── UnitResults.tsx
├── tests/
│   ├── test_engine.py
│   ├── test_advanced.py
│   ├── test_neural_lab.py
│   ├── smoke.cjs
│   ├── smoke-neural.cjs
│   ├── test_units.py
│   └── smoke-units.cjs
├── examples/
│   └── empresa_mensual.csv
├── public/infographics/         # Nueve originales y manifest SHA-256
└── docs/
    ├── ARQUITECTURA.md
    ├── ANALISIS_INFOGRAFIAS.md
    ├── LABORATORIO_NEURONAL.md
    └── VALIDACION.md
```

`node_modules`, `.venv`, `dist`, `dist-electron` y `release` se generan localmente. El ZIP de entrega contiene las fuentes y el lockfile; excluye los entornos y dependencias instaladas.

## 2. Arrancadores

Requisitos: **Node.js 22+ con npm y Python 3.10–3.12**; se recomienda Python 3.12 para disponer de ruedas compatibles con TensorFlow. Primera instalación con acceso a Internet. No se modifica el Python global: los paquetes se instalan en `.venv`.

macOS, desde la carpeta del proyecto:

```bash
chmod +x start_mac.sh
./start_mac.sh
```

Windows: ejecutar `start_windows.bat` desde una terminal o con doble clic. Usa Python Launcher (`py`); si Python está en otra ruta, define `FORECAST_PYTHON` con la ruta del ejecutable.

Para instalar también Prophet, TensorFlow y Gymnasium:

```bash
# macOS
FORECAST_ADVANCED=1 ./start_mac.sh
```

```bat
rem Windows
set FORECAST_ADVANCED=1
start_windows.bat
```

Los arrancadores validan Node, crean el entorno, instalan requisitos, ejecutan `npm ci` cuando existe lockfile, compilan y abren Electron. Se detienen si falla un paso. Las redes no están seleccionadas inicialmente para mantener ágil el primer análisis. Al instalar dependencias nuevas, reinicia la aplicación para actualizar su disponibilidad.

## 3. Motor Python

`engine.py` acepta una petición por proceso y produce una respuesta `{"ok":true,"data":...}` o `{"ok":false,"error":...}`. Los registros y las salidas de las bibliotecas van a stderr. El proceso principal de Electron aplica límites de tamaño y tiempo y comunica errores a la interfaz.

```bash
printf '%s' '{"action":"simulate","records":180,"frequency":"monthly","columns":["Ventas","Costos"],"seed":42}' | .venv/bin/python python/engine.py
```

En Windows, `.venv\Scripts\python.exe python\engine.py` admite el mismo objeto JSON por stdin.

Operaciones:

- `simulate`: columnas dinámicas, semilla, registros, fecha inicial, frecuencia y parámetros `specs` por variable (`base`, `trend`, `seasonality`, `noise`).
- `finance`: `ticker`, `period` y `interval`; descarga bajo demanda con yfinance.
- `import` / `export`: CSV o XLSX, disponibles mediante los diálogos nativos. Al importar varios archivos, informa el resultado individual de cada uno.
- `analyze`: dataset, columna, métodos, periodo estacional, ventanas, pasos de validación, pasos futuros y parámetros de redes/RL.
- `unit_lab`: ejecución por unidad 2–10, datasets OHLC y configuración temporal/metodológica independiente.
- `demo_ohlc`: ejemplo sintético diario 2020–2025 para practicar las unidades.
- `capabilities`: disponibilidad de las bibliotecas opcionales.

Métodos implementados: persistencia, regresión lineal y cuadrática, ARIMA(1,1,1), SARIMA(1,1,1)(1,0,0,s), GARCH(1,1) de retornos, Monte Carlo de retornos logarítmicos, SimpleRNN, CNN 1D, LSTM, Prophet con ciclo por observaciones, Q-Learning tabular y DQN con memoria y red objetivo; GAN condicional con generador y discriminador Dense configurables. ANOVA compara grupos de fase estacional de manera descriptiva; no es un método de pronóstico y no entra al ranking.

### Evaluación y horizontes

1. Se reserva al final de la serie un bloque de test (20%, hasta 60 observaciones; mínimo cuatro).
2. El ranking principal se calcula con ventanas crecientes anteriores a ese bloque. El tamaño de ventana se configura en la interfaz.
3. Se elige por RMSE ascendente, después MAE ascendente y R² descendente en empates. Las métricas agregadas concatenan errores de las ventanas; se incluyen también métricas de cada ventana en JSON.
4. Solo el ganador se evalúa en el test reservado. Se vuelve a entrenar con todo el historial para el pronóstico futuro.
5. Se crean rankings independientes para 15/30/60 días naturales en datos financieros y 1–5 años en datos empresariales. Si no hay entrenamiento suficiente o falla un modelo, aparece la causa; nunca se inventa un ganador.

Los horizontes financieros convierten días naturales a observaciones usando la densidad histórica de fechas. Es una aproximación, especialmente en intradía. Las fechas futuras usan frecuencia de días hábiles o intradía genérica, **sin calendario de festivos ni horario de bolsa**. El usuario configura los pasos futuros por separado del ranking por horizonte. Las métricas futuras no existen hasta observar los valores reales.

## 4. Componentes React y flujo de uso

1. **DataPanel**: genera datos empresariales, descarga un activo o importa varios archivos CSV/XLSX.
2. Selecciona dataset, tipo, frecuencia y variable. La frecuencia inferida al importar se puede corregir; el motor trabaja por observaciones y no rellena huecos.
3. Selecciona métodos, periodo estacional y validación; ejecuta el análisis.
4. **Dashboard**: vistas separadas de historia/TR/VA/TE, test y futuro, errores e histograma. MSE, RMSE, MAE y R² corresponden al test final del ganador.
5. **Report**: ranking de validación, ganadores por horizonte, errores por método y comparación entre datasets/variables de la sesión; exportación JSON. Los errores absolutos no son comparables entre variables con distinta escala.
6. **Education**: fórmulas renderizadas con KaTeX, supuestos, interpretación, limitaciones y 120 preguntas con corrección inmediata. La mejor puntuación por lección se guarda en almacenamiento local.

Los datasets y resultados permanecen en memoria durante la sesión. Exporta CSV/Excel y reportes antes de cerrar; las puntuaciones educativas y configuraciones de redes persisten automáticamente. Se incluye `examples/empresa_mensual.csv` para probar la importación.

### Datos y restricciones explícitas

- CSV con encabezado `date` o `fecha`, fechas ISO 8601 y variables numéricas; XLSX con el mismo esquema en la primera hoja.
- Mínimo 24 registros, hasta 20,000 filas, 30 columnas y 20 MB por archivo. Máximo 20 archivos por importación.
- Se rechazan fechas duplicadas, datos faltantes o no numéricos. No se interpolan datos usando información futura.
- Yahoo Finance: diario con 1Y/2Y/5Y/10Y; intradía de 1 hora o 15 minutos con un mes. Es descarga bajo demanda y puede tener retrasos o restricciones del proveedor, no streaming en tiempo real.
- STL es descriptivo y se calcula sobre el historial completo, separado del entrenamiento. Sin dos ciclos, se usa tendencia móvil y VA=0 con indicación visible.
- GARCH exige valores positivos y modela volatilidad. El precio se obtiene por simulación; el ranking de precio no valida la estimación de volatilidad.
- Monte Carlo supone choques logarítmicos normales. Sus percentiles son escenarios, no intervalos calibrados.
- Q-Learning y DQN son políticas experimentales de dirección con recompensa por error cuadrático normalizado. No ejecutan operaciones ni calculan beneficios de una estrategia.
- Prophet trabaja en pasos de observación con un ciclo configurable; no incorpora regresores ni festivos en esta versión.
- Órdenes ARIMA/SARIMA y grado polinómico son fijos. Las arquitecturas de RNN/CNN/LSTM/DQN/GAN, activaciones, optimizadores y características se editan en el Laboratorio neuronal. Épocas, episodios, α, γ y ε son configurables. La búsqueda aleatoria de hiperparámetros MLP está disponible en U5 con folds causales internos; no se aplica automáticamente al ranking general de pronóstico.
- Se usan validaciones retrospectivas; repetir elecciones mirando el test final lo convierte en información de ajuste. Para un estudio final, conserva datos externos sin consultar.

## Compilar, probar y empaquetar

```bash
npm run check
npm run build
# macOS; Windows: .venv\Scripts\python.exe
.venv/bin/python -m unittest discover -s tests -v
npm run smoke
npm run smoke:neural
npm run smoke:units
npm run smoke:emulator
npm run package
npm run dist
```

`package` crea una aplicación sin instalador; `dist` configura DMG en macOS y NSIS en Windows. Compila en cada sistema de destino. El instalador requiere Python y sus dependencias instaladas: en el primer inicio pide elegir el ejecutable de la `.venv` local; guarda su ruta en `runtime.json` del directorio de datos de Electron. También acepta `FORECAST_PYTHON`. No se copia una venv entre equipos: sus rutas no son portables. Si cambias la ubicación del entorno, elimina ese archivo de configuración o usa la variable de entorno.

**Esta entrega es código fuente ejecutable, no un instalador firmado ni una distribución con Python embebido.** La distribución comercial necesitaría un runtime autónomo por plataforma, firma de código y validación en Windows. Consulta `docs/VALIDACION.md` para distinguir pruebas realizadas y límites pendientes.

## Fuentes técnicas

El puente IPC y aislamiento siguen la documentación oficial de [Electron IPC](https://www.electronjs.org/docs/latest/tutorial/ipc) y [aislamiento de contexto](https://www.electronjs.org/docs/latest/tutorial/context-isolation). La integración financiera utiliza la [API de yfinance](https://ranaroussi.github.io/yfinance/reference/index.html) y respeta la disponibilidad del proveedor.

## Taller de infografías U2–U10

Desde el Laboratorio neuronal abre **Taller de infografías U2–U10**, elige la unidad y crea la demo OHLC o selecciona tus activos diarios. Cada unidad muestra sus diez pasos, imagen original, preset editable y controles específicos. Puedes descargar SPY a diario, ajustar train/validación/test, ejecutar y exportar el experimento con configuración, cortes, purgas, historiales y métricas.

Los valores por defecto estudian 2020–2023 en train, 2024 en validación y 2025 en test. Los indicadores se ajustan con train. U5 reajusta escalas y redes por fold interno; U9 descarta ventanas que cruzan cortes y usa condición causal; U10 conserva posición anterior en el gap y compara referencias sin aprender en test. Las señales supervisadas, en cambio, se cierran al final del mismo intradía.

Consulta [ANALISIS_INFOGRAFIAS.md](docs/ANALISIS_INFOGRAFIAS.md) para las equivalencias con las nueve imágenes, definiciones y límites. Los fragmentos originales son PyTorch; la adaptación ejecutable mantiene el backend TensorFlow/Keras existente, en float32. No se afirma replicar notebooks no aportados o validación Colab/T4.

Para abrir directamente el taller con el entorno Python preparado: `npm run units`. El arrancador habitual continúa abriendo el espacio general.

Para abrir directamente el banco: `npm run emulator`. Ver [VARIANTES_Y_EXPORTACION.md](docs/VARIANTES_Y_EXPORTACION.md) para criterios por tarea, controles de parada y formato de los paquetes.
