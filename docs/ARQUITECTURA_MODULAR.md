# Quantlabs Neurolab 2.0 · Estudios modulares

Abre **Estudio individual · V1–V3** en la barra izquierda o ejecuta `npm run study`. El análisis general, editor neuronal, GAN, infografías y banco de variantes siguen disponibles.

## Estructura

```text
electron/
  main.ts                    # Ventana, IPC y archivos autorizados
  engineProcess.ts           # PythonEngine: procesos, límites, progreso, cierre
python/
  engine_runner.py           # Entrada JSON / salida JSON; diagnóstico por stderr
  engine.py                  # Compatibilidad de acciones e ingesta existente
  study_runner.py            # Caso de uso: descripción, folds, variantes, selección
  domain/
    contracts.py             # Contexto de un ajuste y descriptores de módulos
    decomposition.py         # STL descriptiva y componentes causales separados
    metrics.py               # MSE, RMSE, MAE, R² sin redondear
  methods/
    registry.py              # Registro declarativo; sin switch de algoritmos
    common.py                # Fechas, validación de resultados, parámetros
    regression.py            # Lineal / grado interno / Ridge-Lasso estacional
    anova.py                 # Un factor / interacción / ANCOVA + MANOVA
    garch.py                 # GARCH / EGARCH / GJR-t
    monte_carlo.py           # GBM / deriva estacional / saltos Poisson
    neural_common.py         # Construcción y entrenamiento local de redes
    rnn.py                   # Simple / stacked-dropout / bidireccional-residuos
    cnn.py                   # Local / pooling-kernel / dilatada-residual
    lstm_study.py            # Univariada / componentes / BiLSTM-atención
    prophet.py               # Automático / cambios-regresores / priors internos
    rl.py                    # Tabular / DQN / DDQN-reducción de error
  check_environment.py       # Versión, venv e importaciones reales
src/components/
  IndividualStudy.tsx        # Práctica individual, configuración y reporte
  DataPanel.tsx              # Ingesta reutilizada
  Education.tsx              # Teoría y cuestionario del módulo
start_mac.sh
start_windows.bat
```

La separación corresponde al nuevo flujo de estudio. Los laboratorios anteriores conservan sus interfaces; su migración completa no es requisito para ejecutar los módulos nuevos. `study_runner` utiliza la ingesta y las reglas de horizonte existentes como adaptadores de compatibilidad.

## Uso

1. Carga un escenario empresarial, CSV/Excel o un activo de Yahoo. La ingesta financiera admite 1Y, 2Y, 5Y y 10Y a resolución diaria. Declara correctamente tipo y frecuencia en Análisis para archivos importados.
2. Selecciona un módulo. Lee las tres variantes y su marco teórico con cinco preguntas. Usa «Consolidar varios módulos» para ampliar el ranking; aumenta el coste.
3. Selecciona horizontes, folds y periodo estacional. El ajuste libre muestra los parámetros JSON recomendados de cada variante. Editarlos invalida la revisión del plan.
4. Revisa TR/VA/TE. Solo después se habilita ejecutar. Las variantes se ajustan secuencialmente, con contexto, escaladores y pesos locales a cada ajuste.
5. Compara métricas, gráficos de VA, errores, ganador por horizonte, test reservado y futuro. Descarga el JSON para conservar configuración y resultados. El resultado permanece en la sesión al cambiar de pestaña; exporta antes de cerrar.

1500 épocas es un techo inicial configurable hasta 5000, con Early Stopping opcional, paciencia, mejora mínima y calentamiento. Siempre se restaura el checkpoint de menor MSE interno, incluso sin parada temprana. Los métodos estadísticos no usan épocas. RL usa episodios y reporta su recompensa, sin afirmar rentabilidad.

## Protocolo de evaluación

- **Nombres:** tendencia, estacionalidad y residuo son componentes; TR, VA y TE en la interfaz significan entrenamiento, validación y test. MAE es error absoluto medio; «MSA» no se utiliza como sinónimo estadístico.
- La STL de la serie completa se muestra solo como descripción. Cada ajuste calcula sus propios componentes causales sin acceso a VA externa o TE. LSTM multivariada usa tendencia, estacionalidad y residuo causales.
- Se reservan `h` observaciones finales para TE y `folds × h` anteriores para VA externa. Los TR se expanden cronológicamente. Se rechaza el plan sin historial suficiente.
- Los hiperparámetros internos, escalas y Early Stopping se deciden dentro de cada TR. VA externa clasifica las variantes por MSE; MAE e identificador resuelven empates. Sobre los mismos objetivos, maximizar R² y minimizar MSE producen la misma ordenación cuando la varianza no es cero.
- Persistencia puede ganar si está habilitada y su MSE de VA es menor o igual. No se fuerza una mejora V1 → V2 → V3 ni un R² próximo a uno.
- Solo el ganador por horizonte se ajusta con todo el tramo anterior a TE y se evalúa una vez en TE. Después puede ajustarse con todo el historial observado para proyectar el futuro. Las curvas de las tres variantes pertenecen a VA, no a tres evaluaciones del test.
- Cada horizonte es una tarea distinta; los bloques se pueden solapar. Repetir estudios tras consultar TE deja de ser una confirmación independiente. Los días financieros naturales se convierten a barras según la densidad histórica de fechas; la proyección usa la frecuencia declarada, sin calendario bursátil específico.
- MSE, RMSE y MAE están en las unidades originales. R² es nulo si la muestra es constante o tiene menos de dos puntos; puede ser negativo. No se compara directamente el MSE de variables distintas.
- Las redes hacen previsión recursiva externa. La validación interna es a un paso con historia observada; el resultado externo mide también el error acumulado de la recursión.

## Alcance de cada variante especial

ANOVA V3 pronostica una respuesta mediante ANCOVA, con covariables rezagadas y escenario futuro constante. Si existen dos respuestas, agrega diagnóstico MANOVA; no inventa una MANOVA para un archivo univariado. La autocorrelación limita la interpretación de los p-valores clásicos.

GARCH evalúa escenarios de precio para poder compartir el ranking, pero su objetivo principal es la varianza. Si gana, muestra QLIKE y MSE de varianza con retorno diario al cuadrado como proxy. EGARCH dispone de hasta 2000 iteraciones y rechaza ajustes sin convergencia. No equivale a varianza realizada intradía.

Prophet V2 necesita otras columnas válidas para usar regresores. Se rezagan en entrenamiento; el futuro mantiene el último valor observado como escenario explícito. V3 elige priors con validación interna. CNN V2 elige kernel por MSE interno. CNN V3 contiene convoluciones causales dilatadas con conexiones residuales. Las redes bidireccionales y la atención solo recorren la ventana histórica disponible.

RL V3 usa `delta² − (delta − acción)²`, mejora respecto a no cambio, en unidades normalizadas de TR. DQN y DDQN tienen replay de TR, target, máscara terminal y tres acciones; DDQN separa elección online y evaluación target. Este entorno estudia error de pronóstico, no ejecución bursátil, costes o beneficios.

El estudio exporta JSON con predicciones, métricas, parámetros, diagnósticos y fechas; no serializa pesos de sus 27 variantes. Para los paquetes de inferencia que ya soporta la aplicación, utiliza **Banco de variantes**. No se transforma este informe automáticamente en un bot.

## Arranque y contrato

Los scripts requieren Node 22+ y Python 3.10–3.12; crean `.venv`, instalan los requisitos avanzados por defecto y verifican importaciones antes de compilar/abrir. `FORECAST_ADVANCED=0` permite instalación básica: las variantes que requieran paquetes ausentes se identificarán y reportarán su error. Windows usa `py` o `FORECAST_PYTHON`; macOS busca un intérprete compatible. El script Windows se entrega, pero las pruebas de esta versión se ejecutaron en macOS.

```bash
./start_mac.sh
# Alternativa básica
FORECAST_ADVANCED=0 ./start_mac.sh
# Protocolo JSON sin interfaz
printf '%s' '{"action":"study_catalog"}' | .venv/bin/python python/engine_runner.py
```

`study_preview` y `study_run` reciben `dataset`, `column`, `methods`, `horizons` (índices 0–2 financieros o 0–4 empresariales) y `config`. `config.variantParameters` usa claves de método y variante: `{"lstm":{"3":{"units":[32],"lookback":20}}}`. El ejemplo ejecutable de las tres LSTM está en `python/methods/lstm_study.py`, que delega la arquitectura en `neural_common.py` y devuelve pronósticos y metadatos; el caso de uso calcula las métricas con las etiquetas retenidas.

Cada método registra un `MethodSpec` y exactamente tres `VariantSpec`. `forecast(context, variant)` devuelve `prediction` finita de longitud `context.horizon` y `metadata` JSON. No recibe etiquetas futuras. Las excepciones de candidatos se conservan como fallos del estudio. Para añadir otro módulo, implementa el contrato e inscríbelo en `registry.MODULES`.

## Verificación de la entrega

Consulta `VALIDACION_2.0.md` para los resultados y sus límites. Las capturas `estudio-individual-ui.png` y `estudio-configuracion-ui.png` muestran la interfaz con datos sintéticos.
