# Validación de Quantlabs Neurolab 2.0.0

Fecha: 30 de octubre de 2026. Entorno: macOS, Python 3.12.14, TensorFlow 2.21, ejecución CPU. Las series de las pruebas y capturas de interfaz son sintéticas; no constituyen una medición de rentabilidad ni una validación predictiva sobre EURUSD.

- Suite completa: **118 pruebas aprobadas**, incluidos los cinco grupos modulares iniciales y la compatibilidad de exportaciones, escalado, estrategias e infografías.
- Suite modular final: **7 pruebas aprobadas** tras añadir dos casos de consolidación y fallo de reajuste final; total de casos vigentes: **120**. La suite completa no se repitió después de estos dos casos; se volvió a ejecutar el módulo afectado.
- Ejecución real de **9 módulos × 3 variantes**: salida finita, horizonte correcto, metadatos serializables. Redes con una época para comprobar funcionamiento, no calidad de ajuste. Prophet, arch y RL se ejecutaron con sus bibliotecas reales.
- Verificación de arquitecturas: Bidirectional en RNN/LSTM V3, MultiHeadAttention en LSTM V3, pooling en CNN V2 y conexiones residuales en CNN V3.
- Pruebas de causalidad: componentes invariantes al truncar el prefijo; modificar etiquetas TE no cambia ganador ni clasificación de validación; solo el ganador recibe el reajuste anterior a TE.
- Pruebas matemáticas: R² perfecto, negativo y no definido; objetivo Double DQN y máscara terminal.
- Consolidación real: regresión y ANOVA con dos horizontes, seis candidatos por horizonte. Planes sin historial y estructuras de parámetros inválidas se rechazan.
- Si el reajuste del ganador falla, conserva VA y reporta TE no disponible; no cambia de ganador tras inspeccionar test.
- `npm run build`: compilación Electron/TypeScript/React aprobada.
- `tests/smoke-study.cjs`: nueve opciones, teoría con cinco preguntas, importación CSV, revisión temporal, invalidación al editar, tres regresiones reales, tres LSTM reales con atención, exportación JSON y conservación del resultado al navegar.
- `tests/smoke-strategies.cjs`: búsqueda de estrategias anterior con red real, semillas, test reservado, exportación y conservación al navegar, a través del nuevo gestor de procesos.
- `check_environment.py`: importaciones avanzadas aprobadas. El aviso opcional de Plotly de Prophet no afecta a los gráficos React ni al entrenamiento.
- `bash -n start_mac.sh` y `git diff --check`: aprobados. Windows se revisó como script, **no se ejecutó en Windows**. No se realizó una instalación limpia completa descargando de nuevo todas las dependencias.

Las épocas bajas de los tests solo verifican el circuito. Cada experimento real debe usar un presupuesto adecuado y evaluar su error externo frente a persistencia; la variante más compleja puede perder.
