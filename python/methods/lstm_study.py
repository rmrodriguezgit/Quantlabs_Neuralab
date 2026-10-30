"""Ejemplo de módulo SOLID: tres variantes, contrato compartido y ajuste local."""
from domain.contracts import MethodSpec,VariantSpec
from .neural_common import forecast_neural
SPECS=(VariantSpec(1,'LSTM univariada','Una capa sobre la serie observada.',{'units':[32],'lookback':20}),VariantSpec(2,'Stacked LSTM multivariada','Entradas: tendencia, estacionalidad y residuo calculados causalmente.',{'units':[32,16],'lookback':20,'dropout':.1}),VariantSpec(3,'BiLSTM + Atención','Atención sobre estados bidireccionales de una ventana histórica.',{'units':[32],'lookback':20,'dropout':.1}))
def forecast(ctx,variant):return forecast_neural(ctx,'lstm',variant)
STUDY=MethodSpec('lstm','LSTM · simple / componentes / atención','lstm',SPECS,forecast,('Calcular componentes solo con el prefijo conocido.','Crear entradas multivariadas desplazadas antes del objetivo.','Comparar memoria simple y apilada.','Aplicar atención únicamente a la ventana histórica.','Elegir pesos con VA interna y comparar variantes en VA externa.'),'tensorflow')
