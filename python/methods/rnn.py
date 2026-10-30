from domain.contracts import MethodSpec,VariantSpec
from .neural_common import forecast_neural
SPECS=(VariantSpec(1,'Vanilla RNN','SimpleRNN univariada con ventana fija.',{'units':[32],'lookback':20}),VariantSpec(2,'Stacked RNN + Dropout','Varias capas recurrentes con regularización.',{'units':[32,16],'lookback':20,'dropout':.1}),VariantSpec(3,'BiRNN sobre residuos','Predice residuos causales y reconstruye tendencia/estacionalidad.',{'units':[32],'lookback':20,'dropout':.1}))
def forecast(ctx,variant):return forecast_neural(ctx,'rnn',variant)
STUDY=MethodSpec('rnn','RNN · simple / apilada / bidireccional','rnn',SPECS,forecast,('Crear ventanas que terminan antes de cada objetivo.','Escalar solo TR interna.','SimpleRNN o pila de estados según variante.','En V3 modelar residuos causales y reconstruir la serie.','Restaurar checkpoint interno y evaluar recursivamente.'),'tensorflow')
