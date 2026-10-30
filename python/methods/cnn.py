from domain.contracts import MethodSpec,VariantSpec
from .neural_common import forecast_neural
SPECS=(VariantSpec(1,'CNN 1D local','Convolución causal y resumen global.',{'units':[32],'lookback':20,'kernelSize':3}),VariantSpec(2,'CNN + Pooling adaptable','Selecciona kernel con VA interna; pooling reduce la dimensión temporal.',{'units':[32,16],'lookback':20,'kernels':[3,5],'dropout':.1}),VariantSpec(3,'CNN dilatada / TCN','Dilataciones 1, 2… y conexiones residuales sobre historia.',{'units':[32,32],'lookback':20,'kernelSize':3,'dropout':.1}))
def forecast(ctx,variant):return forecast_neural(ctx,'cnn',variant)
STUDY=MethodSpec('cnn','CNN 1D · local / pooling / dilatada','cnn',SPECS,forecast,('Separar ventanas y objetivos futuros.','Aplicar convoluciones causales.','Comparar pooling y kernel en TR interna.','Ampliar campo receptivo con dilatación y residuos.','Reconstruir escala original y medir fuera de muestra.'),'tensorflow')
