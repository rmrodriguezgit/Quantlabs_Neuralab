import numpy as np
from domain.contracts import MethodSpec,VariantSpec
from domain.decomposition import causal,seasonal_projection
from domain.metrics import regression
from .common import output,params,inner_cut,bounded_int,real
SPECS=(VariantSpec(1,'Lineal sobre tendencia','Proyecta la tendencia causal y añade estacionalidad estimada en entrenamiento.',{'degree':1}),VariantSpec(2,'Polinómica con grado interno','Selecciona el grado usando solo un bloque interno de entrenamiento.',{'maxDegree':4}),VariantSpec(3,'Ridge/Lasso + variación','Regularización interna sobre tiempo y componentes estacionales conocidos.',{'regularizer':'auto','alpha':1.0}))
def fit_curve(ctx,y,h,degree=1,regularizer=None,alpha=1):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler,PolynomialFeatures
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    n=len(y);period=ctx.period;t=np.arange(n,dtype=float);future=np.arange(n,n+h,dtype=float);trend=causal(y,period)[:,0]
    if regularizer:
        def features(x):return np.c_[x,x*x,np.sin(2*np.pi*x/max(period,1)),np.cos(2*np.pi*x/max(period,1))]
        X=features(t);Z=features(future);target=y;model=make_pipeline(StandardScaler(),Ridge(alpha=alpha) if regularizer=='ridge' else Lasso(alpha=alpha,max_iter=5000))
    else:
        X=t[:,None];Z=future[:,None];target=trend;model=make_pipeline(StandardScaler(),PolynomialFeatures(degree),LinearRegression())
    model.fit(X,target);p=model.predict(Z)
    if not regularizer:p=p+seasonal_projection(y,h,period)
    return p

def forecast(ctx,variant):
    p=params(ctx,variant,SPECS);y=ctx.values;cut=inner_cut(len(y));trials=[];chosen={'degree':1}
    if variant==2:
        for degree in range(1,bounded_int(p.get('maxDegree',4),2,6,'Grado máximo')+1):
            pred=fit_curve(ctx,y[:cut],len(y)-cut,degree);trials.append({'degree':degree,'mse':regression(y[cut:],pred)['mse']})
        chosen={'degree':min(trials,key=lambda t:t['mse'])['degree']}
    if variant==3:
        kind=p.get('regularizer','auto')
        if kind not in ('auto','ridge','lasso'):raise ValueError('Regularización: auto, ridge o lasso.')
        center=real(p.get('alpha',1),.000001,1000,'Alpha')
        for regularizer in (['ridge','lasso'] if kind=='auto' else [kind]):
            for alpha in [center/10,center,center*10]:trials.append({'regularizer':regularizer,'alpha':alpha,'mse':regression(y[cut:],fit_curve(ctx,y[:cut],len(y)-cut,regularizer=regularizer,alpha=alpha))['mse']})
        best=min(trials,key=lambda t:t['mse']);chosen={k:best[k] for k in ('regularizer','alpha')}
    return output(fit_curve(ctx,y,ctx.horizon,**chosen),ctx,chosen=chosen,internalTrials=trials,fitLastDate=ctx.index[-1].isoformat(),note='Grado/regularización seleccionados dentro de TR. La descomposición descriptiva completa no se usa para ajustar.')
STUDY=MethodSpec('regression','Regresión lineal / polinómica / regularizada','linear',SPECS,forecast,('Separar entrenamiento y evaluación antes de ajustar.','Calcular tendencia causal usando solo observaciones conocidas.','Proyectar tiempo y variación estacional.','Elegir grado/regularización con validación interna.','Reconstruir la serie y medir en unidades originales.'))
