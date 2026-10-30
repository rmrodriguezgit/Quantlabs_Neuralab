import numpy as np
import pandas as pd
from domain.contracts import MethodSpec,VariantSpec
from domain.metrics import regression
from .common import output,params,inner_cut,dates,real
SPECS=(VariantSpec(1,'Prophet automático','Estacionalidad automática con fechas reales.',{'changepointPrior':.05,'seasonalityPrior':10}),VariantSpec(2,'Changepoints y regresores','Puntos de cambio dentro de TR y covariables rezagadas.',{'changepointFractions':[.25,.5,.75],'changepointPrior':.05,'maxRegressors':2}),VariantSpec(3,'Prior Scale interno','Selecciona priors con un bloque interno de TR.',{'trendPriors':[.01,.05,.2],'seasonPriors':[1,10]}))
def fitted(ctx,p,cp=.05,sp=10,regressors=False):
    from prophet import Prophet
    from features import bounded_int
    n=len(ctx.values);frame=pd.DataFrame({'ds':pd.DatetimeIndex(ctx.index).tz_localize(None),'y':ctx.values});future=pd.DataFrame({'ds':dates(ctx).tz_localize(None)});kwargs={'changepoint_prior_scale':cp,'seasonality_prior_scale':sp,'daily_seasonality':False}
    if regressors:
        fractions=p.get('changepointFractions',[.25,.5,.75])
        if not isinstance(fractions,list) or not 1<=len(fractions)<=8:raise ValueError('Puntos de cambio: una a ocho fracciones.')
        points=sorted(set(max(1,min(n-2,int(real(f,.05,.95,'Fracción changepoint')*n))) for f in fractions));kwargs['changepoints']=frame.ds.iloc[points].tolist()
    model=Prophet(**kwargs);columns=[]
    if regressors:
        for col in [c for c in ctx.frame.columns if c!=ctx.column][:bounded_int(p.get('maxRegressors',2),0,5,'Regresores')]:
            key='cov_'+str(len(columns));values=ctx.frame[col].astype(float)
            if not np.isfinite(values).all() or values.std()<1e-12:continue
            frame[key]=values.shift(1).to_numpy();future[key]=float(values.iloc[-1]);model.add_regressor(key);columns.append(col)
    frame=frame.dropna();model.fit(frame);result=model.predict(future)
    return result.yhat.to_numpy(),columns

def forecast(ctx,variant):
    p=params(ctx,variant,SPECS);cp=real(p.get('changepointPrior',.05),.0001,1,'Changepoint prior');sp=real(p.get('seasonalityPrior',10),.01,100,'Seasonality prior');trials=[]
    if variant==3:
        cut=inner_cut(len(ctx.values));priors=p.get('trendPriors',[.01,.05,.2]);seasons=p.get('seasonPriors',[1,10])
        if not isinstance(priors,list) or not isinstance(seasons,list) or not priors or not seasons or len(priors)*len(seasons)>12:raise ValueError('La búsqueda de priors admite de uno a doce pares.')
        for trend in priors:
            for season in seasons:
                a=real(trend,.0001,1,'Trend prior');b=real(season,.01,100,'Season prior');pred,_=fitted(ctx.prefix(cut,len(ctx.values)-cut),p,a,b);trials.append({'trend':a,'season':b,'mse':regression(ctx.values[cut:],pred)['mse']})
        best=min(trials,key=lambda t:t['mse']);cp=best['trend'];sp=best['season']
    pred,columns=fitted(ctx,p,cp,sp,variant==2)
    return output(pred,ctx,changepointPrior=cp,seasonalityPrior=sp,regressors=columns,internalTrials=trials,note='Usa fechas reales y estacionalidad automática de Prophet. Regresores futuros mantienen el último valor de TR como escenario explícito; no se leen columnas futuras de VA/TE.')
STUDY=MethodSpec('prophet','Prophet · automático / changepoints / priors','prophet',SPECS,forecast,('Usar fechas y frecuencia reales.','Estimar tendencia y estacionalidad en TR.','Definir puntos de cambio solo dentro de TR.','Fijar escenarios de regresores futuros desconocidos.','Elegir priors con VA interna y evaluar externamente.'),'prophet')
