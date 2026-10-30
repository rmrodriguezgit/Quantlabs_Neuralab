"""Descomposición descriptiva separada de las características causales."""
import numpy as np
import pandas as pd
def causal(values,period):
    y=np.asarray(values,dtype=float);n=len(y);trend=pd.Series(y).ewm(span=max(2,period),adjust=False).mean().to_numpy();raw=y-trend;season=np.zeros(n);totals=np.zeros(max(1,period));counts=np.zeros(max(1,period))
    if period>1:
        for i,v in enumerate(raw):
            phase=i%period;totals[phase]+=v;counts[phase]+=1;season[i]=totals[phase]/counts[phase]
    return np.c_[trend,season,y-trend-season]
def descriptive(values,period):
    y=np.asarray(values,dtype=float)
    if period>=2 and len(y)>=2*period:
        from statsmodels.tsa.seasonal import STL
        m=STL(y,period=period,robust=True).fit();return {'trend':m.trend.tolist(),'seasonal':m.seasonal.tolist(),'residual':m.resid.tolist(),'note':'STL sobre la serie completa para descripción. Nunca entra como característica del ajuste ni en la selección.'}
    c=causal(y,period);return {'trend':c[:,0].tolist(),'seasonal':c[:,1].tolist(),'residual':c[:,2].tolist(),'note':'Sin ciclos completos para STL: componentes causales ilustrativos; la descomposición de cada ajuste se recalcula con su propio pasado.'}
def seasonal_projection(y,horizon,period):
    c=causal(y,period);raw=np.asarray(y)-c[:,0];pattern=np.zeros(max(1,period))
    if period>1:
        for phase in range(period):
            values=raw[np.arange(len(y))%period==phase];pattern[phase]=values[-5:].mean() if len(values) else 0
        pattern-=pattern.mean()
    return pattern[(len(y)+np.arange(horizon))%max(1,period)]
