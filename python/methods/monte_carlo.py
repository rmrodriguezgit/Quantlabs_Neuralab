import numpy as np
from domain.contracts import MethodSpec,VariantSpec
from .common import output,params,bounded_int,real
SPECS=(VariantSpec(1,'GBM histórico','Deriva y volatilidad de retornos históricos.',{'paths':1000}),VariantSpec(2,'GBM con deriva estacional','Deriva por fase y volatilidad histórica.',{'paths':1000}),VariantSpec(3,'Difusión con saltos Poisson','Saltos estimados con extremos de TR.',{'paths':1000,'jumpThreshold':2.5}))
def forecast(ctx,variant):
    y=ctx.values;p=params(ctx,variant,SPECS)
    if np.any(y<=0):raise ValueError('GBM requiere precios positivos.')
    returns=np.diff(np.log(y));n=len(returns);count=bounded_int(p.get('paths',1000),100,5000,'Trayectorias');rng=np.random.default_rng(ctx.seed);mu=float(returns.mean());sigma=max(float(returns.std(ddof=1)),1e-8);drift=np.repeat(mu,ctx.horizon);jumps=np.zeros((count,ctx.horizon));intensity=0.
    if variant==2 and ctx.period>1:
        for k in range(ctx.horizon):
            observed=returns[(np.arange(n)+1)%ctx.period==(len(y)+k)%ctx.period];drift[k]=observed.mean() if len(observed)>=3 else mu
    if variant==3:
        extreme=np.abs(returns-mu)>real(p.get('jumpThreshold',2.5),1.5,6,'Umbral saltos')*sigma;tail=returns[extreme];body=returns[~extreme];intensity=float(extreme.mean())
        if len(body)>=10:drift[:]=body.mean();sigma=max(float(body.std(ddof=1)),1e-8)
        if len(tail):
            events=rng.poisson(intensity,size=(count,ctx.horizon));jumps=events*float(tail.mean())+np.sqrt(events)*max(float(tail.std()),1e-8)*rng.normal(size=events.shape)
    shocks=drift+rng.normal(0,sigma,size=(count,ctx.horizon))+jumps;paths=y[-1]*np.exp(np.cumsum(shocks,axis=1))
    return output(np.median(paths,axis=0),ctx,lower=np.quantile(paths,.025,axis=0).tolist(),upper=np.quantile(paths,.975,axis=0).tolist(),samplePaths=paths[:8].tolist(),jumpIntensity=intensity,drift=drift.tolist(),sigma=sigma,note='Simula retornos log por sesión; percentiles sin calibración. Intensidad y tamaño de saltos se estiman solo con TR. No se fuerza una ventaja en MSE por aumentar trayectorias.')
STUDY=MethodSpec('monte_carlo','Monte Carlo / GBM / saltos','monte_carlo',SPECS,forecast,('Estimar retornos y volatilidad únicamente en TR.','Generar innovaciones con semilla reproducible.','Añadir deriva por fase o saltos según variante.','Acumular retornos y reconstruir precios.','Medir punto central e interpretar dispersión sin confundirla con cobertura calibrada.'))
