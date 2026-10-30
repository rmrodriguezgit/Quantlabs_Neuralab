import numpy as np
from domain.contracts import MethodSpec,VariantSpec
from .common import output,params,bounded_int
SPECS=(VariantSpec(1,'GARCH(1,1) normal','Varianza simétrica y residuos normales.',{'paths':1000}),VariantSpec(2,'EGARCH asimétrico','Log-varianza y efecto asimétrico de shocks.',{'paths':1000}),VariantSpec(3,'GJR-GARCH t-Student','Asimetría y colas pesadas con t estandarizada.',{'paths':1000}))
def forecast(ctx,variant):
    from arch import arch_model
    p=params(ctx,variant,SPECS);y=ctx.values
    if np.any(y<=0):raise ValueError('GARCH de retornos requiere precios positivos.')
    r=np.diff(np.log(y))*100
    if len(r)<30 or r.std()<1e-8:raise ValueError('GARCH requiere al menos treinta retornos no constantes.')
    m=arch_model(r,mean='Constant',vol='EGARCH' if variant==2 else 'GARCH',p=1,o=1 if variant>=2 else 0,q=1,dist='t' if variant==3 else 'normal',rescale=False).fit(disp='off',options={'maxiter':2000})
    if m.convergence_flag!=0:raise ValueError('GARCH no convergió; cambia historial o escala.')
    count=bounded_int(p.get('paths',1000),100,5000,'Trayectorias');rng=np.random.default_rng(ctx.seed);nu=float(m.params.get('nu',8));innovations=lambda size:rng.standard_t(nu,size=size)*np.sqrt((nu-2)/nu) if variant==3 else rng.normal(size=size)
    f=m.forecast(horizon=ctx.horizon,method='simulation' if variant==2 else 'analytic',simulations=count,rng=innovations,reindex=False);mu=np.asarray(f.mean)[-1]/100;variance=np.asarray(f.variance)[-1]/10000;sigma=np.sqrt(np.maximum(variance,1e-16));paths=y[-1]*np.exp(np.cumsum(mu+innovations((count,ctx.horizon))*sigma,axis=1))
    return output(np.median(paths,axis=0),ctx,volatility=sigma.tolist(),variance=variance.tolist(),lower=np.quantile(paths,.025,axis=0).tolist(),upper=np.quantile(paths,.975,axis=0).tolist(),parameters={str(k):float(v) for k,v in m.params.items()},note='El objeto principal es volatilidad. El precio es una proyección de escenarios, no una garantía de dirección. Se reporta también QLIKE con proxy de retorno diario al cuadrado.')
STUDY=MethodSpec('garch','GARCH / EGARCH / GJR-GARCH','garch',SPECS,forecast,('Transformar precios positivos en retornos.','Ajustar media y varianza condicional con TR.','Introducir asimetría o colas según variante.','Proyectar volatilidad y escenarios de precios.','Separar error del precio de calidad de volatilidad.'),'arch')
