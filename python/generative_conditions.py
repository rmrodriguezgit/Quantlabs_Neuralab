"""Condiciones observables antes de la ventana; esquema portable de inferencia."""
import numpy as np
from features import bounded_int

CONDITIONS=('volatility','momentum','volume','asset','regime')
def options(cfg):
    names=cfg.get('conditionFeatures',['volatility'])
    if not isinstance(names,list) or not names or len(names)!=len(set(names)) or any(n not in CONDITIONS for n in names):raise ValueError('Selecciona condiciones válidas y únicas.')
    return names,bounded_int(cfg.get('conditionPeriod',20),2,120,'Periodo condición')

def observation(frame,start,cfg,asset=0):
    names,period=options(cfg)
    if start<period+1:raise ValueError('Historial insuficiente para condición causal.')
    prices=frame.Close.iloc[start-period-1:start].to_numpy();r=np.log(prices[1:]/prices[:-1])
    if len(r)!=period or not np.isfinite(r).all():raise ValueError('Retornos previos no finitos.')
    vol=max(float(np.std(r,ddof=1)),1e-8);momentum=float(r.sum());values=[]
    for name in names:
        if name=='volatility':value=np.log(vol)
        elif name=='momentum':value=momentum
        elif name=='volume':
            if 'Volume' not in frame:raise ValueError('Condición volumen: todos los activos necesitan una columna Volume.')
            v=frame.Volume.iloc[start-period:start].to_numpy()
            if not np.isfinite(v).all() or (v<0).any():raise ValueError('Volume debe ser finito y no negativo.')
            value=float(np.log1p(v).mean())
        elif name=='asset':value=asset
        else:value=0 if momentum < -vol*np.sqrt(period) else 2 if momentum > vol*np.sqrt(period) else 1
        values.append(value)
    return np.asarray(values,dtype=np.float32)

def fit_schema(raw,train,cfg,datasets):
    names,_=options(cfg);mean=raw[train].mean(axis=0);scale=np.maximum(raw[train].std(axis=0),1e-8)
    categories={}
    for i,name in enumerate(names):
        if name in ('asset','regime'):mean[i]=0;scale[i]=1;categories[str(i)]=len(datasets) if name=='asset' else 3
    return {'version':1,'names':names,'mean':mean.tolist(),'scale':scale.tolist(),'categories':categories,'assets':[d['id'] for d in datasets],'regimeLabels':['bajista','lateral','alcista'],'regimeRule':'momentum previo frente a ±volatilidad previa*sqrt(periodo); sin etiquetas futuras'}

def transform(raw,schema):return ((np.asarray(raw)-np.asarray(schema['mean']))/np.asarray(schema['scale'])).astype(np.float32)

def inference_condition(frame,cfg,schema,asset=0,regime=None):
    # start=len(frame) includes the last observed close and never a generated return.
    raw=observation(frame,len(frame),cfg,asset)
    if regime is not None:
        if 'regime' not in schema['names']:raise ValueError('Este modelo no utiliza condición de régimen.')
        if regime not in (0,1,2):raise ValueError('Régimen: 0 bajista, 1 lateral, 2 alcista.')
        raw[schema['names'].index('regime')]=regime
    return transform(raw,schema),raw
