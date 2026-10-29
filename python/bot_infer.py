"""Inferencia del paquete exportado. No envía órdenes a intermediarios."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
import pandas as pd
from financial_lab import indicators

def infer(bundle,asset=0,csv=None,position=0.,scenarios=32,asof=None,regime=None):
    root=Path(bundle);manifest=json.loads((root/'manifest.json').read_text());assets=manifest['assets']
    for name,checksum in manifest.get('files',{}).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=checksum:raise ValueError('Archivo modificado o dañado: '+name)
    if not 0<=int(asset)<len(assets):raise ValueError('Índice de activo no disponible.')
    if not 1<=int(scenarios)<=512:raise ValueError('Escenarios: de 1 a 512.')
    entry=assets[int(asset)];frame=pd.read_csv(root/entry['contextFile'],parse_dates=['date']).set_index('date')
    if csv:
        extra=pd.read_csv(csv,parse_dates=['date']).set_index('date');frame=pd.concat([frame,extra]);frame=frame[~frame.index.duplicated(keep='last')].sort_index()
    if asof:frame=frame.loc[:pd.Timestamp(asof)]
    if frame.empty or not np.isfinite(frame[['Open','High','Low','Close']].to_numpy()).all() or (frame[['Open','High','Low','Close']]<=0).any().any():raise ValueError('Se requieren precios OHLC positivos, completos y finitos.')
    cfg=manifest['config']
    if frame.index[-1]<pd.Timestamp(cfg['testStart']):raise ValueError('Este paquete fue seleccionado con validación posterior a esa fecha; no constituye un checkpoint histórico para ese asof.')
    kind=manifest['kind'];s=manifest['scalers'];output={'model':manifest['family'],'asset':entry['name'],'signalDate':frame.index[-1].isoformat(),'paper':True,'ordersSubmitted':False}
    if kind=='generation':
        import tensorflow as tf
        returns=np.log(frame.Close/frame.Close.shift(1)).dropna().to_numpy();period=cfg.get('conditionPeriod',20)
        if len(returns)<period:raise ValueError('Historial insuficiente para condición.')
        latent=cfg.get('latentDim',4);rng=np.random.default_rng(manifest['seed']);z=rng.normal(size=(scenarios,latent)).astype(np.float32)
        if 'conditionSchema' in s:
            from generative_conditions import inference_condition
            c,raw=inference_condition(frame,cfg,s['conditionSchema'],int(asset),regime);conditions=np.repeat(c[None],scenarios,axis=0);output['conditions']={'names':s['conditionSchema']['names'],'values':raw.tolist(),'regimeOverride':regime}
        else:
            c=np.log(max(float(np.std(returns[-period:],ddof=1)),1e-8));conditions=np.full((scenarios,1),(c-s['conditionMean'])/s['conditionScale'],np.float32)
        model=tf.keras.models.load_model(root/'generator.keras',compile=False);ret=model(np.concatenate([z,conditions],axis=1),training=False).numpy()*s['returnScale']+s['returnMean'];output.update(returnScenarios=ret.tolist(),priceScenarios=(frame.Close.iloc[-1]*np.exp(np.cumsum(ret,axis=1))).tolist(),note='Escenarios condicionales sin calibración de cobertura.');return output
    features=indicators(frame,cfg)[manifest['features']].to_numpy();lookback=manifest['lookback'];X=features[-lookback:]
    if len(X)!=lookback or not np.isfinite(X).all():raise ValueError('Historial insuficiente o indicadores no finitos.')
    X=((X-np.asarray(s['featureMean']))/np.asarray(s['featureScale'])).astype(np.float32)
    if kind=='trading':
        import tensorflow as tf
        if position not in (-1.,0.,1.):raise ValueError('Posición permitida: -1, 0 o 1.')
        model=tf.keras.models.load_model(root/'online.keras',compile=False);Q=model(np.r_[X[-1],position][None].astype(np.float32),training=False).numpy()[0];output.update(qValues=Q.tolist(),targetPosition=int(np.argmax(Q))-1,execution='siguiente apertura observada');return output
    if kind=='logistic':
        weights=json.loads((root/'logistic.json').read_text());logit=float(X.ravel()@np.asarray(weights['coef'])+weights['intercept']);probability=float(1/(1+np.exp(-np.clip(logit,-500,500))));output.update(probabilityUp=probability,predictedClass=int(probability>=.5),signal=int(probability>=.5),execution='entrada en próxima apertura y salida en su cierre');return output
    if kind=='classifier':
        import tensorflow as tf
        model=tf.keras.models.load_model(root/'predictor.keras',compile=False);logit=float(model(X[None],training=False).numpy().ravel()[0]);probability=float(1/(1+np.exp(-np.clip(logit,-500,500))));output.update(probabilityUp=probability,predictedClass=int(probability>=.5),signal=int(probability>=.5),execution='entrada en próxima apertura y salida en su cierre');return output
    if kind=='ridge':
        ridge=json.loads((root/'ridge.json').read_text());value=float(X.ravel()@np.asarray(ridge['coef'])+ridge['intercept'])
    elif kind=='latent':
        x=X[-1:]
        if (root/'encoder.keras').exists():
            import tensorflow as tf
            latent=tf.keras.models.load_model(root/'encoder.keras',compile=False)(x,training=False).numpy()
        else:
            pca=json.loads((root/'pca.json').read_text());latent=(x-np.asarray(pca['mean']))@np.asarray(pca['components']).T
        ridge=json.loads((root/'ridge.json').read_text());value=float(np.asarray(latent@np.asarray(ridge['coef'])+ridge['intercept']).ravel()[0])
    else:
        import tensorflow as tf
        model=tf.keras.models.load_model(root/'predictor.keras',compile=False);value=float(model(X[None],training=False).numpy().ravel()[0])
    value=value*s['targetScale']+s['targetMean']
    if not math.isfinite(value):raise ValueError('Predicción no finita; señal rechazada.')
    threshold=2*cfg.get('costBps',5);output.update(predictedIntradayBps=value,signal=int(np.sign(value)) if abs(value)>threshold else 0,signalThresholdBps=threshold,execution='entrada en próxima apertura y salida en su cierre',note='El umbral cubre una aproximación a costes; no garantiza beneficio.');return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='.');p.add_argument('--asset',type=int,default=0);p.add_argument('--csv');p.add_argument('--position',type=float,default=0);p.add_argument('--scenarios',type=int,default=32);p.add_argument('--asof');p.add_argument('--regime',type=int,choices=[0,1,2]);p.add_argument('--output');args=p.parse_args();result=infer(args.bundle,args.asset,args.csv,args.position,args.scenarios,args.asof,args.regime);text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    else:print(text)
