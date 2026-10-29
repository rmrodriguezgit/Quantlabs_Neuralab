"""Datos OHLC, once indicadores causales y particiones con purga de etiquetas."""
import numpy as np
import pandas as pd
from features import bounded_int
from neural import real,architecture
from training_control import settings,MAX_EPOCHS

FEATURES=['ret_lag1','ret_lag2','momentum5','momentum10','ema10_distance','ema20_distance','rsi14','macd','macd_signal','volatility20','range_ratio']

def config(raw=None):
    raw=raw or {}
    if not isinstance(raw,dict):raise ValueError('Configuración de unidad inválida.')
    val=pd.Timestamp(raw.get('validationStart','2024-01-01'));test=pd.Timestamp(raw.get('testStart','2025-01-01'))
    if val>=test:raise ValueError('El inicio de validación debe preceder al test.')
    begin=pd.Timestamp(raw.get('trainStart') or '2020-01-01');end_value=raw.get('testEnd','2026-01-01');finish=pd.Timestamp(end_value) if end_value else None
    if begin>=val or (finish is not None and finish<=test):raise ValueError('Orden requerido: inicio train < validación < test < fin test.')
    selected=raw.get('selectedFeatures',FEATURES)
    if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected) or any(f not in FEATURES for f in selected):raise ValueError('Selecciona indicadores válidos y únicos.')
    cfg={**raw,'trainStart':begin.isoformat(),'testEnd':finish.isoformat() if finish is not None else None,'validationStart':val.isoformat(),'testStart':test.isoformat(),'selectedFeatures':selected,'seed':bounded_int(raw.get('seed',42),0,2**31-1,'Semilla'),'epochs':bounded_int(raw.get('epochs',100),1,MAX_EPOCHS,'Épocas'),'batchSize':bounded_int(raw.get('batchSize',64),8,256,'Lote'),'lookback':bounded_int(raw.get('lookback',20),1,120,'Ventana'),**settings(raw),'ridgeAlpha':real(raw.get('ridgeAlpha',1),0,1000,'Ridge alpha'),'costBps':real(raw.get('costBps',5),0,100,'Coste bps'),'l1':real(raw.get('l1',1e-5),0,.1,'L1'),'l2':real(raw.get('l2',1e-3),0,.1,'L2'),'weightDecay':real(raw.get('weightDecay',0),0,.1,'Weight decay'),'dropout':real(raw.get('dropout',.3),0,.5,'Dropout'),'gradientClip':real(raw.get('gradientClip',1),.01,10,'Norma gradiente'),'architecture':architecture(raw.get('architecture',{'inputUnits':64,'inputActivation':'relu','layers':[{'units':32,'activation':'relu'}]}))}
    cfg['timeoutMinutes']=bounded_int(raw.get('timeoutMinutes',60),1,240,'Minutos máximos de ejecución')
    if not isinstance(raw.get('includeEarlyStopping',True),bool):raise ValueError('Incluir variante EarlyStopping debe ser booleano.')
    cfg['includeEarlyStopping']=raw.get('includeEarlyStopping',True)
    cfg['featurePeriods']={key:bounded_int(raw.get('featurePeriods',{}).get(key,default),2,365,key) for key,default in {'rsi':14,'volatility':20,'emaFast':10,'emaSlow':20,'macdFast':12,'macdSlow':26,'macdSignal':9}.items()}
    if cfg['featurePeriods']['macdFast']>=cfg['featurePeriods']['macdSlow']:raise ValueError('MACD rápido debe ser menor que lento.')
    return cfg


def ohlc_frame(ds):
    from engine import dataset_frame
    f=dataset_frame(ds).copy();lookup={str(c).lower():c for c in f.columns}
    required=['open','high','low','close']
    if any(c not in lookup for c in required):raise ValueError(f"{ds.get('name','Dataset')}: se requieren columnas Open, High, Low y Close.")
    f=f[[lookup[c] for c in required]].astype(float);f.columns=['Open','High','Low','Close']
    if f.index.tz is not None:f.index=f.index.tz_localize(None)
    if not np.isfinite(f.to_numpy()).all() or (f<=0).any().any():raise ValueError('Los precios OHLC deben ser positivos y finitos.')
    if ((f.High<f[['Open','Close','Low']].max(axis=1))|(f.Low>f[['Open','Close','High']].min(axis=1))).any():raise ValueError('OHLC incoherentes: High/Low no contienen apertura y cierre.')
    if ds.get('frequency') not in ('daily','business'):raise ValueError('Las unidades financieras requieren barras diarias; no se interpretan minutos como sesiones.')
    return f


def indicators(f,cfg):
    p=cfg['featurePeriods'];close=f.Close;r=np.log(close/close.shift(1));ema=lambda span:close.ewm(span=span,adjust=False,min_periods=span).mean()
    delta=close.diff();gain=delta.clip(lower=0);loss=-delta.clip(upper=0)
    # Wilder seed via SMA, followed by recursive updates.
    g=np.full(len(f),np.nan);l=g.copy();n=p['rsi']
    if len(f)>n:
        g[n]=gain.iloc[1:n+1].mean();l[n]=loss.iloc[1:n+1].mean()
        for i in range(n+1,len(f)):g[i]=((n-1)*g[i-1]+gain.iloc[i])/n;l[i]=((n-1)*l[i-1]+loss.iloc[i])/n
    with np.errstate(divide='ignore',invalid='ignore'):rsi=100-100/(1+g/l)
    rsi=np.where((g==0)&(l==0),50,np.where((l==0)&(g>0),100,rsi))
    m=ema(p['macdFast'])-ema(p['macdSlow']);signal=m.ewm(span=p['macdSignal'],adjust=False,min_periods=p['macdSignal']).mean()
    return pd.DataFrame({'ret_lag1':r.shift(1),'ret_lag2':r.shift(2),'momentum5':np.log(close/close.shift(5)),'momentum10':np.log(close/close.shift(10)),'ema10_distance':close/ema(p['emaFast'])-1,'ema20_distance':close/ema(p['emaSlow'])-1,'rsi14':rsi/100-.5,'macd':m/close,'macd_signal':signal/close,'volatility20':r.rolling(p['volatility']).std(ddof=1),'range_ratio':(f.High-f.Low)/close},index=f.index)


def panel(datasets,cfg,lookback=1):
    xs=[];ys=[];meta=[];purged=0;excluded=0
    for asset,ds in enumerate(datasets):
        f=ohlc_frame(ds);features=indicators(f,cfg)[cfg['selectedFeatures']].to_numpy()
        for i in range(lookback-1,len(f)-1):
            window=features[i-lookback+1:i+1]
            if not np.isfinite(window).all():continue
            target=i+1;signal_date=f.index[i];target_date=f.index[target]
            if signal_date<pd.Timestamp(cfg['trainStart']) or (cfg.get('testEnd') and target_date>=pd.Timestamp(cfg['testEnd'])):excluded+=1;continue
            part=partition(signal_date,target_date,cfg)
            if part is None:purged+=1;continue
            xs.append(window);ys.append(10000*np.log(f.Close.iloc[target]/f.Open.iloc[target]))
            meta.append({'asset':ds.get('name',str(asset)),'assetId':str(ds.get('id',asset)),'signalDate':signal_date.isoformat(),'targetDate':target_date.isoformat(),'split':part,'gap':float(f.Open.iloc[target]/f.Close.iloc[i]-1),'intraday':float(f.Close.iloc[target]/f.Open.iloc[target]-1)})
    if not xs:raise ValueError('Sin muestras OHLC completas tras calcular indicadores y ventanas.')
    X=np.asarray(xs,dtype=np.float32);y=np.asarray(ys,dtype=float)
    if X.size>20_000_000:raise ValueError('Más de 20 millones de elementos; reduce activos, ventana o indicadores.')
    masks={part:np.asarray([row['split']==part for row in meta]) for part in ('train','validation','test')}
    for part,mask in masks.items():
        if mask.sum()<8:raise ValueError(f'{part}: se requieren ocho muestras, hay {int(mask.sum())}. Ajusta fechas o historial.')
    return {'X':X,'y':y,'meta':meta,'masks':masks,'purged':purged,'excluded':excluded}


def partition(start,end,cfg):
    val=pd.Timestamp(cfg['validationStart']);test=pd.Timestamp(cfg['testStart'])
    if start<pd.Timestamp(cfg.get('trainStart','1900-01-01')) or (cfg.get('testEnd') and end>=pd.Timestamp(cfg['testEnd'])):return None
    if end<val:return 'train'
    if start>=val and end<test:return 'validation'
    if start>=test:return 'test'
    return None


def scale_panel(p,train_mask=None):
    train=p['masks']['train'] if train_mask is None else train_mask
    rows=p['X'][train].reshape(-1,p['X'].shape[-1]);mean=rows.mean(axis=0);std=np.maximum(rows.std(axis=0),1e-8)
    ym=float(p['y'][train].mean());ys=max(float(p['y'][train].std()),1e-8)
    return {'X':((p['X']-mean)/std).astype(np.float32),'y':((p['y']-ym)/ys).astype(np.float32),'featureMean':mean,'featureScale':std,'targetMean':ym,'targetScale':ys}


def split_summary(p):
    return {part:{'samples':int(mask.sum()),'firstSignal':next(row['signalDate'] for row,m in zip(p['meta'],mask) if m),'lastTarget':next(row['targetDate'] for row,m in zip(reversed(p['meta']),reversed(mask)) if m)} for part,mask in p['masks'].items()}


def demo(req):
    from engine import serialize
    n=bounded_int(req.get('records',1565),150,5000,'Registros OHLC');rng=np.random.default_rng(bounded_int(req.get('seed',42),0,2**31-1,'Semilla'))
    vol=.006+.004*(1+np.sin(np.arange(n)/65))/2;gap=rng.normal(.0001,vol*.4);intra=rng.normal(.0002,vol);opens=[];closes=[];last=100.
    for g,r in zip(gap,intra):o=last*np.exp(g);last=o*np.exp(r);opens.append(o);closes.append(last)
    opens=np.asarray(opens);closes=np.asarray(closes);spread=np.abs(rng.normal(.003,.001,n));f=pd.DataFrame({'Open':opens,'High':np.maximum(opens,closes)*(1+spread),'Low':np.minimum(opens,closes)*(1-spread),'Close':closes,'Volume':rng.integers(1000000,5000000,n)},index=pd.bdate_range(req.get('start','2020-01-01'),periods=n))
    return serialize(f,'DEMO OHLC · sintético','financial','business')


def regression_metrics(y,pred):
    from engine import metrics
    return metrics(y,pred)


def classification_metrics(y,logits):
    from scipy.special import expit
    from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss,confusion_matrix,f1_score
    probability=expit(np.asarray(logits));truth=(np.asarray(y)>0).astype(int);signal=(probability>=.5).astype(int)
    bce=float(np.mean(np.logaddexp(0,logits)-truth*np.asarray(logits)))
    return {'bce':bce,'accuracy':float(accuracy_score(truth,signal)),'auc':float(roc_auc_score(truth,probability)) if len(np.unique(truth))>1 else None,'brier':float(brier_score_loss(truth,probability)),'f1':float(f1_score(truth,signal,zero_division=0)),'confusion':confusion_matrix(truth,signal,labels=[0,1]).tolist()}


def trade_path(meta,positions,cost_bps=5,liquidate=True,overnight=True):
    if len(meta)!=len(positions):raise ValueError('Posiciones y sesiones no coinciden.')
    previous={};equity=1.;peak=1.;rows=[];returns=[];turnover=0.;costs=0.
    last_by_asset={row['assetId']:i for i,row in enumerate(meta)}
    for i,(row,new) in enumerate(zip(meta,positions)):
        new=float(new);old=previous.get(row['assetId'],0.);turn=abs(new-old);cost=cost_bps/10000*turn
        if overnight:net=(1+old*row['gap'])*(1-cost)*(1+new*row['intraday'])-1
        else:
            cost=cost_bps/10000*2*abs(new);turn=2*abs(new);net=(1-cost_bps/10000*abs(new))*(1+new*row['intraday'])*(1-cost_bps/10000*abs(new))-1
        if overnight and liquidate and i==last_by_asset[row['assetId']]:net=(1+net)*(1-cost_bps/10000*abs(new))-1;cost+=cost_bps/10000*abs(new);turn+=abs(new)
        if not np.isfinite(net) or 1+net<=0:raise ValueError('El escenario agota el capital o produce un retorno no finito.')
        equity*=1+net;peak=max(peak,equity);turnover+=turn;costs+=cost;returns.append(net);previous[row['assetId']]=new
        rows.append({'date':row['targetDate'],'asset':row['asset'],'position':new,'return':net,'equity':equity,'drawdown':equity/peak-1,'cost':cost})
    std=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.
    return {'metrics':{'netReturn':equity-1,'sharpe':float(np.mean(returns)/std*np.sqrt(252)) if std>1e-12 else None,'maxDrawdown':min(r['drawdown'] for r in rows),'turnover':turnover,'costSum':costs},'path':rows}
