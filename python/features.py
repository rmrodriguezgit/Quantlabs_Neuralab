"""Características incrementales: cada fila t utiliza únicamente valores hasta t."""
from collections import deque
import math
import numpy as np


def bounded_int(value,low,high,label):
    if isinstance(value,bool):raise ValueError(f'{label}: se requiere un entero.')
    try:
        numeric=float(value);result=int(numeric)
    except (ValueError,TypeError,OverflowError):raise ValueError(f'{label}: se requiere un entero.')
    if not math.isfinite(numeric) or result!=numeric or not low<=result<=high:raise ValueError(f'{label}: entero entre {low} y {high}.')
    return result


def normalize_features(raw=None):
    raw=raw or {}
    if not isinstance(raw,dict):raise ValueError('Características: objeto inválido.')
    def periods(key,default,limit):
        values=raw.get(key,default)
        if not isinstance(values,list) or len(values)>limit:raise ValueError(f'{key}: máximo {limit} periodos.')
        result=[bounded_int(v,1 if key=='lags' else 2,365,key) for v in values]
        if len(set(result))!=len(result):raise ValueError(f'{key}: no repitas periodos.')
        return sorted(result)
    rsi=raw.get('rsi',False);macd=raw.get('macd',False);returns=raw.get('returns',False);volatility=raw.get('volatility',False)
    if any(not isinstance(v,bool) for v in (rsi,macd,returns,volatility)):raise ValueError('Los indicadores deben estar activados o desactivados.')
    out={'lags':periods('lags',[1,2,3],20),'ema':periods('ema',[],6),'sma':periods('sma',[],6),'rsi':rsi,'rsiPeriod':bounded_int(raw.get('rsiPeriod',14),2,365,'RSI'),'macd':macd,'macdFast':bounded_int(raw.get('macdFast',12),2,365,'MACD rápido'),'macdSlow':bounded_int(raw.get('macdSlow',26),3,365,'MACD lento'),'macdSignal':bounded_int(raw.get('macdSignal',9),2,365,'Señal MACD'),'returns':returns,'volatility':volatility,'volatilityPeriod':bounded_int(raw.get('volatilityPeriod',20),2,365,'Volatilidad')}
    if macd and out['macdFast']>=out['macdSlow']:raise ValueError('MACD: el periodo rápido debe ser menor que el lento.')
    return out


class FeatureStream:
    """EMA sembrada en la primera observación; RSI de Wilder con semilla SMA."""
    def __init__(self,config=None):
        self.config=normalize_features(config);c=self.config
        self.names=['valor']+[f'lag_{p}' for p in c['lags']]+[f'ema_{p}' for p in c['ema']]+[f'sma_{p}' for p in c['sma']]
        if c['rsi']:self.names+=['rsi']
        if c['macd']:self.names+=['macd','macd_signal','macd_histogram']
        if c['returns']:self.names+=['variacion_relativa']
        if c['volatility']:self.names+=['volatilidad_delta']
        warmups=[0,*c['lags'],*[p-1 for p in c['ema']],*[p-1 for p in c['sma']]]
        if c['rsi']:warmups.append(c['rsiPeriod'])
        if c['macd']:warmups.append(c['macdSlow']+c['macdSignal']-2)
        if c['returns']:warmups.append(1)
        if c['volatility']:warmups.append(c['volatilityPeriod'])
        self.warmup=max(warmups)
        self.values=deque(maxlen=max(367,self.warmup+2))
        self.deltas=deque(maxlen=366);self.count=0
        self.ema={};self.signal=None;self.gain=None;self.loss=None
        self.ema_periods=sorted(set(c['ema']+([c['macdFast'],c['macdSlow']] if c['macd'] else [])))

    def append(self,value):
        value=float(value)
        if not math.isfinite(value):raise ValueError('Valor no finito al calcular características.')
        c=self.config;previous=self.values[-1] if self.values else value
        delta=value-previous
        if self.count:self.deltas.append(delta)
        self.values.append(value);self.count+=1
        for p in self.ema_periods:
            self.ema[p]=value if p not in self.ema else 2/(p+1)*value+(1-2/(p+1))*self.ema[p]
        if c['rsi']:
            p=c['rsiPeriod']
            if self.count==p+1:
                changes=np.asarray(self.deltas)[-p:];self.gain=float(np.maximum(changes,0).mean());self.loss=float(np.maximum(-changes,0).mean())
            elif self.count>p+1:
                self.gain=((p-1)*self.gain+max(delta,0))/p;self.loss=((p-1)*self.loss+max(-delta,0))/p
        macd=0.
        if c['macd']:
            macd=self.ema[c['macdFast']]-self.ema[c['macdSlow']]
            a=2/(c['macdSignal']+1);self.signal=macd if self.signal is None else a*macd+(1-a)*self.signal
        if self.count<=self.warmup:return np.full(len(self.names),np.nan)
        history=list(self.values)
        row=[value,*[history[-p-1] for p in c['lags']],*[self.ema[p] for p in c['ema']],*[float(np.mean(history[-p:])) for p in c['sma']]]
        if c['rsi']:row.append(50. if self.gain==0 and self.loss==0 else 100. if self.loss==0 else 100-100/(1+self.gain/self.loss))
        if c['macd']:row.extend([macd,self.signal,macd-self.signal])
        if c['returns']:row.append(delta/max(abs(previous),1e-8))
        if c['volatility']:row.append(float(np.std(list(self.deltas)[-c['volatilityPeriod']:],ddof=1)))
        result=np.asarray(row,dtype=float)
        if not np.isfinite(result).all():raise ValueError('Características no finitas; revisa la escala de tus datos.')
        return result


def feature_matrix(values,config=None):
    stream=FeatureStream(config)
    rows=np.asarray([stream.append(v) for v in values])
    return rows,stream


def training_windows(values,config,lookback):
    y=np.asarray(values,dtype=float)
    matrix,stream=feature_matrix(y,config)
    start=stream.warmup+lookback
    samples=len(y)-start
    if samples<8:raise ValueError(f'Características/ventana requieren al menos {start+8} observaciones de entrenamiento; hay {len(y)}. Reduce periodos o ventana.')
    if samples*lookback*len(stream.names)>20_000_000:raise ValueError('La matriz supera 20 millones de elementos; reduce registros, ventana o características.')
    # Only feature rows used to predict observed training targets enter the scaler.
    fit_rows=matrix[stream.warmup:-1]
    mean=fit_rows.mean(axis=0);scale=np.maximum(fit_rows.std(axis=0),1e-8)
    normalized=(matrix-mean)/scale
    X=np.asarray([normalized[i-lookback:i] for i in range(start,len(y))],dtype=np.float32)
    target=y[start:];ym=float(target.mean());ys=max(float(target.std()),1e-8)
    return {'X':X,'target':((target-ym)/ys).astype(np.float32),'targetMean':ym,'targetScale':ys,'featureMean':mean,'featureScale':scale,'matrix':matrix,'stream':stream,'samples':samples,'start':start}
