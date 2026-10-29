"""Protocolo: un objeto JSON por stdin, una respuesta JSON por stdout."""
import sys, json, contextlib, importlib.util, math, uuid, copy
from pathlib import Path
import numpy as np
import pandas as pd
from models import predict, METHODS

FREQ={'daily':'D','business':'B','monthly':'MS','annual':'YS','hourly':'h','15min':'15min'}
PERIOD={'daily':7,'business':5,'monthly':12,'annual':1,'hourly':24,'15min':96}

def integer(value,low,high,label):
    v=int(value)
    if v<low or v>high:raise ValueError(f'{label}: rango permitido {low}–{high}.')
    return v

def serialize(frame,name,kind='business',frequency='monthly'):
    frame=frame.copy()
    frame.index=pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:frame.index=frame.index.tz_localize(None)
    frame=frame.sort_index()
    if frame.index.has_duplicates:raise ValueError('Hay fechas duplicadas; agrega o corrige los registros.')
    frame=frame.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    frame=frame.dropna(axis=1,how='all')
    if len(frame)>20000 or len(frame.columns)>30:raise ValueError('Máximo 20,000 filas y 30 variables.')
    if len(frame)<24 or frame.shape[1]==0:raise ValueError('Se requieren al menos 24 filas y una columna numérica.')
    if frame.isna().any().any():raise ValueError('Hay valores faltantes/no numéricos. Corrígelos antes de importar; no se interpolan datos para evitar fuga temporal.')
    return {'id':uuid.uuid4().hex,'name':name,'kind':kind,'frequency':frequency,'columns':list(map(str,frame.columns)), 'rows':[{'date':idx.isoformat(),**{str(k):float(v) for k,v in row.items()}} for idx,row in frame.iterrows()]}

def dataset_frame(ds):
    if ds.get('frequency') not in FREQ:raise ValueError('Frecuencia no compatible.')
    frame=pd.DataFrame(ds['rows'])
    if len(frame)>20000:raise ValueError('Máximo 20,000 registros.')
    frame.index=pd.DatetimeIndex(pd.to_datetime(frame.pop('date'),errors='raise'))
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:raise ValueError('Fechas deben estar ordenadas y ser únicas.')
    return frame

def simulate(req):
    n=integer(req.get('records',144),24,20000,'Registros')
    freq=req.get('frequency','monthly')
    if freq not in ('daily','monthly','annual'):raise ValueError('Frecuencia de simulación inválida.')
    columns=req.get('columns',['Producción','Ventas','Inventarios','Costos','Margen'])
    if not isinstance(columns,list) or not 1<=len(columns)<=30 or len(set(columns))!=len(columns):raise ValueError('Define de 1 a 30 columnas únicas.')
    if any(not isinstance(c,str) or not c.strip() or c=='date' for c in columns):raise ValueError('Nombres de columnas inválidos.')
    seed=integer(req.get('seed',42),0,2**31-1,'Semilla')
    rng=np.random.default_rng(seed);t=np.arange(n);period=PERIOD[freq]
    common=rng.normal(0,1,n)
    data={}
    for j,column in enumerate(columns):
        spec=req.get('specs',{}).get(column,{})
        base=float(spec.get('base',1000+250*j));trend=float(spec.get('trend',3+1.5*j));season=float(spec.get('seasonality',100+15*j));noise=float(spec.get('noise',45))
        if not np.isfinite([base,trend,season,noise]).all() or noise<0:raise ValueError('Parámetros de simulación no válidos.')
        data[column]=base+trend*t+(season*np.sin(2*np.pi*t/period+j*.3) if period>1 else 0)+noise*(.4*common+.6*rng.normal(size=n))
    dates=pd.date_range(req.get('start','2010-01-01'),periods=n,freq=FREQ[freq])
    return serialize(pd.DataFrame(data,index=dates),req.get('name','Escenario empresarial'),'business',freq)

def finance(req):
    import yfinance as yf
    ticker=str(req.get('ticker','AAPL')).strip().upper()
    if not ticker or len(ticker)>30:raise ValueError('Ticker inválido.')
    interval=req.get('interval','1d');period=str(req.get('period','2y')).lower()
    if interval not in ('1d','1h','15m'):raise ValueError('Intervalo no compatible.')
    if period not in ('1y','2y','5y','10y','1mo'):raise ValueError('Rango no compatible.')
    if interval!='1d' and period!='1mo':raise ValueError('Para intradía selecciona 1 mes: Yahoo limita su historial. No se recorta silenciosamente el rango.')
    frame=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False,timeout=25,multi_level_index=False)
    if frame.empty:raise ValueError('Yahoo no devolvió datos. Comprueba ticker, conexión o límite de consultas.')
    frame=frame.dropna()
    return serialize(frame,ticker,'financial',{'1d':'business','1h':'hourly','15m':'15min'}[interval])

def import_files(req):
    out=[]
    paths=req.get('paths',[])
    if len(paths)>20:raise ValueError('Importa hasta 20 archivos por operación.')
    for filename in paths:
        path=Path(filename)
        try:
            if path.stat().st_size>20_000_000:raise ValueError('El archivo supera 20 MB.')
            if path.suffix.lower()=='.xlsx':frame=pd.read_excel(path)
            elif path.suffix.lower()=='.csv':frame=pd.read_csv(path,sep=None,engine='python')
            else:raise ValueError('Usa CSV o XLSX.')
            date=next((c for c in frame.columns if str(c).strip().lower() in ('date','fecha','datetime')),None)
            if date is None:raise ValueError('Incluye una columna date o fecha (ISO 8601).')
            index=pd.to_datetime(frame.pop(date),errors='raise')
            frame.index=pd.DatetimeIndex(index)
            for column in frame.columns:frame[column]=pd.to_numeric(frame[column],errors='raise')
            frame=frame.sort_index()
            diffs=frame.index.to_series().diff().dropna().dt.total_seconds()/86400
            median=float(diffs.median())
            freq='annual' if median>300 else 'monthly' if median>20 else 'daily' if median>=1 else 'hourly' if median>=1/24 else '15min'
            out.append({'dataset':serialize(frame,path.stem,'business',freq)})
        except Exception as exc:out.append({'name':path.name,'error':str(exc)})
    return out

def metrics(actual,predicted):
    a=np.asarray(actual);p=np.asarray(predicted)
    if len(a)!=len(p) or not np.isfinite(p).all():raise ValueError('Predicción inválida.')
    residual=a-p; mse=float(np.mean(residual**2));den=float(np.sum((a-a.mean())**2))
    return {'mse':mse,'rmse':math.sqrt(mse),'mae':float(np.mean(np.abs(residual))),'r2':None if den<1e-12 else float(1-np.sum(residual**2)/den)}

def horizons(ds,index):
    last=index[-1]
    if ds['kind']=='financial':
        # Calendar-day windows; estimate observed bars/day, including exchange closures.
        rate=(len(index)-1)/max((index[-1]-index[0]).total_seconds()/86400,1)
        return [(f'{d} días naturales',max(1,round(d*rate))) for d in (15,30,60)]
    return [(f'{year} año'+('s' if year>1 else ''),max(1,len(pd.date_range(last+pd.tseries.frequencies.to_offset(FREQ[ds['frequency']]), last+pd.DateOffset(years=year),freq=FREQ[ds['frequency']])))) for year in range(1,6)]

def decompose(y,period):
    from statsmodels.tsa.seasonal import STL
    if period>=2 and len(y)>=2*period:
        result=STL(y,period=period,robust=True).fit();tr=result.trend;va=result.seasonal;te=result.resid
        note='STL descriptivo sobre la serie completa; no se usa para entrenar ni evaluar modelos.'
    else:
        tr=pd.Series(y).rolling(min(5,len(y)),min_periods=1,center=True).mean().to_numpy();va=np.zeros(len(y));te=y-tr
        note='Sin dos ciclos completos: tendencia móvil descriptiva, VA=0; no estima estacionalidad.'
    return {'trend':tr.tolist(),'seasonal':va.tolist(),'residual':te.tolist(),'note':note}

def analyze(req):
    ds=req['dataset'];frame=dataset_frame(ds);column=req['column']
    if column not in frame.columns:raise ValueError('Variable no encontrada.')
    y=frame[column].to_numpy(dtype=float)
    if len(y)<24 or not np.isfinite(y).all():raise ValueError('Serie inválida: mínimo 24 observaciones finitas.')
    period=integer(req.get('seasonPeriod',PERIOD[ds['frequency']]),1,365,'Periodo estacional')
    methods=req.get('methods',['naive','linear','arima'])
    if not methods or len(methods)>len(METHODS) or any(m not in METHODS for m in methods):raise ValueError('Selección de métodos inválida.')
    folds=integer(req.get('folds',2),1,5,'Ventanas')
    seed=integer(req.get('seed',42),0,2**31-1,'Semilla')
    # Final holdout reserved before any ranking or model selection.
    test_size=max(4,min(len(y)//5,60));selection=y[:-test_size];holdout=y[-test_size:]
    windows=horizons(ds,frame.index)
    validation_h=integer(req.get('validationSteps',max(2,min(12,len(selection)//8))),1,5000,'Pasos de validación')
    if len(selection)-folds*validation_h<12:raise ValueError('No hay datos suficientes para las ventanas de validación configuradas.')
    options=copy.deepcopy(req.get('options') or {})
    if not isinstance(options,dict):raise ValueError('Opciones inválidas.')
    from neural import network_config
    options['networks']={method:network_config(method,options) for method in methods if method in ('rnn','cnn','lstm','dqn','gan')}
    experiment={'methods':list(methods),'options':options,'seed':seed,'seasonPeriod':period,'folds':folds,'validationSteps':validation_h,'forecastSteps':integer(req.get('forecastSteps',30),1,5000,'Horizonte futuro')}
    ranking=[];failures=[];reports=[]
    def evaluate(name,h):
        actual=[];pred=[];fold_metrics=[]
        for k in range(folds):
            end=len(selection)-(folds-k)*h
            if end<12:raise ValueError(f'Historial insuficiente: {folds} ventanas × {h} pasos + entrenamiento mínimo + test reservado.')
            forecast=predict(name,selection[:end],h,period,seed+k,options)['prediction']
            truth=selection[end:end+h];actual.extend(truth.tolist());pred.extend(forecast);fold_metrics.append(metrics(truth,forecast))
        return {'method':name,**metrics(actual,pred),'folds':folds,'observations':len(actual),'foldMetrics':fold_metrics}
    for name in methods:
        try:ranking.append(evaluate(name,validation_h))
        except Exception as exc:failures.append({'method':name,'scope':'validación','error':str(exc)})
    sort_key=lambda x:(x['rmse'],x['mae'],-(x['r2'] if x['r2'] is not None else -float('inf')))
    ranking.sort(key=sort_key)
    for label,h in windows:
        results=[];errors=[]
        for name in methods:
            try:results.append(evaluate(name,h))
            except Exception as exc:errors.append({'method':name,'error':str(exc)})
        results.sort(key=sort_key)
        reports.append({'label':label,'steps':h,'ranking':results,'errors':errors,'winner':results[0]['method'] if results else None})
    if not ranking:return {'experiment':experiment,'ranking':[],'reports':reports,'failures':failures,'decomposition':decompose(y,period),'dates':[d.isoformat() for d in frame.index],'actual':y.tolist(),'winner':None,'note':'Ningún método pudo evaluarse.'}
    winner=ranking[0]['method']
    # Chosen only on pre-holdout folds; holdout reported exactly once for winner.
    fitted=predict(winner,selection,test_size,period,seed,options)
    test_metrics=metrics(holdout,fitted['prediction'])
    future_h=integer(req.get('forecastSteps',30),1,5000,'Horizonte futuro')
    future=predict(winner,y,future_h,period,seed,options)
    future_dates=pd.date_range(frame.index[-1]+pd.tseries.frequencies.to_offset(FREQ[ds['frequency']]),periods=future_h,freq=FREQ[ds['frequency']])
    residual=holdout-np.asarray(fitted['prediction']);counts,edges=np.histogram(residual,bins=min(12,test_size))
    from scipy.stats import f_oneway
    groups=[y[i::period] for i in range(period)] if period>1 else []
    anova=None
    if len(groups)>1 and all(len(g)>=2 for g in groups):
        result=f_oneway(*groups);anova={'f':float(result.statistic),'p':float(result.pvalue),'note':'ANOVA exploratorio por fase estacional; autocorrelación y tendencia pueden invalidar la inferencia clásica.'}
    return {'experiment':experiment,'winner':winner,'ranking':ranking,'reports':reports,'failures':failures,'test':{'dates':[d.isoformat() for d in frame.index[-test_size:]],'actual':holdout.tolist(),**fitted,'metrics':test_metrics},'future':{'dates':[d.isoformat() for d in future_dates],**future},'dates':[d.isoformat() for d in frame.index],'actual':y.tolist(),'decomposition':decompose(y,period),'residualHistogram':[{'bin':f'{edges[i]:.2f}…{edges[i+1]:.2f}','count':int(counts[i])} for i in range(len(counts))],'anova':anova,'note':'Ranking por RMSE, luego MAE y R² en empates. Ventanas crecientes anteriores al test final; métricas futuras desconocidas. Fechas financieras futuras aproximadas: no incluyen calendario de bolsa.'}

def capabilities():
    return {name:importlib.util.find_spec(module) is not None for name,module in {'numpy':'numpy','pandas':'pandas','statsmodels':'statsmodels','arch':'arch','prophet':'prophet','tensorflow':'tensorflow','gymnasium':'gymnasium','yfinance':'yfinance'}.items()}

def dispatch(req):
    action=req.get('action')
    if action=='preview_experiment':
        from experiment_preview import preview
        return preview(req)
    if action=='bundle_artifacts':
        from model_exports import bundle_artifacts
        return bundle_artifacts(req['artifactRoot'],req['ids'],req.get('report'))
    if action=='emulate':
        from emulator import run
        return run(req)
    if action=='unit_lab':
        from unit_lab import run
        return run(req)
    if action=='demo_ohlc':
        from financial_lab import demo
        return demo(req)
    if action=='simulate':return simulate(req)
    if action=='finance':return finance(req)
    if action=='import':return import_files(req)
    if action=='analyze':return analyze(req)
    if action=='capabilities':return capabilities()
    if action=='export':
        frame=dataset_frame(req['dataset']);frame.index.name='date';path=Path(req['path'])
        if req['format']=='xlsx':frame.to_excel(path)
        elif req['format']=='csv':frame.to_csv(path)
        else:raise ValueError('Formato inválido.')
        return {'saved':True}
    raise ValueError('Acción no reconocida.')

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None
    if isinstance(value,np.integer):return int(value)
    return value

if __name__=='__main__':
    try:
        raw=sys.stdin.read(20_000_001)
        if len(raw)>20_000_000:raise ValueError('Solicitud demasiado grande.')
        req=json.loads(raw)
        # Libraries may print training logs; reserve stdout for protocol JSON only.
        with contextlib.redirect_stdout(sys.stderr):data=dispatch(req)
        response={'ok':True,'data':clean(data)}
    except Exception as exc:response={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    sys.stdout.write(json.dumps(response,ensure_ascii=False,allow_nan=False))
