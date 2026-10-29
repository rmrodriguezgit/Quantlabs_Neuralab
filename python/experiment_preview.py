"""Comprobación de datos y presupuesto sin importar TensorFlow ni entrenar."""
from financial_lab import config,panel,split_summary
from features import bounded_int

def preview(req):
    cfg=config(req.get('config'));datasets=req.get('datasets',[]);kind=req.get('kind','unit');unit=bounded_int(req.get('unit',2),2,10,'Unidad');warnings=[]
    if not isinstance(datasets,list) or not 1<=len(datasets)<=10:raise ValueError('Paso 2: selecciona de uno a diez activos OHLC diarios.')
    mode=req.get('mode','predictive');trading=(kind=='unit' and unit==10) or (kind=='bank' and mode=='trading')
    if trading and len(datasets)!=1:raise ValueError('Paso 2: trading requiere exactamente un activo; desmarca los restantes.')
    if kind=='bank':
        from emulator import normalize_search,PREDICTIVE,GENERATION,TRADING
        allowed=PREDICTIVE if mode=='predictive' else ['logistic','mlp','cnn','rnn','lstm','gru','bilstm','bigru'] if mode=='classification' else GENERATION if mode=='generation' else TRADING if mode=='trading' else []
        families=req.get('families',[])
        if not isinstance(families,list) or not families or any(f not in allowed for f in families):raise ValueError('Paso 1: selecciona al menos una familia compatible con la tarea.')
        search=normalize_search(req.get('search') or {},cfg);lookback=max(search['lookbacks']);fits=len(set(families))*bounded_int(req.get('trialsPerFamily',1),1,12,'Variantes')
        if any(f not in ('ridge','logistic','pca_ridge') for f in families) and not search['optimizers']:raise ValueError('Selecciona un optimizador.')
    else:
        lookback=cfg['lookback'] if unit in (6,7) else 1
        fits={2:1,3:3,4:6 if cfg['includeEarlyStopping'] else 5,5:bounded_int(cfg.get('trials',4),1,30,'Trials')*(len(cfg.get('foldYears',[2022,2023])))+1,6:2,7:len(cfg.get('recurrentModels',['lstm','gru','bilstm'])),8:2,9:len(cfg.get('generativeModels',['cvae','cgan'])),10:2}[unit]
        if fits==0:raise ValueError('Paso 3: selecciona al menos un modelo.')
    if (kind=='unit' and unit==9) or (kind=='bank' and mode=='generation'):
        from unit_generative import windows
        p=windows(datasets,cfg);splits={name:{'samples':int(mask.sum())} for name,mask in p['masks'].items()};criterion='Distribución en validación; ruido fijo'
    else:
        p=panel(datasets,cfg,lookback);splits=split_summary(p);criterion='Retorno neto con costes' if trading else 'Reconstrucción en validación' if kind=='unit' and unit==8 else 'BCE de validación' if mode=='classification' or kind=='unit' and (unit==4 or unit in (2,3) and cfg.get('task')=='classification') else 'MSE/RMSE de validación'
        if int(p['masks']['train'].sum())<16:raise ValueError('Paso 2: hacen falta al menos 16 muestras de entrenamiento después del calentamiento.')
    if kind=='unit' and unit==5:
        import pandas as pd
        years=cfg.get('foldYears',[2022,2023])
        for year in years:
            start=pd.Timestamp(year=int(year),month=1,day=1);end=pd.Timestamp(year=int(year)+1,month=1,day=1)
            if end>pd.Timestamp(cfg['validationStart']):raise ValueError('Paso 3: los folds internos deben terminar antes de la validación externa.')
            tr=sum(r['split']=='train' and pd.Timestamp(r['targetDate'])<start for r in p['meta']);va=sum(r['split']=='train' and pd.Timestamp(r['signalDate'])>=start and pd.Timestamp(r['targetDate'])<end for r in p['meta'])
            if tr<16 or va<8:raise ValueError(f'Fold {year}: requiere 16 muestras train y ocho de validación; hay {tr}/{va}.')
    if kind=='unit' and unit==6 and cfg.get('cnnPadding','valid')=='valid':
        needed=1+sum(l['kernelSize']-1 for l in cfg['architecture']['layers'])
        if cfg['lookback']<needed:raise ValueError(f'Paso 3: la CNN requiere una ventana de al menos {needed} sesiones con esos kernels; aumenta la ventana o usa padding causal.')
    if trading:
        batch=bounded_int(cfg.get('replayBatch',64),8,256,'Lote replay');bounded_int(cfg.get('replayWarmup',128),batch,cfg.get('replayCapacity',10000),'Warmup replay')
        if cfg.get('epsilonEnd',.05)>cfg.get('epsilonStart',1):raise ValueError('Epsilon final debe ser menor o igual al inicial.')
        if cfg.get('episodes',10)*splits['train']['samples']>200000:raise ValueError('El presupuesto supera 200,000 pasos. Reduce episodios o historial.')
        warnings.append('Los agentes usan episodios; Early Stopping por épocas no se aplica.')
    elif cfg['earlyStopping'] and cfg['warmupEpochs']>=cfg['epochs']:warnings.append('El calentamiento alcanza el máximo de épocas: no habrá parada anticipada en este ensayo.')
    if not trading and cfg['epochs']>=1000 and not cfg['earlyStopping']:warnings.append('Early Stopping está apagado: cada red ejecutará el presupuesto completo.')
    if fits>10:warnings.append('Se probarán muchas configuraciones. Conviene validar primero una prueba rápida.')
    return {'valid':True,'splits':splits,'purgedSamples':p['purged'],'excludedSamples':p.get('excluded',0),'fits':fits,'maxEpochs':None if trading else cfg['epochs'],'earlyStopping':cfg['earlyStopping'] if not trading else None,'criterion':criterion,'warnings':warnings,'assets':[d['name'] for d in datasets],'message':'Datos y particiones comprobados. No se ha entrenado ningún modelo.'}
