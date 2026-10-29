"""Auditoría de particiones, escalas y generalización. No selecciona usando test."""
import numpy as np
from financial_lab import regression_metrics,scale_panel,split_summary

def data_audit(p,cfg,scaled=None,datasets=None):
    s=scaled or scale_panel(p);tr=p['masks']['train'];center=float(p['y'][tr].mean());std=float(p['y'][tr].std());splits={};warnings=[]
    training_last=max(m['targetDate'] for m,b in zip(p['meta'],tr) if b)
    validation_first=min(m['signalDate'] for m,b in zip(p['meta'],p['masks']['validation']) if b)
    validation_last=max(m['targetDate'] for m,b in zip(p['meta'],p['masks']['validation']) if b)
    test_first=min(m['signalDate'] for m,b in zip(p['meta'],p['masks']['test']) if b)
    chronological=training_last<validation_first and validation_last<test_first
    if not chronological:raise ValueError('Auditoría temporal: etiquetas TR/VA/TE se solapan.')
    for name,mask in p['masks'].items():
        values=p['y'][mask];x=s['X'][mask].reshape(-1,p['X'].shape[-1]);assets=[]
        for asset in dict.fromkeys(m['assetId'] for m in p['meta']):
            rows=[m for m,keep in zip(p['meta'],mask) if keep and m['assetId']==asset]
            if rows:assets.append({'asset':rows[0]['asset'],'samples':len(rows),'firstSignal':min(r['signalDate'] for r in rows),'lastTarget':max(r['targetDate'] for r in rows)})
        splits[name]={'samples':int(mask.sum()),'targetMean':float(values.mean()),'targetStd':float(values.std()),'targetMin':float(values.min()),'targetMax':float(values.max()),'upFraction':float(np.mean(values>0)),'meanShiftInTrainStd':float(abs(values.mean()-center)/std) if std>1e-12 else None,'outsideFiveScaledUnits':float(np.mean(np.abs(x)>5)),'assets':assets}
    va=splits['validation']
    if va['samples']<100:warnings.append('Validación con menos de 100 etiquetas: las métricas pueden ser inestables. Amplía historial o usa folds temporales.')
    if va['meanShiftInTrainStd'] is not None and va['meanShiftInTrainStd']>1:warnings.append('La media del objetivo cambia más de una desviación TR en VA. Compara una ventana de entrenamiento más reciente, sin tocar TE.')
    if va['outsideFiveScaledUnits']>.05 and s.get('featureMethod')=='standard':warnings.append('Más del 5% de valores VA están a más de cinco unidades escaladas. Revisa outliers y compara RobustScaler, sin recortar colas automáticamente.')
    if std<1e-8:warnings.append('Objetivo casi constante en TR: revisa columna, unidades y datos; R² puede ser indefinido o inestable.')
    if any(d.get('simulationMode')=='noise' or d.get('name')=='DEMO OHLC · sintético' for d in datasets or []):warnings.append('Demo de ruido: la dirección/retorno intradía se genera aleatoriamente. Un R² alto fuera de muestra no es el resultado esperado. Usa la demo con señal controlada para comprobar aprendizaje.')
    if 'return_current' not in cfg['selectedFeatures']:warnings.append('No incluyes el retorno conocido al cierre actual: ret_lag1 empieza una sesión antes. Puedes comparar return_current e intraday_current usando solo VA.')
    rows=p['X'][tr].reshape(-1,p['X'].shape[-1]);constant=np.std(rows,axis=0)<1e-8
    features=[{'name':f,'center':float(s['featureMean'][i]),'scale':float(s['featureScale'][i]),'constantInTrain':bool(constant[i])} for i,f in enumerate(cfg['selectedFeatures'])]
    return {'chronological':chronological,'fitUses':'TR solamente; filas históricas repetidas se cuentan una vez por activo/fecha','targetUnits':'bps log intradía siguiente','mseUnits':'bps²','featureMethod':s.get('featureMethod','standard'),'targetMethod':s.get('targetMethod','standard'),'targetCenter':s['targetMean'],'targetScale':s['targetScale'],'fitFeatureRows':s.get('fitFeatureRows'),'fitTargetRows':s.get('fitTargetRows'),'firstTrainInput':min(m.get('inputStartDate',m['signalDate']) for m,b in zip(p['meta'],tr) if b),'lastTrainTarget':training_last,'splits':splits,'features':features,'purgedSamples':p['purged'],'warmupDropped':p.get('warmupDropped',0),'warnings':warnings,'policy':'TR aprende pesos/escalas; VA selecciona checkpoint/variante; TE evalúa después. El contexto histórico de una ventana VA/TE puede contener sesiones previas al corte: es información conocida, no etiquetas futuras.'}

def prediction_audit(p,cfg,predict,scaled,parameters=0,task='regression',datasets=None):
    audit=data_audit(p,cfg,scaled,datasets);parts={};tr=p['masks']['train'];mean=float(p['y'][tr].mean());recommendations=[]
    if task!='regression':return audit
    for name,mask in p['masks'].items():
        y=p['y'][mask];pred=np.asarray(predict(mask),dtype=float);metric=regression_metrics(y,pred);baseline=regression_metrics(y,np.full(len(y),mean));zero=regression_metrics(y,np.zeros(len(y)))
        parts[name]={'metrics':metric,'meanTrainBaseline':baseline,'zeroReturnBaseline':zero,'mseNormalized':metric['mse']/scaled['targetScale']**2,'skillVsTrainMean':float(1-metric['mse']/baseline['mse']) if baseline['mse']>0 else None,'predictionStd':float(pred.std()),'targetStd':float(y.std())}
    train_mse=parts['train']['metrics']['mse'];val_mse=parts['validation']['metrics']['mse'];base_mse=parts['validation']['meanTrainBaseline']['mse']
    if val_mse>=base_mse:recommendations.append('No superas en VA a predecir la media de TR. Mantén esa referencia, reduce complejidad y compara indicadores/ventanas; no aumentes épocas por defecto.')
    if train_mse>0 and val_mse/train_mse>2:recommendations.append('Brecha TR→VA grande: posible sobreajuste o cambio de régimen. Prueba menos capas/neuronas, dropout moderado y Early Stopping; elige solo con VA.')
    if parameters>int(tr.sum())*10:recommendations.append('Más de diez parámetros por muestra TR: usa una red más compacta y compara con Ridge. Es una señal de complejidad, no un umbral universal.')
    if parts['validation']['predictionStd']<parts['validation']['targetStd']*.1:recommendations.append('Predicción casi plana en VA: revisa señal, activaciones, LR y objetivo. Una predicción media puede ser razonable si no hay señal; compruébalo con la demo controlada.')
    recommendations.append('R² = 1 − MSE/varianza del objetivo en esa partición. Cambiar de unidad o escalar/desescalar correctamente no aumenta R² por sí mismo.')
    audit.update(predictionSplits=parts,recommendations=recommendations,parametersPerTrainSample=parameters/int(tr.sum()),recommendationsUse='Solo TR/VA; TE no se usa para elegir ajustes.')
    return audit
