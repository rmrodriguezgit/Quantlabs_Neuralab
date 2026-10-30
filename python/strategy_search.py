"""Búsqueda escalonada de pronóstico: selección exclusivamente en VA; TE una vez."""
import copy,gc,hashlib,itertools,json,math,sys,time
import numpy as np
from features import bounded_int,normalize_features
from neural import real,network_config
from models import predict

METHODS=('cnn','rnn','lstm')
FEATURE_SETS={
 'minimal':{'lags':[1],'returns':True},
 'compact':{'lags':[1,2],'returns':True,'volatility':True},
 'trend':{'lags':[1,2],'ema':[10,20],'rsi':True,'returns':True,'volatility':True},
}

def progress(stage,completed,total,detail):
 print('LAB_PROGRESS '+json.dumps({'scope':'strategy','stage':stage,'completed':completed,'total':total,'detail':detail},ensure_ascii=False),file=sys.stderr,flush=True)

def array(raw,key,default,lo=None,hi=None,allowed=None):
 value=raw.get(key,default)
 if not isinstance(value,list) or not value or len(value)>20:raise ValueError(f'{key}: lista no vacía de hasta veinte valores.')
 if allowed is not None:
  if any(v not in allowed for v in value):raise ValueError(f'{key}: valores no admitidos.')
  return list(dict.fromkeys(value))
 return list(dict.fromkeys(bounded_int(v,lo,hi,key) for v in value))

def prepare(req):
 from engine import dataset_frame
 ds=req.get('dataset');column=req.get('column');raw=req.get('config') or {}
 if not isinstance(ds,dict) or not isinstance(raw,dict):raise ValueError('Selecciona una serie y configuración válidas.')
 f=dataset_frame(ds)
 if column not in f:raise ValueError('Selecciona una variable numérica.')
 y=f[column].to_numpy(dtype=float)
 if not np.isfinite(y).all():raise ValueError('La serie debe ser finita y completa.')
 c={'horizon':bounded_int(raw.get('horizon',20),1,120,'Horizonte'),'folds':bounded_int(raw.get('folds',6),4,12,'Folds'),'epochs':bounded_int(raw.get('epochs',100),1,5000,'Épocas'),'patience':bounded_int(raw.get('patience',10),1,1000,'Paciencia'),'warmupEpochs':bounded_int(raw.get('warmupEpochs',5),0,5000,'Calentamiento'),'minDelta':real(raw.get('minDelta',1e-5),0,1,'Mejora mínima'),'batchSize':bounded_int(raw.get('batchSize',32),8,256,'Lote'),'screenLimit':bounded_int(raw.get('screenLimit',9),1,81,'Límite etapa 1'),'refineLimit':bounded_int(raw.get('refineLimit',6),0,40,'Límite etapa 2'),'finalists':bounded_int(raw.get('finalists',3),1,5,'Finalistas'),'stabilityPenalty':real(raw.get('stabilityPenalty',.1),0,1,'Penalización estabilidad'),'minimumSkill':real(raw.get('minimumSkill',.01),0,.5,'Mejora mínima contra persistencia'),'earlyStopping':raw.get('earlyStopping',True),'timeoutMinutes':bounded_int(raw.get('timeoutMinutes',120),1,240,'Tiempo máximo')}
 if not isinstance(c['earlyStopping'],bool):raise ValueError('Early Stopping debe ser booleano.')
 c['methods']=array(raw,'methods',list(METHODS),allowed=METHODS);c['histories']=array(raw,'histories',[250,500,1000],80,10000);c['targets']=array(raw,'targets',['level','difference','log_return'],allowed=['level','difference','log_return']);c['seeds']=array(raw,'seeds',[42,43,44],0,2**31-1)
 if not 2<=len(c['seeds'])<=5:raise ValueError('Usa entre dos y cinco semillas distintas para medir estabilidad.')
 c['units']=array(raw,'units',[16,32],1,128);c['depths']=array(raw,'depths',[1,2],1,3);c['lookbacks']=array(raw,'lookbacks',[10,20],2,60)
 c['scalers']=array(raw,'scalers',['standard','robust'],allowed=['standard','robust','minmax','none']);c['featureSets']=array(raw,'featureSets',['minimal','compact','trend'],allowed=list(FEATURE_SETS))
 for key,default,lo,hi in [('learningRates',[.0003,.001],1e-6,.1),('dropouts',[0,.1],0,.5),('modelWeights',[0,.25,.5,.75,1],0,1)]:
  values=raw.get(key,default)
  if not isinstance(values,list) or not 1<=len(values)<=10:raise ValueError(f'{key}: lista de uno a diez valores.')
  c[key]=list(dict.fromkeys(real(v,lo,hi,key) for v in values))
 c['modelWeights']=sorted(set([0.]+c['modelWeights']))
 nselect=len(y)-c['horizon'];nsearch=c['folds']//2;first_train=nselect-c['folds']*c['horizon']
 if first_train<80:raise ValueError('Historial insuficiente para folds + TE y al menos 80 filas iniciales. Reduce horizonte/folds.')
 warnings=[]
 histories=[h for h in c['histories'] if h<=first_train]
 if not histories:raise ValueError(f'La primera TR tiene {first_train} filas. Reduce el historial candidato a ese tamaño o carga más datos.')
 if len(histories)!=len(c['histories']):warnings.append(f'Historiales mayores que la primera TR ({first_train}) se omiten: no se comparan configuraciones equivalentes por truncado.')
 c['histories']=histories
 if 'log_return' in c['targets'] and np.any(y[:nselect]<=0):c['targets'].remove('log_return');warnings.append('Log-retorno omitido: la parte previa a TE contiene valores no positivos.')
 if not c['targets']:raise ValueError('No quedan objetivos compatibles con la serie.')
 for history in histories:
  for lookback in c['lookbacks']:
   for feature in c['featureSets']:
    from features import FeatureStream
    stream=FeatureStream(FEATURE_SETS[feature]);samples=int(history*.9)-stream.warmup-lookback
    if samples<8:raise ValueError(f'Historial {history}, ventana {lookback} e indicadores {feature}: menos de ocho objetivos TR internos. Amplía historial o reduce ventana/indicadores.')
    if samples*lookback*len(stream.names)>20_000_000:raise ValueError('La matriz prevista supera veinte millones de elementos.')
 folds=[]
 for i in range(c['folds']):
  end=nselect-(c['folds']-i)*c['horizon'];folds.append({'index':i,'end':end,'stage':'search' if i<nsearch else 'final_validation','trainFirst':f.index[0].isoformat(),'trainLast':f.index[end-1].isoformat(),'validationFirst':f.index[end].isoformat(),'validationLast':f.index[end+c['horizon']-1].isoformat(),'samples':c['horizon']})
 combos=len(c['methods'])*len(histories)*len(c['targets']);screens=min(combos,c['screenLimit']);finalists=min(c['finalists'],screens+c['refineLimit']);searchfits=(screens+c['refineLimit'])*nsearch;confirmfits=finalists*(c['folds']-nsearch)*len(c['seeds']);estimated=searchfits+confirmfits+2*len(c['seeds'])
 if estimated>600:raise ValueError(f'El plan supera 600 ajustes ({estimated}). Reduce candidatos, folds, semillas o finalistas.')
 if screens<combos:warnings.append(f'Etapa 1 explora {screens} de {combos} combinaciones mediante una selección reproducible; no es exhaustiva.')
 if (c['folds']-nsearch)*c['horizon']<50:warnings.append('VA final tiene menos de 50 etiquetas; las métricas por bloque pueden ser inestables.')
 if not c['earlyStopping']:warnings.append('ES apagado: cada ajuste ejecutará todas las épocas.')
 warnings.append('TE se reserva dentro de esta ejecución. Si ya lo examinaste en otros experimentos, no es una confirmación independiente; necesitarás un periodo nuevo.')
 plan={'config':c,'folds':folds,'searchFolds':nsearch,'finalValidationFolds':c['folds']-nsearch,'searchSamples':nsearch*c['horizon'],'finalValidationSamples':(c['folds']-nsearch)*c['horizon'],'estimatedFitsUpperBound':estimated,'screenCandidates':screens,'refineCandidatesUpperBound':c['refineLimit'],'test':{'samples':c['horizon'],'firstDate':f.index[nselect].isoformat(),'lastDate':f.index[-1].isoformat()},'warnings':warnings,'policy':'Búsqueda usa los primeros folds VA; pesos del ensamble se fijan allí. VA final selecciona con varias semillas. TE se evalúa una vez, solo para el ganador y persistencia.'}
 return y,f,ds,column,plan

def preview(req):return prepare(req)[4]

def identifier(params):return hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]

def parameters(method,history,target):return {'method':method,'history':history,'targetTransform':target,'units':16,'depth':1,'lookback':20,'learningRate':.0003,'dropout':0.,'featureScaler':'standard','featureSet':'trend'}

def neural_config(p,c):
 activation='tanh' if p['method'] in ('rnn','lstm') else 'relu'
 return network_config(p['method'],{'networks':{p['method']:{'inputUnits':p['units'],'inputActivation':'linear','layers':[{'units':p['units'],'activation':activation,'recurrentActivation':'sigmoid','kernelSize':3,'dropout':p['dropout']} for _ in range(p['depth'])],'learningRate':p['learningRate'],'optimizer':'adam','lookback':p['lookback'],'maxTrainSamples':p['history'],'targetTransform':p['targetTransform'],'featureScaler':p['featureScaler'],'targetScaler':'standard','features':FEATURE_SETS[p['featureSet']],'validationFraction':.1,'batchSize':c['batchSize'],'epochs':c['epochs'],'earlyStopping':c['earlyStopping'],'patience':c['patience'],'warmupEpochs':c['warmupEpochs'],'minDelta':c['minDelta']}}})

def choose_weight(truth,model,baseline,weights):
 return min(weights,key=lambda w:(float(np.mean((truth-(w*model+(1-w)*baseline))**2)),w))

def run(req):
 from engine import metrics,FREQ
 started=time.monotonic();y,frame,ds,column,plan=prepare(req);c=plan['config'];nselect=len(y)-c['horizon'];selection=y[:nselect].copy();search=plan['folds'][:plan['searchFolds']];validation=plan['folds'][plan['searchFolds']:];rng=np.random.default_rng(c['seeds'][0]);cache={};trials={};failures=[];completed=0;fit_count=0
 def baseline_for(folds):return np.concatenate([np.repeat(selection[f['end']-1],c['horizon']) for f in folds])
 def truth_for(folds):return np.concatenate([selection[f['end']:f['end']+c['horizon']] for f in folds])
 search_truth=truth_for(search);search_base=baseline_for(search);confirm_truth=truth_for(validation);confirm_base=baseline_for(validation)
 def fit_fold(p,fold,seed):
  nonlocal fit_count,completed
  key=(identifier(p),fold['index'],seed)
  if key in cache:return cache[key]
  progress(fold['stage'],completed,plan['estimatedFitsUpperBound'],f"{p['method']} · historial {p['history']} · {p['targetTransform']} · fold {fold['index']+1} · semilla {seed}")
  cfg=neural_config(p,c);fitted=predict(p['method'],selection[:fold['end']],c['horizon'],5,(seed+1009*fold['index'])%(2**31-1),{'networks':{p['method']:cfg}});values=np.asarray(fitted['prediction'],dtype=float)
  if values.shape!=(c['horizon'],) or not np.isfinite(values).all():raise ValueError('Pronóstico inválido en la búsqueda.')
  info=fitted.get('network',{});scale=info.get('scaling',{});offset=info.get('historyOffset',0);last=scale.get('featureFitLastIndex')
  if last is not None and last+offset>=fold['end']:raise ValueError('Scaler fuera de la TR del fold.')
  cache[key]={'prediction':values,'trainingControl':fitted.get('trainingControl'),'scaleAudit':{'historyOffset':offset,'featureScaler':cfg['featureScaler'],'targetScaler':cfg['targetScaler'],'featureFitLastIndex':last,'featureFitLastDate':frame.index[last+offset].isoformat() if last is not None else None,'trainLastDate':fold['trainLast'],'validationFirst':fold['validationFirst']}};fit_count+=1;completed+=1;gc.collect();return cache[key]
 def screen(p,stage):
  key=identifier(p)
  if key in trials:return trials[key]
  try:
   outputs=[fit_fold(p,f,c['seeds'][0]) for f in search];pred=np.concatenate([o['prediction'] for o in outputs]);row={'id':key,'stage':stage,'parameters':p,'config':neural_config(p,c),'search':metrics(search_truth,pred),'training':[o['trainingControl'] for o in outputs],'scaleAudit':[o['scaleAudit'] for o in outputs]};trials[key]={'row':row,'prediction':pred};return trials[key]
  except Exception as e:failures.append({'stage':stage,'parameters':p,'error':str(e)});progress('failed',completed,plan['estimatedFitsUpperBound'],str(e));return None
 # Round-robin families with reproducible target/history ordering; selected subset is explicit.
 pool={m:list(itertools.product(c['histories'],c['targets'])) for m in c['methods']}
 for values in pool.values():rng.shuffle(values)
 initial=[]
 while any(pool.values()):
  for method in c['methods']:
   if pool[method]:h,t=pool[method].pop();initial.append(parameters(method,h,t))
 for p in initial[:plan['screenCandidates']]:
  p['lookback']=c['lookbacks'][-1];p['featureSet']='trend' if 'trend' in c['featureSets'] else c['featureSets'][0];p['featureScaler']='standard' if 'standard' in c['scalers'] else c['scalers'][0];screen(p,'history_target')
 ordered=sorted(trials.values(),key=lambda t:t['row']['search']['mse']);bases=[]
 for method in c['methods']:
  match=next((t for t in ordered if t['row']['parameters']['method']==method),None)
  if match:bases.append(match['row']['parameters'])
 # Refine only configurations chosen from search folds; confirmation data remains untouched.
 attempts=0;refined=0
 while bases and refined<c['refineLimit'] and attempts<1000:
  p=copy.deepcopy(bases[refined%len(bases)]);attempts+=1
  for field,choices in [('units',c['units']),('depth',c['depths']),('lookback',c['lookbacks']),('learningRate',c['learningRates']),('dropout',c['dropouts']),('featureScaler',c['scalers']),('featureSet',c['featureSets'])]:p[field]=choices[int(rng.integers(len(choices)))]
  if identifier(p) in trials:continue
  screen(p,'architecture_features_scaling');refined+=1
 ordered=sorted(trials.values(),key=lambda t:t['row']['search']['mse']);nominees=[]
 for method in c['methods']:
  match=next((t for t in ordered if t['row']['parameters']['method']==method),None)
  if match:nominees.append(match)
 nominees=sorted(nominees,key=lambda t:t['row']['search']['mse'])[:c['finalists']]
 for candidate in ordered:
  if len(nominees)>=c['finalists']:break
  if all(t['row']['id']!=candidate['row']['id'] for t in nominees):nominees.append(candidate)
 baseline_fold=[metrics(selection[f['end']:f['end']+c['horizon']],np.repeat(selection[f['end']-1],c['horizon'])) for f in validation];baseline_metric=metrics(confirm_truth,confirm_base)
 rows=[];stored_predictions={}
 for candidate in nominees:
  row=candidate['row'];p=row['parameters'];weight=choose_weight(search_truth,candidate['prediction'],search_base,c['modelWeights'])
  try:
   seed_predictions=[];confirm_audits=[]
   for seed in c['seeds']:
    outputs=[fit_fold(p,f,seed) for f in validation] if weight>0 else [];confirm_audits.extend([{'seed':seed,**o['scaleAudit']} for o in outputs]);raw=np.concatenate([o['prediction'] for o in outputs]) if outputs else confirm_base.copy();seed_predictions.append(weight*raw+(1-weight)*confirm_base)
   aggregate=np.mean(seed_predictions,axis=0);fold_metrics=[metrics(confirm_truth[i*c['horizon']:(i+1)*c['horizon']],aggregate[i*c['horizon']:(i+1)*c['horizon']]) for i in range(len(validation))];m=metrics(confirm_truth,aggregate);std=float(np.std([v['mse'] for v in fold_metrics]));seed_mses=[metrics(confirm_truth,pred)['mse'] for pred in seed_predictions];skill=1-m['mse']/baseline_metric['mse'] if baseline_metric['mse']>0 else None;win_fraction=float(np.mean([a['mse']<b['mse'] for a,b in zip(fold_metrics,baseline_fold)]));score=m['mse']+c['stabilityPenalty']*std
   rows.append({'id':row['id'],'method':p['method'],'parameters':p,'config':row['config'],'modelWeight':weight,'validation':m,'foldMetrics':fold_metrics,'confirmationScaleAudit':confirm_audits,'seedMse':seed_mses,'seedMseStd':float(np.std(seed_mses)),'foldMseStd':std,'skillVsPersistence':skill,'winningFoldFraction':win_fraction,'selectionScore':score,'eligible':bool(weight>0 and skill is not None and skill>=c['minimumSkill'] and win_fraction>=.5)});stored_predictions[row['id']]=aggregate
  except Exception as e:failures.append({'stage':'final_validation','parameters':p,'error':str(e)})
 eligible=sorted([r for r in rows if r['eligible']],key=lambda r:(r['selectionScore'],r['validation']['mse'],r['id']))
 chosen=eligible[0] if eligible else {'id':'persistence','method':'naive','parameters':None,'config':None,'modelWeight':0.,'validation':baseline_metric,'skillVsPersistence':0.,'selectionScore':baseline_metric['mse'],'eligible':True}
 # Freeze the winner before reading TE labels. Seeds are averaged; no best seed is selected.
 test_predictions=[];future_predictions=[];final_training=[]
 for seed in (c['seeds'] if chosen['modelWeight']>0 else [c['seeds'][0]]):
  if chosen['modelWeight']>0:
   progress('test_winner',completed,plan['estimatedFitsUpperBound'],f"Ganador fijado: {chosen['method']} · semilla {seed}")
   fitted=predict(chosen['method'],selection,c['horizon'],5,seed,{'networks':{chosen['method']:chosen['config']}});raw=np.asarray(fitted['prediction']);fit_count+=1;completed+=1;final_training.append({'seed':seed,'trainingControl':fitted.get('trainingControl'),'scaling':fitted.get('network',{}).get('scaling'),'historyOffset':fitted.get('network',{}).get('historyOffset',0),'lastAvailableDate':frame.index[nselect-1].isoformat(),'testFirstDate':frame.index[nselect].isoformat()})
  else:raw=np.repeat(selection[-1],c['horizon'])
  test_predictions.append(chosen['modelWeight']*raw+(1-chosen['modelWeight'])*selection[-1])
 test_pred=np.mean(test_predictions,axis=0);test_truth=y[nselect:];test_metric=metrics(test_truth,test_pred);test_base=metrics(test_truth,np.repeat(selection[-1],c['horizon']))
 # Future refits may use all now-observed rows after the reserved evaluation is complete.
 future_error=None
 try:
  for seed in (c['seeds'] if chosen['modelWeight']>0 else [c['seeds'][0]]):
   if chosen['modelWeight']>0:
    fitted=predict(chosen['method'],y,c['horizon'],5,seed,{'networks':{chosen['method']:chosen['config']}});raw=np.asarray(fitted['prediction']);fit_count+=1;completed+=1
   else:raw=np.repeat(y[-1],c['horizon'])
   future_predictions.append(chosen['modelWeight']*raw+(1-chosen['modelWeight'])*y[-1])
 except Exception as e:future_error=str(e);future_predictions=[]
 import pandas as pd
 future_dates=pd.date_range(frame.index[-1]+pd.tseries.frequencies.to_offset(FREQ[ds['frequency']]),periods=c['horizon'],freq=FREQ[ds['frequency']]);forex=ds.get('name','').upper().endswith('=X') and column.lower() in ('open','high','low','close','adj close');pip=.01 if forex and 'JPY' in ds['name'].upper() else .0001 if forex else None
 progress('complete',completed,plan['estimatedFitsUpperBound'],f"Finalizado: {chosen['method']}")
 return {'schemaVersion':'1.0','dataset':{'name':ds.get('name'),'column':column,'records':len(y),'firstDate':frame.index[0].isoformat(),'lastDate':frame.index[-1].isoformat()},'plan':plan,'searchTrials':[t['row'] for t in ordered],'finalValidation':sorted(rows,key=lambda r:r['selectionScore']),'baselineValidation':baseline_metric,'selected':chosen,'allCandidatesFailed':not bool(trials),'fallbackToPersistence':not bool(eligible),'test':{'metrics':test_metric,'persistence':test_base,'skillVsPersistence':1-test_metric['mse']/test_base['mse'] if test_base['mse']>0 else None,'rmsePips':test_metric['rmse']/pip if pip else None,'dates':[d.isoformat() for d in frame.index[nselect:]],'actual':test_truth.tolist(),'prediction':test_pred.tolist()},'future':{'dates':[d.isoformat() for d in future_dates],'prediction':np.mean(future_predictions,axis=0).tolist() if future_predictions else [],'error':future_error},'pipSize':pip,'training':final_training,'failures':failures,'actualFits':fit_count,'elapsedSeconds':time.monotonic()-started,'policy':plan['policy'],'note':'La selección usa MSE medio + penalización × dispersión de MSE entre bloques de VA final. Se exige skill mínimo y ganar al menos la mitad de esos bloques. No se seleccionan semillas ni pesos usando TE. JSON contiene parámetros y predicciones; no contiene pesos Keras entrenados. Futuro reentrena con historial observado completo, respetando el tope de historial.'}
