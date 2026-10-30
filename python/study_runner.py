"""Caso de uso lineal: datos → descripción → ajustes aislados → variantes → evaluación."""
import copy,json,sys,time
import numpy as np
from features import bounded_int
from domain.contracts import StudyContext
from domain.metrics import regression
from domain.decomposition import descriptive
from methods.registry import registry,catalog

def progress(stage,done,total,detail):print('LAB_PROGRESS '+json.dumps({'scope':'study','stage':stage,'completed':done,'total':total,'detail':detail},ensure_ascii=False),file=sys.stderr,flush=True)

def prepare(req):
    from engine import dataset_frame,horizons,PERIOD
    from training_control import settings
    ds=req.get('dataset');raw=req.get('config') or {}
    if not isinstance(ds,dict) or not isinstance(raw,dict):raise ValueError('Selecciona datos y configuración válidos.')
    frame=dataset_frame(ds);column=req.get('column')
    if column not in frame:raise ValueError('Variable no encontrada.')
    values=frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():raise ValueError('Serie incompleta o no finita.')
    methods=req.get('methods',['regression']);registered=registry()
    if not isinstance(methods,list) or not methods or any(k not in registered for k in methods):raise ValueError('Selecciona módulos válidos.')
    methods=list(dict.fromkeys(methods));folds=bounded_int(raw.get('folds',2),1,5,'Folds');period=bounded_int(raw.get('period',PERIOD[ds['frequency']]),1,365,'Periodo');seed=bounded_int(raw.get('seed',42),0,2**31-1,'Semilla');epochs=bounded_int(raw.get('epochs',60),1,5000,'Épocas');batch=bounded_int(raw.get('batchSize',32),8,256,'Batch');settings_={**settings(raw),'epochs':epochs,'batchSize':batch}
    if not isinstance(raw.get('allowPersistence',True),bool):raise ValueError('Referencia elegible debe ser booleano.')
    choices=horizons(ds,frame.index);requested=req.get('horizons',[0])
    if not isinstance(requested,list) or not requested or len(requested)>5:raise ValueError('Selecciona de uno a cinco horizontes.')
    selected=[];warnings=[]
    for value in dict.fromkeys(requested):
        i=bounded_int(value,0,len(choices)-1,'Horizonte');label,h=choices[i];first=len(values)-(folds+1)*h
        if first<max(32,period+2):raise ValueError(f'{label}: historial insuficiente para TR inicial, {folds} bloques VA y TE reservado. Reduce folds o elige otro horizonte.')
        selected.append({'index':i,'label':label,'steps':h,'initialTrainSamples':first,'validationSamples':folds*h,'testSamples':h,'testFirst':frame.index[-h].isoformat(),'testLast':frame.index[-1].isoformat()})
    for key in methods:
        spec=registered[key]
        if spec.dependency and not next(r for r in catalog() if r['key']==key)['available']:warnings.append(f'{spec.title}: falta dependencia {spec.dependency}; sus variantes reportarán error hasta instalar requisitos avanzados.')
    raw_params=raw.get('variantParameters',{})
    if not isinstance(raw_params,dict) or len(json.dumps(raw_params))>50000:raise ValueError('Parámetros de variantes inválidos o demasiado grandes.')
    for method,variants in raw_params.items():
        if method not in registered or not isinstance(variants,dict):raise ValueError('Parámetros: módulo desconocido o variantes inválidas.')
        for variant,parameters in variants.items():
            if variant not in ('1','2','3') or not isinstance(parameters,dict):raise ValueError('Parámetros: usa variantes 1, 2 y 3 con objetos JSON.')
    return frame,values,ds,column,methods,registered,{'folds':folds,'period':period,'seed':seed,'settings':settings_,'variantParameters':raw_params,'allowPersistence':raw.get('allowPersistence',True),'horizons':selected,'estimatedVariantFits':len(methods)*3*folds*len(selected),'warnings':warnings,'policy':'La descomposición completa es descriptiva. Cada variante se ajusta con TR y selección interna; VA externa clasifica; TE se evalúa una vez para el ganador de cada horizonte. El ajuste futuro usa solo observaciones ya disponibles.'}

def preview(req):return prepare(req)[-1]

def run(req):
    from methods.common import dates
    started=time.monotonic();frame,y,ds,column,methods,registered,plan=prepare(req);description=descriptive(y,plan['period']);rows=[];reports=[];failures=[];done=0;total=plan['estimatedVariantFits']
    def ctx(end,h,method,variant,seed):
        spec=registered[method];custom=plan['variantParameters'].get(method,{}).get(str(variant),{})
        if not isinstance(custom,dict):raise ValueError('Parámetros por variante deben ser objetos JSON.')
        params={**spec.variants[variant-1].defaults,**custom};return StudyContext(y[:end].copy(),frame.index[:end].copy(),frame.iloc[:end].copy(),column,h,plan['period'],seed,copy.deepcopy(plan['settings']),copy.deepcopy(params),ds['frequency'])
    for horizon in plan['horizons']:
        h=horizon['steps'];end_selection=len(y)-h;truth=[];baseline=[];folds=[]
        for k in range(plan['folds']):
            end=end_selection-(plan['folds']-k)*h;truth.extend(y[end:end+h]);baseline.extend(np.repeat(y[end-1],h));folds.append({'fold':k+1,'end':end,'trainFirst':frame.index[0].isoformat(),'trainLast':frame.index[end-1].isoformat(),'validationFirst':frame.index[end].isoformat(),'validationLast':frame.index[end+h-1].isoformat(),'samples':h})
        reference=regression(truth,baseline);comparisons=[]
        for method in methods:
            spec=registered[method]
            for variant in (1,2,3):
                actual=[];pred=[];dates_=[];fit_rows=[]
                try:
                    for fold in folds:
                        progress('variant',done,total,f"{spec.title} · V{variant} · {horizon['label']} · fold {fold['fold']}")
                        context=ctx(fold['end'],h,method,variant,plan['seed']+fold['fold']-1);result=spec.forecast(context,variant);p=np.asarray(result['prediction'],dtype=float);m=regression(y[fold['end']:fold['end']+h],p);fit_rows.append({**fold,'metrics':m,'metadata':result.get('metadata',{})});actual.extend(y[fold['end']:fold['end']+h]);pred.extend(p);dates_.extend(frame.index[fold['end']:fold['end']+h].astype(str));done+=1
                    metric=regression(actual,pred);row={'id':f'{method}:v{variant}','method':method,'methodTitle':spec.title,'variant':variant,'title':spec.variants[variant-1].title,'validation':metric,'skillVsPersistence':1-metric['mse']/reference['mse'] if reference['mse']>0 else None,'folds':fit_rows,'validationPredictions':{'dates':dates_,'actual':list(map(float,actual)),'prediction':list(map(float,pred))},'horizon':horizon['label']};comparisons.append(row);rows.append(row)
                except Exception as e:failures.append({'method':method,'variant':variant,'horizon':horizon['label'],'error':str(e)});progress('failed',done,total,str(e))
        comparisons.sort(key=lambda r:(r['validation']['mse'],r['validation']['mae'],r['id']));winner=comparisons[0] if comparisons else None
        if plan['allowPersistence'] and (winner is None or reference['mse']<=winner['validation']['mse']):winner={'id':'persistence','method':'naive','variant':0,'title':'Persistencia','validation':reference}
        if winner is None:reports.append({'horizon':horizon,'comparisons':[],'winner':None,'reference':reference});continue
        progress('test_winner',done,total,f"Ganador de VA: {winner['title']}");metadata={};future_error=None
        if winner['method']=='naive':test_pred=np.repeat(y[end_selection-1],h);future_pred=np.repeat(y[-1],h);future_dates=dates(StudyContext(y,frame.index,frame,column,h,plan['period'],plan['seed'],{}, {},ds['frequency']))
        else:
            context=ctx(end_selection,h,winner['method'],winner['variant'],plan['seed'])
            try:
                test_result=registered[winner['method']].forecast(context,winner['variant']);test_pred=np.asarray(test_result['prediction']);regression(y[-h:],test_pred)
            except Exception as e:
                failures.append({'method':winner['method'],'variant':winner['variant'],'horizon':horizon['label'],'stage':'test_winner','error':str(e)})
                reports.append({'horizon':horizon,'comparisons':comparisons,'winner':{'id':winner['id'],'method':winner['method'],'variant':winner['variant'],'title':winner['title'],'validation':winner['validation']},'reference':reference,'folds':folds,'testError':str(e)});continue
            metadata=test_result.get('metadata',{});future_pred=[];future_dates=dates(ctx(len(y),h,winner['method'],winner['variant'],plan['seed']))
            try:future_pred=registered[winner['method']].forecast(ctx(len(y),h,winner['method'],winner['variant'],plan['seed']),winner['variant'])['prediction']
            except Exception as e:future_error=str(e)
        test_metric=regression(y[-h:],test_pred);base_test=regression(y[-h:],np.repeat(y[end_selection-1],h));error=y[-h:]-test_pred;counts,edges=np.histogram(error,bins=min(12,h));volatility=None
        if metadata.get('variance') is not None and y[end_selection-1]>0 and np.all(y[-h:]>0):
            proxy=np.diff(np.log(np.r_[y[end_selection-1],y[-h:]]))**2;variance=np.maximum(metadata['variance'],1e-12);volatility={'qlike':float(np.mean(np.log(variance)+proxy/variance)),'varianceMse':float(np.mean((proxy-variance)**2)),'note':'Proxy = retorno diario al cuadrado, no varianza realizada intradía.'}
        reports.append({'horizon':horizon,'comparisons':comparisons,'winner':{'id':winner['id'],'method':winner['method'],'variant':winner['variant'],'title':winner['title'],'validation':winner['validation']},'reference':reference,'folds':folds,'test':{'dates':frame.index[-h:].astype(str).tolist(),'actual':y[-h:].tolist(),'prediction':list(map(float,test_pred)),'metrics':test_metric,'persistence':base_test,'metadata':metadata,'volatility':volatility},'future':{'dates':future_dates.astype(str).tolist(),'prediction':list(map(float,future_pred)),'error':future_error},'errorHistogram':[{'bin':float((edges[i]+edges[i+1])/2),'count':int(counts[i])} for i in range(len(counts))]})
    progress('complete',done,total,'Estudios completados')
    return {'schemaVersion':'1.0','dataset':{'name':ds['name'],'column':column,'kind':ds['kind'],'frequency':ds['frequency']},'catalog':[r for r in catalog() if r['key'] in methods],'plan':plan,'decomposition':{'dates':frame.index.astype(str).tolist(),'actual':y.tolist(),**description},'reports':reports,'failures':failures,'elapsedSeconds':time.monotonic()-started,'policy':plan['policy'],'note':'Variantes se ejecutan secuencialmente. No se garantiza mejora monotónica V1→V2→V3 ni R² cercano a uno. MSE/RMSE/MAE se calculan en unidad original (MAE corresponde al error absoluto, no se denomina MSA). Cada horizonte es una tarea distinta; TE ya usado en otros estudios no es confirmación independiente.'}
