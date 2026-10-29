"""Unidades 2–7: checkpoints causales, regularización y búsqueda interna."""
import copy
import numpy as np
import pandas as pd
from financial_lab import panel,scale_panel,split_summary,regression_metrics,classification_metrics,trade_path
from neural import architecture,optimizer,real
from features import bounded_int
from training_control import TrainingControl,checkpoint_callback


def model(tf,kind,shape,cfg,task='regression',variant=None):
    arch=architecture(cfg.get('architecture'));seed=cfg['seed'];reg=variant or {}
    penalty=tf.keras.regularizers.L1L2(l1=reg.get('l1',0),l2=reg.get('l2',0));x=tf.keras.Input(shape=shape);h=x
    hidden=arch['layers']
    if kind=='mlp':
        h=tf.keras.layers.Flatten()(h)
        widths=cfg.get('denseWidths',([arch['inputUnits']] if cfg.get('projectInput',True) else [])+[l['units'] for l in hidden])
        for i,width in enumerate(widths):
            activation=arch['inputActivation'] if i==0 else hidden[min(i-1,len(hidden)-1)]['activation']
            h=tf.keras.layers.Dense(width,kernel_initializer=tf.keras.initializers.HeNormal(seed+i),kernel_regularizer=penalty,name=f'dense_{i}')(h)
            if reg.get('batchNorm'):h=tf.keras.layers.BatchNormalization(name=f'bn_{i}')(h)
            h=tf.keras.layers.Activation(activation)(h)
            dropout=reg.get('dropout',hidden[min(i-1,len(hidden)-1)]['dropout'] if i>0 and variant is None else 0)
            if dropout:h=tf.keras.layers.Dropout(dropout,seed=seed+200+i)(h)
    else:
        if kind=='cnn' and cfg.get('cnnPadding','valid')=='valid' and shape[0]<1+sum(l['kernelSize']-1 for l in hidden):raise ValueError('Ventana CNN demasiado corta para kernels con padding valid; aumenta sesiones o usa causal/same.')
        if cfg.get('projectInput',False):h=tf.keras.layers.Dense(arch['inputUnits'],activation=arch['inputActivation'],name='projection')(h)
        for i,l in enumerate(hidden):
            if kind=='cnn':h=tf.keras.layers.Conv1D(l['units'],l['kernelSize'],padding=cfg.get('cnnPadding','valid'),activation=l['activation'],kernel_regularizer=penalty,kernel_initializer=tf.keras.initializers.HeNormal(seed+i),name=f'conv_{i}')(h)
            else:
                cls=tf.keras.layers.SimpleRNN if kind=='rnn' else tf.keras.layers.GRU if 'gru' in kind else tf.keras.layers.LSTM
                gates={} if kind=='rnn' else {'recurrent_activation':l['recurrentActivation']}
                recurrent=cls(l['units'],activation=l['activation'],**gates,return_sequences=i<len(hidden)-1,stateful=False,kernel_regularizer=penalty,kernel_initializer=tf.keras.initializers.GlorotUniform(seed+i),recurrent_initializer=tf.keras.initializers.Orthogonal(seed+100+i),name=f'recurrent_{i}')
                h=tf.keras.layers.Bidirectional(recurrent,merge_mode='concat',name=f'bidirectional_{i}')(h) if kind.startswith('bi') else recurrent(h)
            if l['dropout']:h=tf.keras.layers.Dropout(l['dropout'],seed=seed+300+i)(h)
        if kind=='cnn':h=tf.keras.layers.GlobalAveragePooling1D(name='global_average')(h)
    out=tf.keras.layers.Dense(1,activation='linear',kernel_initializer=tf.keras.initializers.GlorotUniform(seed+99),kernel_regularizer=penalty,name='output')(h)
    result=tf.keras.Model(x,out,name=kind)
    optcfg={**arch,'weightDecay':cfg.get('weightDecay',0),'gradientClip':cfg.get('gradientClip',1)}
    result.compile(optimizer=optimizer(tf,optcfg),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True) if task=='classification' else 'mse')
    return result


def fit(p,cfg,kind='mlp',task='regression',variant=None,train_mask=None,val_mask=None):
    import tensorflow as tf
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(cfg['seed'])
    tr=p['masks']['train'] if train_mask is None else train_mask;va=p['masks']['validation'] if val_mask is None else val_mask
    if tr.sum()<16 or va.sum()<8:raise ValueError('Cada ajuste necesita 16 muestras train y ocho de validación.')
    s=scale_panel(p,tr);target=(p['y']>0).astype(np.float32) if task=='classification' else s['y']
    m=model(tf,kind,s['X'].shape[1:],cfg,task,variant)
    control=TrainingControl({**cfg,'earlyStopping':(variant or {}).get('earlyStopping',cfg['earlyStopping'])})
    def loss(mask):
        pred=m(s['X'][mask],training=False).numpy().ravel()
        return float(np.mean(np.logaddexp(0,pred)-target[mask]*pred)) if task=='classification' else float(np.mean((target[mask]-pred)**2))
    callback=checkpoint_callback(tf,control,lambda:loss(tr),lambda:loss(va))
    m.fit(s['X'][tr],target[tr],epochs=cfg['epochs'],batch_size=cfg['batchSize'],shuffle=False,verbose=0,callbacks=[callback])
    m.set_weights(callback.best_weights)
    def predict(mask):
        output=m(s['X'][mask],training=False).numpy().ravel()
        return output if task=='classification' else output*s['targetScale']+s['targetMean']
    return {'model':m,'scale':s,'history':callback.history,'bestEpoch':control.best_epoch,'trainingControl':control.report(cfg['epochs'],'validation BCE' if task=='classification' else 'validation MSE normalizado'),'predict':predict,'parameters':m.count_params(),'architecture':[{'name':l.name,'type':type(l).__name__,'units':getattr(l,'units',getattr(l,'filters',None))} for l in m.layers]}


def evaluate(p,pred,mask,task,cfg):
    from scipy.special import expit
    truth=p['y'][mask];meta=[row for row,m in zip(p['meta'],mask) if m];metric=classification_metrics(truth,pred) if task=='classification' else regression_metrics(truth,pred)
    positions=(expit(pred)>=.5).astype(int) if task=='classification' else np.sign(pred)
    byasset=[]
    for asset in dict.fromkeys(row['assetId'] for row in meta):
        take=np.asarray([row['assetId']==asset for row in meta]);trade=trade_path([row for row,b in zip(meta,take) if b],positions[take],cfg['costBps'],overnight=False);byasset.append({'asset':next(row['asset'] for row in meta if row['assetId']==asset),**trade})
    rows=[{**row,'actual':float(y>0) if task=='classification' else float(y),'prediction':float(expit(value)) if task=='classification' else float(value)} for row,y,value in zip(meta,truth,pred)]
    return {'metrics':metric,'predictions':rows,'tradingByAsset':byasset}


def run(datasets,cfg,unit):
    task='classification' if unit==4 else cfg.get('task','regression')
    if task not in ('regression','classification'):raise ValueError('Tarea no admitida.')
    if unit in (5,6,7):task='regression'
    lookback=cfg['lookback'] if unit in (6,7) else 1;p=panel(datasets,cfg,lookback);validation=p['masks']['validation'];test=p['masks']['test'];comparison=[];fitted={};trials=[];failures=[]
    variants=[('MLP','mlp',cfg,None)]
    if unit==3:
        variants=[]
        for opt,default in [('sgd',.01),('rmsprop',.001),('adamw',.001)]:
            c=copy.deepcopy(cfg);c['architecture']['optimizer']=opt;c['architecture']['learningRate']=real(cfg.get('optimizerRates',{}).get(opt,default),1e-6,.1,'Learning rate');variants.append((opt.upper(),'mlp',c,None))
    if unit==4:cfg={**cfg,'earlyStopping':False};variants=[('Base','mlp',cfg,{}),('Dropout','mlp',cfg,{'dropout':cfg['dropout']}),('L1','mlp',cfg,{'l1':cfg['l1']}),('L2','mlp',cfg,{'l2':cfg['l2']}),('BatchNorm','mlp',cfg,{'batchNorm':True})]+([('EarlyStopping','mlp',cfg,{'earlyStopping':True})] if cfg['includeEarlyStopping'] else [])
    if unit==5:
        cfg,trials=hpo(p,cfg);variants=[('MLP HPO','mlp',cfg,None)]
    if unit==6:variants=[('CNN 1D','cnn',cfg,None),('MLP ventana','mlp',cfg,None)]
    if unit==7:
        kinds=cfg.get('recurrentModels',['lstm','gru','bilstm'])
        if not isinstance(kinds,list) or not kinds or any(k not in ('lstm','gru','bilstm','bigru') for k in kinds):raise ValueError('Selecciona LSTM/GRU unidireccional o bidireccional.')
        variants=[(kind.upper(),kind,cfg,None) for kind in dict.fromkeys(kinds)]
    for name,kind,c,variant in variants:
        try:
            result=fit(p,c,kind,task,variant);ev=evaluate(p,result['predict'](validation),validation,task,c)
            comparison.append({'model':name,'validation':ev['metrics'],'parameters':result['parameters'],'bestEpoch':result['bestEpoch'],'history':result['history'],'trainingControl':result['trainingControl'],'architecture':result['architecture'],'optimizer':c['architecture']['optimizer'],'learningRate':c['architecture']['learningRate']});fitted[name]=(result,c)
        except (ValueError,FloatingPointError) as exc:failures.append({'model':name,'error':str(exc)})
    if not fitted:raise ValueError('Ninguna red pudo evaluarse. '+ '; '.join(f"{f['model']}: {f['error']}" for f in failures))
    # Reference is fitted only on train and shares exactly the same input windows.
    from sklearn.linear_model import Ridge,LogisticRegression
    s=scale_panel(p);X=s['X'].reshape(len(s['X']),-1);tr=p['masks']['train'];ref=LogisticRegression(C=1/max(cfg['ridgeAlpha'],1e-6),random_state=cfg['seed'],max_iter=500).fit(X[tr],(p['y'][tr]>0).astype(int)) if task=='classification' else Ridge(alpha=cfg['ridgeAlpha']).fit(X[tr],s['y'][tr])
    refpredict=lambda mask:ref.decision_function(X[mask]) if task=='classification' else ref.predict(X[mask])*s['targetScale']+s['targetMean']
    refname='Logística referencia' if task=='classification' else 'Ridge referencia'
    comparison.append({'model':refname,'reference':True,'validation':evaluate(p,refpredict(validation),validation,task,cfg)['metrics'],'parameters':int(X.shape[1]+1),'bestEpoch':None,'history':[]})
    key='bce' if task=='classification' else 'rmse';comparison.sort(key=lambda row:row['validation'][key]);winner=next(row['model'] for row in comparison if not row.get('reference'))
    selected=fitted.get(winner);testpred=selected[0]['predict'](test) if selected else refpredict(test)
    test_result=evaluate(p,testpred,test,task,cfg)
    baseline=evaluate(p,refpredict(test),test,task,cfg)
    return {'unit':unit,'task':task,'winner':winner,'criterion':f'validation {key}','config':cfg,'features':cfg['selectedFeatures'],'splits':split_summary(p),'purgedSamples':p['purged'],'excludedSamples':p['excluded'],'comparison':comparison,'failures':failures,'test':test_result,'referenceTest':baseline,'trials':trials,'checkpointPolicy':'Pesos con menor pérdida de datos en validación; test consultado tras selección. Estado recurrente reiniciado por ventana.','units':'probabilidad de alza' if task=='classification' else 'bps log intradía siguiente','note':'Evaluación retrospectiva: señal al cierre t, entrada en apertura t+1 y salida al cierre t+1; sin exposición nocturna, costes de entrada y salida. Cada activo tiene su propia curva de capital; no se mezclan capitales de activos.'}


def hpo(p,cfg):
    rng=np.random.default_rng(cfg['seed']);n=bounded_int(cfg.get('trials',4),1,30,'Trials');widths=cfg.get('widths',[32,64,128]);depths=cfg.get('depths',[1,2,3]);decays=cfg.get('weightDecays',[0,1e-4,1e-3]);years=cfg.get('foldYears',[2022,2023]);low=real(cfg.get('lrLogMin',-4),-6,-1,'log10 lr mínimo');high=real(cfg.get('lrLogMax',-2.5),-6,-1,'log10 lr máximo')
    if low>high:raise ValueError('Rango log de learning rate invertido.')
    widths=[bounded_int(w,1,256,'Width') for w in widths];depths=[bounded_int(d,1,5,'Depth') for d in depths];decays=[real(d,0,.1,'Weight decay') for d in decays]
    if not widths or not depths or not decays or not isinstance(years,list) or not 1<=len(years)<=5:raise ValueError('Espacio de búsqueda o folds vacío/inválido.')
    folds=[]
    for year in years:
        year=bounded_int(year,1901,2199,'Año fold');start=pd.Timestamp(year=year,month=1,day=1);end=pd.Timestamp(year=year+1,month=1,day=1)
        if end>pd.Timestamp(cfg['validationStart']):raise ValueError('Todos los folds HPO deben quedar dentro de train, antes de validación externa.')
        train=np.asarray([pd.Timestamp(r['targetDate'])<start and r['split']=='train' for r in p['meta']]);valid=np.asarray([pd.Timestamp(r['signalDate'])>=start and pd.Timestamp(r['targetDate'])<end and r['split']=='train' for r in p['meta']]);folds.append((year,train,valid))
    trials=[]
    for trial in range(n):
        c=copy.deepcopy(cfg);width=int(rng.choice(widths));depth=int(rng.choice(depths));c['denseWidths']=[width]*depth;c['weightDecay']=float(rng.choice(decays));c['architecture']['optimizer']='adamw';c['architecture']['learningRate']=float(10**rng.uniform(low,high));scores=[]
        for year,tr,va in folds:
            r=fit(p,c,'mlp','regression',train_mask=tr,val_mask=va);score=regression_metrics(p['y'][va],r['predict'](va))['rmse'];scores.append({'year':year,'rmseBps':score,'trainSamples':int(tr.sum()),'validationSamples':int(va.sum()),'bestEpoch':r['bestEpoch'],'trainingControl':r['trainingControl'],'scalerMean':r['scale']['featureMean'].tolist()})
        trials.append({'trial':trial+1,'width':width,'depth':depth,'learningRate':c['architecture']['learningRate'],'weightDecay':c['weightDecay'],'folds':scores,'meanRmseBps':float(np.mean([s['rmseBps'] for s in scores]))})
    best=min(trials,key=lambda t:t['meanRmseBps']);final=copy.deepcopy(cfg);final['denseWidths']=[best['width']]*best['depth'];final['weightDecay']=best['weightDecay'];final['architecture']['optimizer']='adamw';final['architecture']['learningRate']=best['learningRate'];final['selectedTrial']=best['trial']
    return final,trials
