"""Banco de variantes: búsqueda en validación, test solo para ganadores por familia."""
import copy,json,sys,gc,math,time
import numpy as np
from features import bounded_int
from neural import architecture,real,optimizer
from financial_lab import config,panel,scale_panel,regression_metrics,classification_metrics,split_summary,trade_path
from unit_supervised import fit
from training_control import TrainingControl,checkpoint_callback
from model_exports import export_bundle

PREDICTIVE=['ridge','mlp','cnn','rnn','lstm','gru','bilstm','bigru','ae_ridge','dae_ridge','pca_ridge']
GENERATION=['cvae','cgan','cdcgan']
TRADING=['dqn','ddqn']

def progress(value):
    print('LAB_PROGRESS '+json.dumps(value,ensure_ascii=False,allow_nan=False),file=sys.stderr,flush=True)

def candidate(base,rng,search,family):
    c=copy.deepcopy(base);width=int(rng.choice(search['widths']));depth=int(rng.choice(search['depths']));lr=float(10**rng.uniform(search['lrLogMin'],search['lrLogMax']));opt=str(rng.choice(search['optimizers']));dropout=float(rng.choice(search['dropouts']));activation='tanh' if family in ('rnn','lstm','gru','bilstm','bigru') else 'relu'
    a=architecture({'inputUnits':width,'inputActivation':activation,'layers':[{'units':width,'activation':activation,'dropout':dropout,'kernelSize':3} for _ in range(depth)],'optimizer':opt,'learningRate':lr})
    c.update(architecture=a,encoder=a,decoder=a,generator=a,discriminator=copy.deepcopy(a),projectInput=family=='mlp',denseWidths=[width]*depth,lookback=int(rng.choice(search['lookbacks'])),latentDim=int(rng.choice(search['latentDims'])),cnnPadding='causal',generativeModels=[family] if family in GENERATION else ['cvae','cgan'])
    c['discriminator']['optimizer']=str(rng.choice(search['optimizers']));c['discriminator']['learningRate']=float(10**rng.uniform(search['lrLogMin'],search['lrLogMax']))
    return c

def normalize_search(raw,base):
    def ints(key,default,lo,hi):
        values=raw.get(key,default)
        if not isinstance(values,list) or not 1<=len(values)<=20:raise ValueError(f'{key}: lista de 1–20 valores.')
        return [bounded_int(v,lo,hi,key) for v in values]
    opts=raw.get('optimizers',['adam','rmsprop','adamw'])
    if not isinstance(opts,list) or not opts or any(o not in ('adam','adamw','rmsprop','sgd','adamax','nadam','adagrad') for o in opts):raise ValueError('Optimizadores del banco inválidos.')
    drops=raw.get('dropouts',[0,.1]);low=real(raw.get('lrLogMin',-4),-6,-1,'log lr mínimo');high=real(raw.get('lrLogMax',-2.5),-6,-1,'log lr máximo')
    if low>high:raise ValueError('Rango de learning rate invertido.')
    return {'widths':ints('widths',[32,64],1,256),'depths':ints('depths',[1,2],1,5),'lookbacks':ints('lookbacks',[base['lookback']],1,120),'latentDims':ints('latentDims',[2,3],1,len(base['selectedFeatures'])),'optimizers':opts,'dropouts':[real(v,0,.5,'Dropout') for v in drops],'lrLogMin':low,'lrLogMax':high}


def latent_fit(p,cfg,family):
    import tensorflow as tf
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    s=scale_panel(p);X=s['X'][:,-1,:];tr=p['masks']['train'];va=p['masks']['validation'];latent=min(cfg['latentDim'],X.shape[1]);models={};extras={};control_info=None
    if family=='pca_ridge':
        pca=PCA(n_components=latent,random_state=cfg['seed']).fit(X[tr]);encode=pca.transform;extras['pca.json']={'mean':pca.mean_.tolist(),'components':pca.components_.tolist(),'explainedVarianceRatio':pca.explained_variance_ratio_.tolist()};params=int(pca.components_.size)
    else:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(cfg['seed']);a=cfg['architecture']
        def stack(n,out,name):
            inp=tf.keras.Input((n,));h=inp
            for l in a['layers']:
                h=tf.keras.layers.Dense(l['units'],activation=l['activation'])(h)
                if l['dropout']:h=tf.keras.layers.Dropout(l['dropout'])(h)
            return tf.keras.Model(inp,tf.keras.layers.Dense(out,activation='linear')(h),name=name)
        enc=stack(X.shape[1],latent,'encoder');dec=stack(latent,X.shape[1],'decoder');inp=tf.keras.Input((X.shape[1],));h=tf.keras.layers.GaussianNoise(cfg.get('noiseStd',.1),seed=cfg['seed'])(inp) if family=='dae_ridge' else inp;ae=tf.keras.Model(inp,dec(enc(h)));ae.compile(optimizer=optimizer(tf,a),loss='mse');control=TrainingControl(cfg)
        score=lambda mask:float(np.mean((ae(X[mask],training=False).numpy()-X[mask])**2));callback=checkpoint_callback(tf,control,lambda:score(tr),lambda:score(va));ae.fit(X[tr],X[tr],epochs=cfg['epochs'],batch_size=cfg['batchSize'],shuffle=False,verbose=0,callbacks=[callback]);ae.set_weights(callback.best_weights);encode=lambda values:enc(values,training=False).numpy();models={'encoder':enc,'decoder':dec};params=ae.count_params();control_info=control.report(cfg['epochs'],'reconstrucción de validación del extractor')
    ridge=Ridge(alpha=cfg['ridgeAlpha']).fit(encode(X[tr]),s['y'][tr]);extras['ridge.json']={'coef':ridge.coef_.tolist(),'intercept':float(ridge.intercept_)}
    predict=lambda mask:ridge.predict(encode(X[mask]))*s['targetScale']+s['targetMean']
    return {'predict':predict,'scale':s,'models':models,'extras':extras,'parameters':params,'trainingControl':control_info,'kind':'latent','lookback':1}


def test_trading(model,X,meta,cfg):
    from unit_trading import TradingEnv
    e=TradingEnv(X,meta,cfg['costBps'],cfg.get('liquidateEnd',True));state=e.reset();positions=[]
    while True:
        action=int(np.argmax(model(state[None],training=False).numpy()[0]));state,_,done,info=e.step(action);positions.append(info['position'])
        if done:break
    return trade_path(meta,positions,cfg['costBps'],cfg.get('liquidateEnd',True))


def run(req):
    started=time.monotonic();datasets=req.get('datasets',[]);mode=req.get('mode','predictive');allowed=PREDICTIVE if mode=='predictive' else ['logistic','mlp','cnn','rnn','lstm','gru','bilstm','bigru'] if mode=='classification' else GENERATION if mode=='generation' else TRADING if mode=='trading' else []
    families=req.get('families',['mlp','lstm'] if mode=='predictive' else allowed);trials=bounded_int(req.get('trialsPerFamily',3),1,12,'Variantes por familia')
    if not isinstance(families,list) or not families or any(f not in allowed for f in families):raise ValueError('Familias no válidas para la tarea elegida.')
    families=list(dict.fromkeys(families))
    if not isinstance(datasets,list) or not 1<=len(datasets)<=10:raise ValueError('Selecciona de uno a diez activos OHLC.')
    if mode=='trading' and len(datasets)!=1:raise ValueError('El banco de trading admite un activo por ejecución.')
    base=config(req.get('config'))
    if mode in ('predictive','classification'):base['task']='classification' if mode=='classification' else 'regression'
    search=normalize_search(req.get('search') or {},base);rng=np.random.default_rng(base['seed']);history=[];best={};failures=[];total=len(families)*trials;done=0
    source_panel=panel(datasets,base,max(search['lookbacks'])) if mode in ('predictive','classification') else None
    asset_dates={str(d['id']):[r['date'] for r in d['rows']] for d in datasets}
    for family in families:
        for trial in range(1,trials+1):
            cfg=candidate(base,rng,search,family);variant=f'{family}-{trial}';progress({'stage':'training','family':family,'trial':trial,'completed':done,'total':total})
            try:
                if mode in ('predictive','classification'):
                    p={**source_panel,'X':source_panel['X'][:,-cfg['lookback']:],'meta':[{**m,'inputStartIndex':m['inputStartIndex']+source_panel['X'].shape[1]-cfg['lookback'],'inputStartDate':asset_dates[m['assetId']][m['inputStartIndex']+source_panel['X'].shape[1]-cfg['lookback']]} for m in source_panel['meta']],'scalerConfig':{'featureScaler':cfg.get('featureScaler','standard'),'targetScaler':cfg.get('targetScaler','standard')}}
                    if family=='logistic':
                        from sklearn.linear_model import LogisticRegression
                        s=scale_panel(p);X=s['X'].reshape(len(s['X']),-1);train=p['masks']['train'];linear=LogisticRegression(C=1/max(cfg['ridgeAlpha'],1e-6),random_state=cfg['seed'],max_iter=1000).fit(X[train],(p['y'][train]>0).astype(int));fitted={'predict':lambda mask,m=linear,z=X:m.decision_function(z[mask]),'scale':s,'models':{},'extras':{'logistic.json':{'coef':linear.coef_.ravel().tolist(),'intercept':float(linear.intercept_[0])}},'parameters':int(X.shape[1]+1),'trainingControl':None,'kind':'logistic','lookback':cfg['lookback']}
                    elif family=='ridge':
                        from sklearn.linear_model import Ridge
                        s=scale_panel(p);X=s['X'].reshape(len(s['X']),-1);train=p['masks']['train'];cfg['ridgeAlpha']=float(rng.choice([.1,1,10]));ridge=Ridge(alpha=cfg['ridgeAlpha']).fit(X[train],s['y'][train]);fitted={'predict':lambda mask,m=ridge,z=X,scale=s:m.predict(z[mask])*scale['targetScale']+scale['targetMean'],'scale':s,'models':{},'extras':{'ridge.json':{'coef':ridge.coef_.tolist(),'intercept':float(ridge.intercept_)}},'parameters':int(X.shape[1]+1),'trainingControl':None,'kind':'ridge','lookback':cfg['lookback']}
                    elif family.endswith('_ridge'):fitted=latent_fit(p,cfg,family)
                    else:
                        result=fit(p,cfg,family,task='classification' if mode=='classification' else 'regression',variant={'dropout':cfg['architecture']['layers'][0]['dropout']});fitted={'predict':result['predict'],'scale':result['scale'],'models':{'predictor':result['model']},'extras':{},'parameters':result['parameters'],'trainingControl':result['trainingControl'],'kind':'classifier' if mode=='classification' else 'predictive','lookback':cfg['lookback']}
                    validation=(classification_metrics if mode=='classification' else regression_metrics)(p['y'][p['masks']['validation']],fitted['predict'](p['masks']['validation']));score=validation['bce' if mode=='classification' else 'mse'];fitted['panel']=p
                elif mode=='generation':
                    from unit_generative import run as generate
                    result=generate(datasets,cfg,retain_models=True,evaluate_test=False);row=result['comparison'][0];validation=row['validation'];score=validation['distributionScore'];fitted={'kind':'generation','models':{'generator':result['_generator'],'encoder' if family=='cvae' else 'discriminator':result['_auxiliary']},'prepared':result['_prepared'],'parameters':row['parameters'],'trainingControl':row.get('trainingControl'),'lookback':1,'extras':{}}
                else:
                    from unit_trading import run as trade
                    result=trade(datasets,cfg,retain_models=True,evaluate_test=False,algorithm=family);row=result['comparison'][0];validation=row['validation'];score=-validation['netReturn'];fitted={'kind':'trading','models':{'online':result['_online'],'target':result['_target']},'panel':result['_panel'],'scale':result['_scale'],'parameters':row['parameters'],'trainingControl':None,'lookback':1,'extras':{}}
                if not math.isfinite(score):raise ValueError('Criterio de validación no finito.')
                record={'id':variant,'family':family,'trial':trial,'config':cfg,'validation':validation,'selectionScore':float(score),'parameters':fitted['parameters'],'trainingControl':fitted['trainingControl']};history.append(record)
                if family not in best or (score,fitted['parameters'])<(best[family]['record']['selectionScore'],best[family]['record']['parameters']):best[family]={'record':record,'fitted':fitted}
                progress({'stage':'validated','family':family,'trial':trial,'completed':done+1,'total':total,'validation':validation,'bestVariant':best[family]['record']['id']})
            except Exception as exc:failures.append({'id':variant,'family':family,'error':str(exc)});progress({'stage':'failed','family':family,'trial':trial,'completed':done+1,'total':total,'error':str(exc)})
            done+=1;gc.collect()
    if not best:raise ValueError('Ninguna variante pudo evaluarse. '+'; '.join(f['error'] for f in failures))
    winners=[];root=req.get('artifactRoot')
    for family,b in best.items():
        record=b['record'];f=b['fitted'];cfg=record['config'];progress({'stage':'test_winner','family':family,'completed':total,'total':total})
        if mode in ('predictive','classification'):
            p=f['panel'];te=p['masks']['test'];pred=f['predict'](te);test=(classification_metrics if mode=='classification' else regression_metrics)(p['y'][te],pred);s=f['scale'];scalers={key:np.asarray(s[key]).tolist() for key in ('featureMean','featureScale','targetMean','targetScale')}
        elif mode=='generation':
            from unit_generative import distribution,regime_evaluation
            p=f['prepared'];te=p['masks']['test'];noise=np.random.default_rng(base['seed']+999).normal(size=(int(te.sum()),cfg.get('latentDim',4))).astype(np.float32);samples=f['models']['generator'](np.concatenate([noise,p['c'][te]],axis=1),training=False).numpy()*p['std']+p['mean'];test=distribution(p['raw'][te],samples,float(np.quantile(p['raw'][p['masks']['train']],.05)));test['byRegime']=regime_evaluation(p,te,samples);scalers={'returnMean':p['mean'],'returnScale':p['std'],'conditionMean':p['conditionMean'],'conditionScale':p['conditionScale'],'conditionSchema':p['conditionSchema']}
        else:
            p=f['panel'];te=p['masks']['test'];test=test_trading(f['models']['online'],f['scale']['X'][te,0,:],[r for r,m in zip(p['meta'],te) if m],cfg)['metrics'];s=f['scale'];scalers={key:np.asarray(s[key]).tolist() for key in ('featureMean','featureScale','targetMean','targetScale')}
        audit=None
        if mode in ('predictive','classification'):
            from experiment_diagnostics import prediction_audit
            audit=prediction_audit(p,cfg,f['predict'],f['scale'],f['parameters'],'regression' if mode=='predictive' else 'classification',datasets)
        winner={**record,'diagnostics':audit,'test':test,'scalers':scalers,'kind':f['kind'],'lookback':f['lookback'],'features':base['selectedFeatures']}
        if root:
            winner['artifact']=export_bundle(root,family,f['kind'],cfg,datasets,scalers,f['models'],f['extras'],record['validation'],test,f['lookback'])
        winners.append(winner)
    winners.sort(key=lambda w:(w['selectionScore'],w['parameters']));progress({'stage':'complete','family':winners[0]['family'],'completed':total,'total':total});splits=split_summary(source_panel) if source_panel is not None else None
    return {'elapsedSeconds':time.monotonic()-started,'mode':mode,'criterion':'MSE menor; R² mayor sobre las mismas etiquetas de validación' if mode=='predictive' else 'BCE menor y AUC reportado en validación' if mode=='classification' else 'Wasserstein + diferencia std + distancia temporal ACF en validación' if mode=='generation' else 'retorno neto de validación mayor','units':'bps log intradía siguiente' if mode=='predictive' else 'probabilidad de alza intradía' if mode=='classification' else 'retornos generados' if mode=='generation' else 'retorno neto de trading','bestFamily':winners[0]['family'],'winners':winners,'variants':history,'failures':failures,'search':search,'config':base,'splits':splits,'testPolicy':'Test evaluado solo para la variante ganadora de cada familia, después de toda la búsqueda. No se usa para elegir variantes.','note':'Con las mismas etiquetas, minimizar MSE equivale a maximizar R² si su varianza no es cero. Las métricas entre tareas diferentes no se mezclan.'}
