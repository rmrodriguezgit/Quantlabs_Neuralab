"""U9: ventanas completas de retornos, CVAE y cGAN condicionados por log-vol."""
import numpy as np
import pandas as pd
from scipy import stats
from features import bounded_int
from neural import architecture,optimizer,real
from financial_lab import ohlc_frame,partition
from training_control import TrainingControl
from generative_conditions import observation,fit_schema,transform,inference_condition,options
from generative_networks import build


def windows(datasets,cfg):
    options(cfg)
    if cfg.get('scenarioAsset') and cfg['scenarioAsset'] not in [d['id'] for d in datasets]:raise ValueError('El activo de escenarios debe estar seleccionado en datos.')
    real(cfg.get('temporalWeight',1),0,10,'Peso temporal')
    real(cfg.get('ganNegativeSlope',.01),0,1,'Pendiente LeakyReLU')
    bounded_int(cfg.get('conditionEmbeddingDim',4),1,32,'Embedding')
    if cfg.get('scenarioRegime','observed') not in ('observed','bear','neutral','bull'):raise ValueError('Régimen de escenario inválido.')
    if cfg.get('scenarioRegime','observed')!='observed' and 'regime' not in cfg.get('conditionFeatures',['volatility']):raise ValueError('Activa la condición régimen para fijar un régimen hipotético.')
    if cfg.get('conditionFusion','concat') not in ('concat','multiply'):raise ValueError('Fusión de condición inválida.')
    if cfg.get('ganUpsampling','transpose') not in ('transpose','upsample'):raise ValueError('Expansión temporal inválida.')
    horizon=bounded_int(cfg.get('horizon',16),2,64,'Horizonte generativo');cond_period=bounded_int(cfg.get('conditionPeriod',20),2,120,'Periodo condición');rows=[];conditions=[];meta=[];purged=0;excluded=0
    for asset,ds in enumerate(datasets):
        f=ohlc_frame(ds);returns=np.log(f.Close/f.Close.shift(1)).to_numpy()
        for start in range(cond_period+1,len(f)-horizon+1):
            end=start+horizon-1
            if f.index[start]<pd.Timestamp(cfg['trainStart']) or (cfg.get('testEnd') and f.index[end]>=pd.Timestamp(cfg['testEnd'])):excluded+=1;continue
            split=partition(f.index[start],f.index[end],cfg)
            if split is None:purged+=1;continue
            rows.append(returns[start:end+1]);conditions.append(observation(f,start,cfg,asset));meta.append({'asset':ds['name'],'start':f.index[start].isoformat(),'end':f.index[end].isoformat(),'split':split,'assetIndex':asset,'regime':int(observation(f,start,{**cfg,'conditionFeatures':['regime']},asset)[0])})
    if not rows:raise ValueError('No hay ventanas generativas completas.')
    X=np.asarray(rows,dtype=np.float32);c=np.asarray(conditions,dtype=np.float32);masks={part:np.asarray([r['split']==part for r in meta]) for part in ('train','validation','test')}
    for part,mask in masks.items():
        if mask.sum()<8:raise ValueError(f'{part}: menos de ocho ventanas completas. Amplía historial o reduce horizonte.')
    tr=masks['train'];mean=float(X[tr].mean());std=max(float(X[tr].std()),1e-8);schema=fit_schema(c,tr,cfg,datasets);cm=schema['mean'][0];cs=schema['scale'][0]
    return {'X':(X-mean)/std,'c':transform(c,schema),'rawConditions':c,'conditionSchema':schema,'raw':X,'masks':masks,'meta':meta,'mean':mean,'std':std,'conditionMean':cm,'conditionScale':cs,'purged':purged,'excluded':excluded,'horizon':horizon,'conditionPeriod':cond_period}


def distribution(real_windows,fake_windows,threshold=None):
    a=np.asarray(real_windows);b=np.asarray(fake_windows);af=a.ravel();bf=b.ravel();threshold=float(np.quantile(af,.05)) if threshold is None else threshold
    def summarize(x):
        flat=x.ravel();q=float(np.quantile(flat,.05));pairs=x[:,:-1].ravel();following=x[:,1:].ravel();mask=pairs<threshold;tail=float(np.mean(following[mask]<threshold)) if mask.any() else None
        return {'mean':float(flat.mean()),'std':float(flat.std()),'skew':float(stats.skew(flat)) if flat.std()>1e-12 else None,'kurtosis':float(stats.kurtosis(flat)) if flat.std()>1e-12 else None,'var05':q,'es05':float(flat[flat<=q].mean()),'lag1':float(np.corrcoef(pairs,following)[0,1]) if pairs.std()>1e-12 and following.std()>1e-12 else None,'tailDependence05':tail}
    def temporal(x):
        values={}
        for lag in (1,5):
            for squared in (False,True):
                y=x*x if squared else x
                key=('squaredReturnAcf' if squared else 'returnAcf')+str(lag)
                if x.shape[1]<=lag:values[key]=None;continue
                left=y[:,:-lag].ravel();right=y[:,lag:].ravel()
                values[key]=float(np.corrcoef(left,right)[0,1]) if left.std()>1e-12 and right.std()>1e-12 else None
        return values
    ar=summarize(a);br=summarize(b);ar.update(temporal(a));br.update(temporal(b))
    diversity={'meanStdAcrossScenarios':float(np.mean(np.std(b,axis=0))),'uniqueFraction':float(len(np.unique(np.round(b,8),axis=0))/len(b)),'lowDiversity':bool(np.mean(np.std(b,axis=0))<1e-8)}
    return {'diversity':diversity,'real':ar,'synthetic':br,'wasserstein':float(stats.wasserstein_distance(af,bf)),'ks':float(stats.ks_2samp(af,bf).statistic),'tailThreshold':threshold}


def temporal_distance(a,b):
    errors=[]
    for lag in (1,5):
        if a.shape[1]<=lag:continue
        for squared in (False,True):
            def acf(x):
                y=x*x if squared else x;left=y[:,:-lag].ravel();right=y[:,lag:].ravel()
                return float(np.corrcoef(left,right)[0,1]) if left.std()>1e-12 and right.std()>1e-12 else 0.
            errors.append(abs(acf(a)-acf(b)))
    return float(np.mean(errors)) if errors else 0.


def selection_score(real_windows,fake_windows,temporal_weight=1):
    weight=real(temporal_weight,0,10,'Peso temporal')
    if not np.isfinite(fake_windows).all():return float('inf')
    return float(stats.wasserstein_distance(real_windows.ravel(),fake_windows.ravel())+abs(real_windows.std()-fake_windows.std())+weight*temporal_distance(real_windows,fake_windows))


def regime_evaluation(p,mask,generated):
    regimes=np.asarray([m['regime'] for m in p['meta']])[mask];rows=[]
    for k,label in enumerate(['bajista','lateral','alcista']):
        selected=regimes==k;n=int(selected.sum())
        rows.append({'regime':label,'samples':n,'metrics':distribution(p['raw'][mask][selected],generated[selected]) if n>=8 else None,'note':None if n>=8 else 'Menos de ocho ventanas; diagnóstico no disponible.'})
    return rows


def run(datasets,cfg,retain_models=False,evaluate_test=True):
    import tensorflow as tf
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);p=windows(datasets,cfg);latent=bounded_int(cfg.get('latentDim',4),1,64,'Latente');beta=real(cfg.get('beta',.01),0,10,'Beta KL');samples=bounded_int(cfg.get('scenarioCount',64),8,512,'Escenarios');enc_cfg=architecture(cfg.get('encoder',cfg['architecture']));dec_cfg=architecture(cfg.get('decoder',cfg['architecture']));gen_cfg=architecture(cfg.get('generator',cfg['architecture']));disc_cfg=architecture(cfg.get('discriminator',cfg['architecture']));models=cfg.get('generativeModels',['cvae','cgan'])
    if not isinstance(models,list) or not models or any(m not in ('cvae','cgan','cdcgan') for m in models):raise ValueError('Selecciona CVAE, cGAN o cDCGAN.')
    tr=p['masks']['train'];va=p['masks']['validation'];te=p['masks']['test'];X=p['X'].astype(np.float32);C=p['c'].astype(np.float32);fixed_noise=rng.normal(size=(int(va.sum()),latent)).astype(np.float32);test_noise=rng.normal(size=(int(te.sum()),latent)).astype(np.float32);results=[];stored={};stored_aux={}
    def dense_net(shape,arch,outputs,name):
        inp=tf.keras.Input((shape,));h=inp
        if cfg.get('projectInput',False):h=tf.keras.layers.Dense(arch['inputUnits'],activation=arch['inputActivation'])(h)
        for l in arch['layers']:
            h=tf.keras.layers.Dense(l['units'],activation=l['activation'])(h)
            if l['dropout']:h=tf.keras.layers.Dropout(l['dropout'])(h)
        out=tf.keras.layers.Dense(outputs,activation='linear')(h);return tf.keras.Model(inp,out,name=name)
    for method in dict.fromkeys(models):
        tf.keras.utils.set_random_seed(cfg['seed']);history=[];control=TrainingControl(cfg);best_weights=None
        if method=='cvae':
            encoder=dense_net(p['horizon']+C.shape[1],enc_cfg,2*latent,'encoder');generator=build(tf,dec_cfg,cfg,p['conditionSchema'],latent,p['horizon'],'generator');opt=optimizer(tf,{**enc_cfg,'gradientClip':cfg['gradientClip']})
        else:
            generator=build(tf,gen_cfg,cfg,p['conditionSchema'],latent,p['horizon'],'generator',method=='cdcgan');discriminator=build(tf,disc_cfg,cfg,p['conditionSchema'],latent,p['horizon'],'discriminator',method=='cdcgan');go=optimizer(tf,{**gen_cfg,'gradientClip':cfg['gradientClip']});do=optimizer(tf,{**disc_cfg,'gradientClip':cfg['gradientClip']});bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for epoch in range(cfg['epochs']):
            epoch_losses=[];order=rng.permutation(int(tr.sum()));trainX=X[tr][order];trainC=C[tr][order]
            for start in range(0,len(trainX),cfg['batchSize']):
                x=tf.convert_to_tensor(trainX[start:start+cfg['batchSize']]);c=tf.convert_to_tensor(trainC[start:start+cfg['batchSize']]);size=len(x)
                if method=='cvae':
                    with tf.GradientTape() as tape:
                        encoded=encoder(tf.concat([x,c],axis=1),training=True);mu,lv=tf.split(encoded,2,axis=1);lv=tf.clip_by_value(lv,-8.,8.);z=mu+tf.exp(.5*lv)*tf.random.normal(tf.shape(mu));pred=generator(tf.concat([z,c],axis=1),training=True);mse=tf.reduce_mean(tf.square(x-pred));kl=.5*tf.reduce_mean(tf.reduce_sum(tf.square(mu)+tf.exp(lv)-1-lv,axis=1));loss=mse+beta*kl
                    variables=encoder.trainable_variables+generator.trainable_variables;opt.apply_gradients(zip(tape.gradient(loss,variables),variables));epoch_losses.append([float(loss),float(mse),float(kl)])
                else:
                    with tf.GradientTape() as tape:
                        fake=tf.stop_gradient(generator(tf.concat([tf.random.normal((size,latent)),c],axis=1),training=False));real_score=discriminator(tf.concat([x,c],axis=1),training=True);fake_score=discriminator(tf.concat([fake,c],axis=1),training=True);dl=.5*(bce(tf.ones_like(real_score),real_score)+bce(tf.zeros_like(fake_score),fake_score))
                    do.apply_gradients(zip(tape.gradient(dl,discriminator.trainable_variables),discriminator.trainable_variables))
                    with tf.GradientTape() as tape:
                        fake=generator(tf.concat([tf.random.normal((size,latent)),c],axis=1),training=True);score=discriminator(tf.concat([fake,c],axis=1),training=False);gl=bce(tf.ones_like(score),score)
                    go.apply_gradients(zip(tape.gradient(gl,generator.trainable_variables),generator.trainable_variables));epoch_losses.append([float(gl),float(dl),0.])
            fake=generator(np.concatenate([fixed_noise,C[va]],axis=1),training=False).numpy();score=selection_score(X[va],fake,cfg.get('temporalWeight',1));losses=np.mean(epoch_losses,axis=0)
            if not np.isfinite([score,*losses]).all():raise ValueError('Entrenamiento generativo divergente.')
            history.append({'epoch':epoch+1,'train':float(losses[0]),'auxiliary':float(losses[1]),'kl':float(losses[2]),'validation':score})
            improved,stop=control.observe(epoch+1,score)
            if improved:best_weights=([w.copy() for w in generator.get_weights()],[w.copy() for w in (encoder if method=='cvae' else discriminator).get_weights()])
            if stop:break
        generator.set_weights(best_weights[0]);(encoder if method=='cvae' else discriminator).set_weights(best_weights[1]);validation_fake=generator(np.concatenate([fixed_noise,C[va]],axis=1),training=False).numpy()*p['std']+p['mean'];results.append({'model':method.upper(),'validation':{'temporalDistance':temporal_distance(X[va],(validation_fake-p['mean'])/p['std']),'distributionScore':control.best,'wasserstein':distribution(X[va]*p['std']+p['mean'],validation_fake)['wasserstein']},'parameters':generator.count_params()+(encoder.count_params() if method=='cvae' else discriminator.count_params()),'bestEpoch':control.best_epoch,'history':history,'trainingControl':control.report(cfg['epochs'],'validation Wasserstein + std + ACF, ruido fijo')});stored[method.upper()]=generator;stored_aux[method.upper()]=encoder if method=='cvae' else discriminator
    results.sort(key=lambda r:r['validation']['distributionScore']);winner=results[0]['model'];G=stored[winner]
    if not evaluate_test:return {'unit':9,'winner':winner,'comparison':results,'config':cfg,'_generator':G,'_auxiliary':stored_aux[winner],'_prepared':p}
    test_fake=G(np.concatenate([test_noise,C[te]],axis=1),training=False).numpy()*p['std']+p['mean'];evaluation=distribution(p['raw'][te],test_fake,float(np.quantile(p['raw'][tr],.05)));evaluation['byRegime']=regime_evaluation(p,te,test_fake)
    # Conditional scenarios for the last selected asset, using only its observed past.
    scenario_asset=next((i for i,d in enumerate(datasets) if d['id']==cfg.get('scenarioAsset')),len(datasets)-1)
    last=ohlc_frame(datasets[scenario_asset]);returns=np.log(last.Close/last.Close.shift(1)).dropna().to_numpy();condition=np.log(max(float(np.std(returns[-p['conditionPeriod']:],ddof=1)),1e-8));condition_scaled,condition_raw=inference_condition(last,cfg,p['conditionSchema'],scenario_asset,{'observed':None,'bear':0,'neutral':1,'bull':2}.get(cfg.get('scenarioRegime','observed')));noise=rng.normal(size=(samples,latent)).astype(np.float32);scenarios=G(np.concatenate([noise,np.repeat(condition_scaled[None],samples,axis=0)],axis=1),training=False).numpy()*p['std']+p['mean'];counts,edges=np.histogram(np.r_[p['raw'][te].ravel(),test_fake.ravel()],bins=24);real_counts,_=np.histogram(p['raw'][te],bins=edges,density=True);fake_counts,_=np.histogram(test_fake,bins=edges,density=True)
    comparisons=[]
    if 'regime' in p['conditionSchema']['names']:
        for regime,label in enumerate(['bajista','lateral','alcista']):
            changed,_=inference_condition(last,cfg,p['conditionSchema'],scenario_asset,regime)
            generated=G(np.concatenate([noise,np.repeat(changed[None],samples,axis=0)],axis=1),training=False).numpy()*p['std']+p['mean']
            comparisons.append({'regime':label,'mean':float(generated.mean()),'std':float(generated.std()),'var05':float(np.quantile(generated,.05)),'meanPath':generated.mean(axis=0).tolist(),'scenarios':generated.tolist()})
    return {'scenarioComparisons':comparisons,'unit':9,'task':'generation','winner':winner,'criterion':'Wasserstein + diferencia std + peso temporal × distancia ACF, validación estandarizada y ruido fijo','config':{**cfg,'latentDim':latent,'beta':beta,'horizon':p['horizon'],'conditionPeriod':p['conditionPeriod'],'encoder':enc_cfg,'decoder':dec_cfg,'generator':gen_cfg,'discriminator':disc_cfg},'comparison':results,'splits':{part:{'samples':int(mask.sum())} for part,mask in p['masks'].items()},'purgedSamples':p['purged'],'excludedSamples':p['excluded'],'scalers':{'returnMean':p['mean'],'returnScale':p['std'],'conditionMean':p['conditionMean'],'conditionScale':p['conditionScale'],'conditionSchema':p['conditionSchema']},'test':{'metrics':evaluation},'scenarios':scenarios.tolist(),'histogram':[{'bin':float((edges[i]+edges[i+1])/2),'real':float(real_counts[i]),'synthetic':float(fake_counts[i])} for i in range(len(real_counts))],'condition':{'asset':datasets[scenario_asset]['name'],'logVolatility':float(condition),'sessions':p['conditionPeriod'],'names':p['conditionSchema']['names'],'values':condition_raw.tolist(),'regimeOverride':cfg.get('scenarioRegime','observed')},'note':'cDCGAN: convoluciones 1D con expansión temporal y embeddings configurables. Diagnósticos por régimen y diversidad; no modela dependencia conjunta entre activos. Ventanas completas sin cruzar cortes. CVAE: reparametrización, log-varianza limitada a [-8,8], MSE + beta·KL. cGAN: BCE con logits y actualizaciones alternadas. Ruido fijo solo para seleccionar épocas en validación; test separado. Escenarios de retornos, sin cobertura calibrada.'}
