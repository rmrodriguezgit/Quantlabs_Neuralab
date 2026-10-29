"""U8: PCA, AE y DAE; reconstrucción y Ridge son tareas separadas."""
import numpy as np
from features import bounded_int
from neural import architecture,optimizer,real
from training_control import TrainingControl,checkpoint_callback
from financial_lab import panel,scale_panel,split_summary,regression_metrics


def run(datasets,cfg):
    import tensorflow as tf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    p=panel(datasets,cfg,1);s=scale_panel(p);X=s['X'][:,0,:];masks=p['masks'];d=X.shape[1];latent=bounded_int(cfg.get('latentDim',3),1,min(d,int(masks['train'].sum())),'Dimensión latente');noise=real(cfg.get('noiseStd',.1),0,2,'Ruido DAE');enc_cfg=architecture(cfg.get('encoder',{'inputUnits':32,'inputActivation':'relu','layers':[{'units':32,'activation':'relu'}]}));dec_cfg=architecture(cfg.get('decoder',enc_cfg));comparison=[];artifacts={}
    pca=PCA(n_components=latent,random_state=cfg['seed']).fit(X[masks['train']]);reconstruct=lambda values:pca.inverse_transform(pca.transform(values));encode=lambda values:pca.transform(values)
    artifacts['PCA']={'reconstruct':reconstruct,'encode':encode,'history':[],'parameters':int(pca.components_.size),'bestEpoch':None,'variance':pca.explained_variance_ratio_.tolist()}
    for name in ('AE','DAE'):
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(cfg['seed'])
        def stack(shape,c,label,out):
            inp=tf.keras.Input(shape=shape);h=inp
            if cfg.get('projectInput',False):h=tf.keras.layers.Dense(c['inputUnits'],activation=c['inputActivation'],kernel_initializer=tf.keras.initializers.HeNormal(cfg['seed']))(h)
            for i,l in enumerate(c['layers']):
                h=tf.keras.layers.Dense(l['units'],activation=l['activation'],kernel_initializer=tf.keras.initializers.HeNormal(cfg['seed']+i+1))(h)
                if l['dropout']:h=tf.keras.layers.Dropout(l['dropout'],seed=cfg['seed']+100+i)(h)
            result=tf.keras.layers.Dense(out,activation='linear',name=label+'_output')(h);return tf.keras.Model(inp,result,name=label)
        encoder=stack((d,),enc_cfg,'encoder',latent);decoder=stack((latent,),dec_cfg,'decoder',d);inp=tf.keras.Input((d,));corrupted=tf.keras.layers.GaussianNoise(noise,seed=cfg['seed'])(inp) if name=='DAE' else inp;ae=tf.keras.Model(inp,decoder(encoder(corrupted)),name=name);ae.compile(optimizer=optimizer(tf,{**enc_cfg,'gradientClip':cfg['gradientClip']}),loss='mse')
        control=TrainingControl(cfg)
        def score(mask):return float(np.mean((ae(X[mask],training=False).numpy()-X[mask])**2))
        callback=checkpoint_callback(tf,control,lambda:score(masks['train']),lambda:score(masks['validation']))
        ae.fit(X[masks['train']],X[masks['train']],epochs=cfg['epochs'],batch_size=cfg['batchSize'],shuffle=False,verbose=0,callbacks=[callback]);ae.set_weights(callback.best_weights)
        artifacts[name]={'reconstruct':lambda values,m=ae:m(values,training=False).numpy(),'encode':lambda values,m=encoder:m(values,training=False).numpy(),'history':callback.history,'parameters':ae.count_params(),'bestEpoch':control.best_epoch,'trainingControl':control.report(cfg['epochs'],'validation reconstruction MSE'),'encoderShape':[d,latent],'decoderShape':[latent,d]}
    for name,a in artifacts.items():
        va=masks['validation'];tr=masks['train'];recon=a['reconstruct'](X[va]);ridge=Ridge(alpha=cfg['ridgeAlpha']).fit(a['encode'](X[tr]),s['y'][tr]);a['ridge']=ridge
        downstream=ridge.predict(a['encode'](X[va]))*s['targetScale']+s['targetMean']
        comparison.append({'model':name,'validation':{'reconstructionMse':float(np.mean((recon-X[va])**2)),'downstreamRmseBps':regression_metrics(p['y'][va],downstream)['rmse']},'parameters':a['parameters'],'bestEpoch':a['bestEpoch'],'history':a['history'],'trainingControl':a.get('trainingControl')})
    comparison.sort(key=lambda r:r['validation']['reconstructionMse']);winner=comparison[0]['model'];te=masks['test'];tests=[]
    for name,a in artifacts.items():
        recon=a['reconstruct'](X[te]);factors=a['encode'](X[te]);pred=a['ridge'].predict(factors)*s['targetScale']+s['targetMean'];tests.append({'model':name,'reconstructionMse':float(np.mean((recon-X[te])**2)),'downstream':regression_metrics(p['y'][te],pred)})
    selected=artifacts[winner];recon=selected['reconstruct'](X[te]);factors=selected['encode'](X[te]);prediction=selected['ridge'].predict(factors)*s['targetScale']+s['targetMean'];testmeta=[r for r,b in zip(p['meta'],te) if b]
    from experiment_diagnostics import prediction_audit
    predict=lambda mask:selected['ridge'].predict(selected['encode'](X[mask]))*s['targetScale']+s['targetMean']
    audit=prediction_audit(p,cfg,predict,s,selected['parameters'] or 0,'regression',datasets)
    return {'diagnostics':audit,'unit':8,'task':'reconstruction','winner':winner,'criterion':'validation reconstruction MSE (indicadores estandarizados)','config':{**cfg,'latentDim':latent,'noiseStd':noise,'encoder':enc_cfg,'decoder':dec_cfg},'features':cfg['selectedFeatures'],'splits':split_summary(p),'purgedSamples':p['purged'],'excludedSamples':p['excluded'],'comparison':comparison,'testModels':tests,'test':{'metrics':next(t for t in tests if t['model']==winner),'predictions':[{**r,'actual':float(y),'prediction':float(v)} for r,y,v in zip(testmeta,p['y'][te],prediction)]},'reconstruction':[{'feature':f,'mse':float(np.mean((recon[:,i]-X[te,i])**2))} for i,f in enumerate(cfg['selectedFeatures'])],'latent':factors[:500].tolist(),'pcaVariance':artifacts['PCA']['variance'],'note':'AE/DAE/PCA seleccionados por reconstrucción de validación; utilidad predictiva medida separadamente con Ridge entrenado solo en factores de train. DAE añade ruido únicamente al entrenar; objetivo limpio.'}
