"""Arquitecturas configurables y GAN condicional con características causales."""
import math
import numpy as np
from features import bounded_int,normalize_features,training_windows,feature_matrix
from training_control import MAX_EPOCHS,settings,TrainingControl,checkpoint_callback

ACTIVATIONS={'linear','relu','tanh','sigmoid','elu','selu','softplus','swish'}
OPTIMIZERS={'adam','adamw','rmsprop','sgd','adamax','nadam','adagrad'}
NETWORKS={'rnn','cnn','lstm','dqn','gan'}


def real(value,low,high,label):
    try:value=float(value)
    except (ValueError,TypeError):raise ValueError(f'{label}: valor numérico requerido.')
    if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{label}: rango {low}–{high}.')
    return value


def architecture(raw=None,kind='dense'):
    raw=raw or {}
    if not isinstance(raw,dict):raise ValueError('Arquitectura inválida.')
    activation=raw.get('inputActivation','linear')
    if activation not in ACTIVATIONS:raise ValueError('Activación de entrada no admitida.')
    layers=raw.get('layers',[{'units':24,'activation':'tanh' if kind in ('rnn','lstm') else 'relu'}])
    if not isinstance(layers,list) or not 1<=len(layers)<=5:raise ValueError('Define entre una y cinco capas intermedias.')
    normalized=[]
    for layer in layers:
        if not isinstance(layer,dict) or layer.get('activation','relu') not in ACTIVATIONS:raise ValueError('Activación intermedia no admitida.')
        gate=layer.get('recurrentActivation','sigmoid')
        if gate not in ('sigmoid','tanh','linear'):raise ValueError('Activación recurrente no admitida.')
        normalized.append({'units':bounded_int(layer.get('units',24),1,256,'Neuronas/filtros'),'activation':layer.get('activation','relu'),'recurrentActivation':gate,'dropout':real(layer.get('dropout',0),0,.5,'Dropout'),'kernelSize':bounded_int(layer.get('kernelSize',3),1,9,'Kernel CNN')})
    optimizer=raw.get('optimizer','adam')
    if optimizer not in OPTIMIZERS:raise ValueError('Optimizador no admitido.')
    return {'inputUnits':bounded_int(raw.get('inputUnits',16),1,256,'Neuronas de entrada'),'inputActivation':activation,'layers':normalized,'optimizer':optimizer,'learningRate':real(raw.get('learningRate',.001),1e-6,.1,'Tasa de aprendizaje')}


def network_config(name,options=None):
    options=options or {};networks=options.get('networks',{})
    if not isinstance(networks,dict):raise ValueError('Configuraciones de red inválidas.')
    raw=networks.get(name,{})
    if not isinstance(raw,dict):raise ValueError('Configuración de red inválida.')
    cfg={**architecture(raw,name),'lookback':bounded_int(raw.get('lookback',8),2,120,'Ventana de entrada'),'epochs':bounded_int(raw.get('epochs',options.get('epochs',100)),1,MAX_EPOCHS,'Épocas'),'batchSize':bounded_int(raw.get('batchSize',32),8,256,'Tamaño de lote'),'features':normalize_features(raw.get('features')),**settings(raw),'validationFraction':real(raw.get('validationFraction',.2),.1,.4,'Fracción de validación interna')}
    if name=='gan':cfg.update(discriminator=architecture(raw.get('discriminator'),'dense'),latentDim=bounded_int(raw.get('latentDim',8),1,64,'Dimensión latente'),draws=bounded_int(raw.get('draws',32),8,256,'Escenarios por paso'),reconstructionWeight=real(raw.get('reconstructionWeight',10),0,100,'Peso de reconstrucción'))
    return cfg


def optimizer(tf,config):
    cls={'adam':tf.keras.optimizers.Adam,'adamw':tf.keras.optimizers.AdamW,'rmsprop':tf.keras.optimizers.RMSprop,'sgd':tf.keras.optimizers.SGD,'adamax':tf.keras.optimizers.Adamax,'nadam':tf.keras.optimizers.Nadam,'adagrad':tf.keras.optimizers.Adagrad}[config['optimizer']]
    return cls(learning_rate=config['learningRate'],global_clipnorm=config.get('gradientClip',1.),weight_decay=config.get('weightDecay',0.))


def build_network(tf,name,config,input_shape,outputs=1,output_activation='linear',compile_model=True):
    layers=[tf.keras.layers.Input(shape=input_shape,name='datos'),tf.keras.layers.Dense(config['inputUnits'],activation=config['inputActivation'],name='proyeccion_entrada')]
    for i,layer in enumerate(config['layers']):
        kwargs={'name':f'intermedia_{i+1}','activation':layer['activation']}
        if name=='rnn':layers.append(tf.keras.layers.SimpleRNN(layer['units'],return_sequences=i<len(config['layers'])-1,**kwargs))
        elif name=='lstm':layers.append(tf.keras.layers.LSTM(layer['units'],recurrent_activation=layer['recurrentActivation'],return_sequences=i<len(config['layers'])-1,**kwargs))
        elif name=='cnn':layers.append(tf.keras.layers.Conv1D(layer['units'],layer['kernelSize'],padding='causal',**kwargs))
        else:layers.append(tf.keras.layers.Dense(layer['units'],**kwargs))
        if layer['dropout']:layers.append(tf.keras.layers.Dropout(layer['dropout'],name=f'dropout_{i+1}'))
    if name=='cnn':layers.append(tf.keras.layers.GlobalAveragePooling1D(name='resumen_temporal'))
    layers.append(tf.keras.layers.Dense(outputs,activation=output_activation,name='salida'))
    model=tf.keras.Sequential(layers,name='red_'+name)
    if compile_model:model.compile(optimizer=optimizer(tf,config),loss='mse')
    return model


def metadata(model,cfg,prepared):
    return {'config':cfg,'inputShape':list(model.input_shape[1:]),'outputShape':list(model.output_shape[1:]),'features':prepared['stream'].names,'trainingSamples':prepared['samples'],'warmup':prepared['stream'].warmup,'parameters':model.count_params(),'validationSamples':prepared.get('validationSamples',0),'trainingEndIndex':prepared.get('trainingEndIndex'),'layers':[{'name':layer.name,'type':type(layer).__name__,'units':getattr(layer,'units',getattr(layer,'filters',None)),'activation':getattr(getattr(layer,'activation',None),'__name__',None)} for layer in model.layers]}



def prepared_validation(y,cfg):
    """Bloque interno exclusivo de la ventana de entrenamiento externa."""
    y=np.asarray(y,dtype=float);cut=len(y)-max(4,int(len(y)*cfg['validationFraction']))
    p=training_windows(y[:cut],cfg['features'],cfg['lookback'])
    matrix,stream=feature_matrix(y,cfg['features']);normalized=(matrix-p['featureMean'])/p['featureScale']
    p['validationX']=np.asarray([normalized[i-cfg['lookback']:i] for i in range(cut,len(y))],dtype=np.float32)
    p['validationY']=((y[cut:]-p['targetMean'])/p['targetScale']).astype(np.float32)
    p['matrix']=matrix;p['stream']=stream;p['validationSamples']=len(y)-cut;p['trainingEndIndex']=cut-1
    return p

def neural_forecast(name,y,steps,seed,options):
    import tensorflow as tf
    cfg=network_config(name,options)
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    p=prepared_validation(y,cfg)
    model=build_network(tf,name,cfg,p['X'].shape[1:])
    control=TrainingControl(cfg)
    def score(X,target):return float(np.mean((model(X,training=False).numpy().ravel()-target)**2))
    callback=checkpoint_callback(tf,control,lambda:score(p['X'],p['target']),lambda:score(p['validationX'],p['validationY']))
    model.fit(p['X'],p['target'],epochs=cfg['epochs'],batch_size=cfg['batchSize'],shuffle=False,verbose=0,callbacks=[callback])
    model.set_weights(callback.best_weights)
    rows=list(p['matrix'][-cfg['lookback']:]);out=[]
    for _ in range(steps):
        X=((np.asarray(rows[-cfg['lookback']:])-p['featureMean'])/p['featureScale']).astype(np.float32)[None]
        value=float(model(X,training=False).numpy()[0,0])*p['targetScale']+p['targetMean']
        if not math.isfinite(value):raise ValueError('Pronóstico no finito; revisa arquitectura, activación y tasa de aprendizaje.')
        out.append(value);rows.append(p['stream'].append(value))
    return {'prediction':out,'network':metadata(model,cfg,p),'trainingLoss':[h['train'] for h in callback.history],'validationLoss':[h['validation'] for h in callback.history],'trainingControl':control.report(cfg['epochs'],'MSE de validación interna, anterior al test externo'),'note':'La cola interna reserva validación del ajuste; no se usa el test externo. Indicadores calculados con el pasado disponible; pasos futuros usan predicciones recursivas.'}


def gan_forecast(y,steps,seed,options):
    """cGAN: G(contexto,ruido)→y; D(contexto,y)→probabilidad real."""
    import tensorflow as tf
    cfg=network_config('gan',options)
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    rng=np.random.default_rng(seed);p=prepared_validation(y,cfg)
    X=p['X'].reshape(len(p['X']),-1);target=p['target'][:,None];width=X.shape[1]
    G=build_network(tf,'gan_generator',cfg,(width+cfg['latentDim'],),compile_model=False)
    D=build_network(tf,'gan_discriminator',cfg['discriminator'],(width+1,),output_activation='sigmoid',compile_model=False)
    go=optimizer(tf,cfg);do=optimizer(tf,cfg['discriminator']);bce=tf.keras.losses.BinaryCrossentropy()
    losses=[];control=TrainingControl(cfg);best_weights=None
    validation_context=p['validationX'].reshape(len(p['validationX']),-1)
    fixed_noise=rng.normal(size=(len(validation_context),cfg['latentDim'])).astype(np.float32)
    for epoch_index in range(cfg['epochs']):
        epoch=[]
        # Chronological batches: no target from evaluation windows is supplied.
        for start in range(0,len(X),cfg['batchSize']):
            context=tf.convert_to_tensor(X[start:start+cfg['batchSize']]);truth=tf.convert_to_tensor(target[start:start+cfg['batchSize']]);size=len(context)
            noise=tf.random.normal((size,cfg['latentDim']))
            with tf.GradientTape() as tape:
                fake=tf.stop_gradient(G(tf.concat([context,noise],axis=1),training=False))
                real_score=D(tf.concat([context,truth],axis=1),training=True);fake_score=D(tf.concat([context,fake],axis=1),training=True)
                dl=.5*(bce(tf.ones_like(real_score),real_score)+bce(tf.zeros_like(fake_score),fake_score))
            do.apply_gradients(zip(tape.gradient(dl,D.trainable_variables),D.trainable_variables))
            with tf.GradientTape() as tape:
                generated=G(tf.concat([context,tf.random.normal((size,cfg['latentDim']))],axis=1),training=True)
                score=D(tf.concat([context,generated],axis=1),training=False)
                adversarial=bce(tf.ones_like(score),score);reconstruction=tf.reduce_mean(tf.abs(truth-generated));gl=adversarial+cfg['reconstructionWeight']*reconstruction
            go.apply_gradients(zip(tape.gradient(gl,G.trainable_variables),G.trainable_variables))
            epoch.append([float(gl),float(dl),float(reconstruction)])
        means=np.mean(epoch,axis=0)
        if not np.isfinite(means).all():raise ValueError('GAN divergente; reduce tasas de aprendizaje y tamaño de arquitectura.')
        valpred=G(np.concatenate([validation_context,fixed_noise],axis=1),training=False).numpy().ravel()
        validation_score=float(np.mean((valpred-p['validationY'])**2))
        losses.append({'generator':float(means[0]),'discriminator':float(means[1]),'reconstruction':float(means[2]),'validation':validation_score})
        improved,stop=control.observe(epoch_index+1,validation_score)
        if improved:best_weights=([w.copy() for w in G.get_weights()],[w.copy() for w in D.get_weights()])
        if stop:break
    G.set_weights(best_weights[0]);D.set_weights(best_weights[1])
    rows=list(p['matrix'][-cfg['lookback']:]);out=[];lower=[];upper=[]
    for _ in range(steps):
        context=((np.asarray(rows[-cfg['lookback']:])-p['featureMean'])/p['featureScale']).astype(np.float32).reshape(1,-1)
        noise=rng.normal(size=(cfg['draws'],cfg['latentDim'])).astype(np.float32)
        draws=G(np.concatenate([np.repeat(context,cfg['draws'],axis=0),noise],axis=1),training=False).numpy().ravel()*p['targetScale']+p['targetMean']
        if not np.isfinite(draws).all():raise ValueError('El GAN produjo escenarios no finitos.')
        value=float(np.mean(draws));out.append(value);lower.append(float(np.quantile(draws,.025)));upper.append(float(np.quantile(draws,.975)));rows.append(p['stream'].append(value))
    info=metadata(G,cfg,p);info['discriminator']=metadata(D,cfg['discriminator'],p)
    return {'prediction':out,'lower':lower,'upper':upper,'intervalLabel':'Percentiles generativos 2.5–97.5% por paso; contexto recursivo medio, sin calibración','network':info,'ganLoss':losses,'trainingControl':control.report(cfg['epochs'],'MSE interno del generador con ruido fijo'),'note':'GAN condicional con pérdida adversarial + reconstrucción L1. El contexto futuro usa medias predichas. Puede sufrir colapso de modos; sus percentiles no son intervalos de confianza.'}


def dqn_forecast(y,steps,seed,options):
    import tensorflow as tf
    import gymnasium as gym
    from gymnasium import spaces
    from collections import deque
    cfg=network_config('dqn',options);p=training_windows(y,cfg['features'],cfg['lookback'])
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);rng=np.random.default_rng(seed)
    # State is the full causal feature window, flattened. Action predicts next change.
    start=p['stream'].warmup+cfg['lookback']-1
    normalized=(p['matrix']-p['featureMean'])/p['featureScale']
    states=np.asarray([normalized[t-cfg['lookback']+1:t+1].ravel() for t in range(start,len(y))],dtype=np.float32)
    scale=max(float(np.std(np.diff(y))),1e-8);changes=np.diff(y)[start:]/scale
    class DirectionEnv(gym.Env):
        observation_space=spaces.Box(-np.inf,np.inf,shape=(states.shape[1],),dtype=np.float32)
        action_space=spaces.Discrete(3)
        def reset(self,seed=None,options=None):super().reset(seed=seed);self.i=0;return states[0].copy(),{}
        def step(self,action):
            reward=-float((changes[self.i]-(action-1))**2);self.i+=1
            return states[self.i].copy(),reward,self.i>=len(states)-1,False,{}
    env=DirectionEnv();model=build_network(tf,'dqn',cfg,(states.shape[1],),outputs=3);target=build_network(tf,'dqn',cfg,(states.shape[1],),outputs=3);target.set_weights(model.get_weights())
    memory=deque(maxlen=4000);episodes=bounded_int(options.get('episodes',60),10,500,'Episodios RL');gamma=real(options.get('gamma',.9),0,.9999,'Gamma');epsilon=real(options.get('epsilon',.15),0,1,'Epsilon');rewards=[]
    for episode in range(episodes):
        state,_=env.reset(seed=seed+episode);total=0;tick=0
        while True:
            action=int(rng.integers(3)) if rng.random()<epsilon else int(np.argmax(model(state[None],training=False).numpy()[0]))
            ns,reward,done,_,_=env.step(action);memory.append((state,action,reward,ns,done));state=ns;total+=reward;tick+=1
            if len(memory)>=cfg['batchSize'] and tick%8==0:
                batch=[memory[int(i)] for i in rng.choice(len(memory),cfg['batchSize'],replace=False)]
                ss,aa,rr,nn,dd=map(np.asarray,zip(*batch));q=model(ss,training=False).numpy();q[np.arange(len(batch)),aa]=rr+gamma*(1-dd)*np.max(target(nn,training=False).numpy(),axis=1)
                loss=model.train_on_batch(ss,q)
                if not np.isfinite(loss):raise ValueError('DQN divergente; reduce la tasa de aprendizaje.')
            if done:break
        if episode%5==0:target.set_weights(model.get_weights())
        rewards.append(total)
    rows=list(p['matrix'][-cfg['lookback']:]);out=[];actions=[];last=float(y[-1])
    for _ in range(steps):
        state=((np.asarray(rows[-cfg['lookback']:])-p['featureMean'])/p['featureScale']).astype(np.float32).reshape(1,-1)
        a=int(np.argmax(model(state,training=False).numpy()[0]))-1;last+=a*scale
        if not math.isfinite(last):raise ValueError('Pronóstico DQN no finito.')
        out.append(last);actions.append(a);rows.append(p['stream'].append(last))
    return {'prediction':out,'actions':actions,'trainingRewards':rewards,'network':metadata(model,cfg,p),'note':'DQN con tres valores Q lineales para bajar/mantener/subir, características causales y memoria de experiencias. Política experimental, sin validación de rentabilidad.'}
