"""Bloques de redes para estudios. Todo estado entrenado pertenece al ajuste local."""
import numpy as np
from features import bounded_int
from neural import real
from scaling import fit_affine
from training_control import TrainingControl,checkpoint_callback
from domain.decomposition import causal,seasonal_projection
from .common import output

def build(tf,family,variant,shape,units,dropout,kernel):
    x=tf.keras.Input(shape);h=x
    if family=='rnn':
        for i,width in enumerate(units):
            cell=tf.keras.layers.SimpleRNN(width,return_sequences=i<len(units)-1,activation='tanh')
            h=tf.keras.layers.Bidirectional(cell)(h) if variant==3 else cell(h)
            if dropout:h=tf.keras.layers.Dropout(dropout)(h)
    elif family=='cnn':
        for i,width in enumerate(units):
            shortcut=h;h=tf.keras.layers.Conv1D(width,kernel,padding='causal',dilation_rate=2**i if variant==3 else 1,activation='relu')(h)
            if variant==3:
                if shortcut.shape[-1]!=width:shortcut=tf.keras.layers.Conv1D(width,1)(shortcut)
                h=tf.keras.layers.Add()([shortcut,h])
            if variant==2 and i==0 and int(h.shape[1])>=2:h=tf.keras.layers.AveragePooling1D(2)(h)
            if dropout:h=tf.keras.layers.Dropout(dropout)(h)
        h=tf.keras.layers.GlobalAveragePooling1D()(h)
    else:
        for i,width in enumerate(units):
            sequence=variant==3 or i<len(units)-1;cell=tf.keras.layers.LSTM(width,return_sequences=sequence,activation='tanh',recurrent_activation='sigmoid');h=tf.keras.layers.Bidirectional(cell)(h) if variant==3 else cell(h)
            if dropout:h=tf.keras.layers.Dropout(dropout)(h)
        if variant==3:
            attended=tf.keras.layers.MultiHeadAttention(num_heads=2,key_dim=max(1,units[-1]//2))(h,h);h=tf.keras.layers.Add()([h,attended]);h=tf.keras.layers.GlobalAveragePooling1D()(h)
    return tf.keras.Model(x,tf.keras.layers.Dense(1,activation='linear')(h),name=f'{family}_study_v{variant}')

def forecast_neural(ctx,family,variant):
    import tensorflow as tf
    p=ctx.parameters;settings=ctx.settings;lookback=bounded_int(p.get('lookback',20),2,120,'Ventana');default=[32] if variant==1 else [32,16];units=p.get('units',default)
    if not isinstance(units,list) or not 1<=len(units)<=5:raise ValueError('Configura de una a cinco capas.')
    units=[bounded_int(v,1,128,'Unidades/filtros') for v in units];dropout=real(p.get('dropout',0 if variant==1 else .1),0,.5,'Dropout');lr=real(p.get('learningRate',.001),1e-6,.1,'LR');y=ctx.values;c=causal(y,ctx.period);mode='residual' if family=='rnn' and variant==3 else 'components' if family=='lstm' and variant==2 else 'level';matrix=c[:,2:3] if mode=='residual' else c if mode=='components' else y[:,None];target=c[:,2] if mode=='residual' else y;cut=len(y)-max(4,int(len(y)*.2))
    if cut-lookback<8:raise ValueError('La ventana y VA interna requieren más historial.')
    xc,xs=fit_affine(matrix[:cut-1],p.get('featureScaler','standard'));yc,ys=fit_affine(target[lookback:cut],p.get('targetScaler','standard'));yc=float(yc);ys=float(ys);z=(matrix-xc)/xs
    X=np.asarray([z[i-lookback:i] for i in range(lookback,len(y))],dtype=np.float32);Y=((target[lookback:]-yc)/ys).astype(np.float32);ntrain=cut-lookback;trX=X[:ntrain];trY=Y[:ntrain];vaX=X[ntrain:];vaY=Y[ntrain:];kernels=p.get('kernels',[3,5]) if family=='cnn' and variant==2 else [p.get('kernelSize',3)]
    if not isinstance(kernels,list) or not 1<=len(kernels)<=4:raise ValueError('Kernels: lista de uno a cuatro tamaños.')
    kernels=[bounded_int(v,1,9,'Kernel') for v in kernels];best=None;inner_trials=[]
    for kernel in kernels:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(ctx.seed);m=build(tf,family,variant,trX.shape[1:],units,dropout,kernel);m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,global_clipnorm=1),loss='mse');control=TrainingControl(settings)
        score=lambda a,b:float(np.mean((m(a,training=False).numpy().ravel()-b)**2));cb=checkpoint_callback(tf,control,lambda:score(trX,trY),lambda:score(vaX,vaY));m.fit(trX,trY,epochs=settings['epochs'],batch_size=settings['batchSize'],shuffle=False,verbose=0,callbacks=[cb]);m.set_weights(cb.best_weights);row={'kernel':kernel,'validationMse':control.best,'bestEpoch':control.best_epoch};inner_trials.append(row)
        if best is None or control.best<best['score']:best={'model':m,'kernel':kernel,'score':control.best,'control':control.report(settings['epochs'],'MSE normalizado de VA interna, un paso'),'history':cb.history}
    m=best['model'];history=list(y);feature_rows=list(matrix);pred=[]
    if mode=='residual':
        from sklearn.linear_model import LinearRegression
        n=len(y);tail=min(n,max(24,ctx.period*3));t=np.arange(n);trend=LinearRegression().fit(t[-tail:,None],c[-tail:,0]).predict(np.arange(n,n+ctx.horizon)[:,None]);baseline=trend+seasonal_projection(y,ctx.horizon,ctx.period)
    for step in range(ctx.horizon):
        inp=((np.asarray(feature_rows[-lookback:])-xc)/xs).astype(np.float32)[None];value=float(m(inp,training=False).numpy()[0,0])*ys+yc
        if mode=='residual':pred.append(float(baseline[step]+value));feature_rows.append(np.array([value]))
        else:
            pred.append(value);history.append(value);feature_rows.append(causal(history,ctx.period)[-1] if mode=='components' else np.array([value]))
    layers=[{'name':l.name,'type':type(l).__name__} for l in m.layers]
    return output(pred,ctx,layers=layers,parameters=int(m.count_params()),featureMode=mode,selectedKernel=best['kernel'],innerTrials=inner_trials,history=best['history'],trainingControl=best['control'],scalers={'featureCenter':xc.tolist(),'featureScale':xs.tolist(),'targetCenter':yc,'targetScale':ys,'fitLastInputDate':ctx.index[cut-2].isoformat(),'fitLastTargetDate':ctx.index[cut-1].isoformat(),'internalValidationFirst':ctx.index[cut].isoformat()},note='Características causales recalculadas en cada ajuste; escalas TR interna. Bidireccionalidad/atención solo dentro de la ventana histórica. Pronóstico externo recursivo. Los componentes completos descriptivos no se suministran al entrenamiento.')
