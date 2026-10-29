"""Modelos serializables sin Lambda: cGAN denso y cDCGAN temporal 1D."""
import math
from features import bounded_int
from neural import real

def build(tf,arch,cfg,schema,latent,horizon,role,convolutional=False):
    n=len(schema['names']);prefix=latent if role=='generator' else horizon;total=prefix+n
    inp=tf.keras.Input((total,));packed=tf.keras.layers.Reshape((total,1))(inp)
    def segment(start,length):
        return tf.keras.layers.Flatten()(tf.keras.layers.Cropping1D((start,total-start-length))(packed))
    data=segment(0,prefix);parts=[]
    embed=bounded_int(cfg.get('conditionEmbeddingDim',4),1,32,'Dimensión embedding')
    for i in range(n):
        value=segment(prefix+i,1)
        if str(i) in schema['categories']:
            value=tf.keras.layers.Flatten()(tf.keras.layers.Embedding(schema['categories'][str(i)],embed,name=f'condition_{schema["names"][i]}')(value))
        parts.append(value)
    condition=parts[0] if len(parts)==1 else tf.keras.layers.Concatenate()(parts)
    fusion=cfg.get('conditionFusion','concat')
    if fusion not in ('concat','multiply'):raise ValueError('Fusión de condición no válida.')
    if role=='generator' and fusion=='multiply':data=tf.keras.layers.Multiply()([data,tf.keras.layers.Dense(latent,activation='tanh')(condition)])
    joined=tf.keras.layers.Concatenate()([data,condition])
    slope=real(cfg.get('ganNegativeSlope',.01),0,1,'Pendiente LeakyReLU')
    def block(h,l,index):
        if l.get('batchNorm',cfg.get('ganBatchNorm',False)) and not (role=='discriminator' and index==0):h=tf.keras.layers.BatchNormalization()(h)
        if cfg.get('ganLeaky',False) or l['activation']=='leaky_relu':h=tf.keras.layers.LeakyReLU(negative_slope=slope)(h)
        else:h=tf.keras.layers.Activation(l['activation'])(h)
        if l['dropout']:h=tf.keras.layers.Dropout(l['dropout'])(h)
        return h
    if not convolutional:
        h=joined
        if cfg.get('projectInput',False):h=tf.keras.layers.Dense(arch['inputUnits'],activation=arch['inputActivation'])(h)
        for i,l in enumerate(arch['layers']):h=block(tf.keras.layers.Dense(l['units'])(h),l,i)
        out=tf.keras.layers.Dense(horizon if role=='generator' else 1)(h)
    elif role=='generator':
        strides=[bounded_int(l.get('stride',2),1,4,'Stride GAN') for l in arch['layers']];factor=math.prod(strides);length=math.ceil(horizon/factor)
        if length*factor>1024:raise ValueError('Arquitectura GAN demasiado amplia: reduce strides o capas.')
        h=tf.keras.layers.Reshape((length,arch['inputUnits']))(tf.keras.layers.Dense(length*arch['inputUnits'],activation=arch['inputActivation'])(joined))
        for i,(l,stride) in enumerate(zip(arch['layers'],strides)):
            if cfg.get('ganUpsampling','transpose')=='transpose':h=tf.keras.layers.Conv1DTranspose(l['units'],l['kernelSize'],strides=stride,padding='same')(h)
            elif cfg.get('ganUpsampling')=='upsample':h=tf.keras.layers.Conv1D(l['units'],l['kernelSize'],padding='same')(tf.keras.layers.UpSampling1D(stride)(h))
            else:raise ValueError('Método de expansión GAN no válido.')
            h=block(h,l,i)
        h=tf.keras.layers.Conv1D(1,1,padding='same')(h)
        if length*factor>horizon:h=tf.keras.layers.Cropping1D((0,length*factor-horizon))(h)
        out=tf.keras.layers.Reshape((horizon,))(h)
    else:
        h=tf.keras.layers.Concatenate()([tf.keras.layers.Reshape((horizon,1))(data),tf.keras.layers.RepeatVector(horizon)(condition)])
        for i,l in enumerate(arch['layers']):
            h=block(tf.keras.layers.Conv1D(l['units'],l['kernelSize'],strides=bounded_int(l.get('stride',2),1,4,'Stride GAN'),padding='same')(h),l,i)
        out=tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(h))
    return tf.keras.Model(inp,out,name=('cdcgan_' if convolutional else 'conditional_')+role)
