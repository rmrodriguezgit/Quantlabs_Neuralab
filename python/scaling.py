"""Escaladores afines sin clipping, ajustados exclusivamente con entrenamiento."""
import numpy as np
METHODS=('standard','robust','minmax','none')
def validate_method(name):
    if name not in METHODS:raise ValueError('Scaler válido: standard, robust, minmax o none.')
    return name

def fit_affine(values,method='standard'):
    validate_method(method);x=np.asarray(values,dtype=np.float64)
    if not x.size or not np.isfinite(x).all():raise ValueError('Scaler requiere entrenamiento finito y no vacío.')
    if method=='standard':center=x.mean(axis=0);scale=x.std(axis=0)
    elif method=='robust':center=np.median(x,axis=0);scale=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
    elif method=='minmax':lo=x.min(axis=0);hi=x.max(axis=0);center=(lo+hi)/2;scale=(hi-lo)/2
    else:center=np.zeros(x.shape[1:]);scale=np.ones(x.shape[1:])
    # Constant/near-constant columns keep a finite divisor; never fit on VA/TE.
    return np.asarray(center),np.where(np.asarray(scale)>1e-8,scale,1.)
