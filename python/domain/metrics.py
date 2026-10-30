import math
import numpy as np
def regression(actual,predicted):
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float)
    if a.ndim!=1 or p.ndim!=1 or a.shape!=p.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():raise ValueError('Métricas requieren vectores 1D finitos, no vacíos y de igual longitud.')
    residual=a-p;mse=float(np.mean(residual**2));den=float(np.sum((a-a.mean())**2))
    return {'mse':mse,'rmse':math.sqrt(mse),'mae':float(np.mean(np.abs(residual))),'r2':None if len(a)<2 or np.all(a==a[0]) or den==0 else float(1-np.sum(residual**2)/den)}
