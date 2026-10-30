import numpy as np
import pandas as pd
from features import bounded_int
from neural import real
from domain.contracts import VariantSpec
FREQUENCIES={'daily':'D','business':'B','monthly':'MS','annual':'YS','hourly':'h','15min':'15min'}
def dates(ctx,horizon=None):
    return pd.date_range(ctx.index[-1]+pd.tseries.frequencies.to_offset(FREQUENCIES[ctx.frequency]),periods=horizon or ctx.horizon,freq=FREQUENCIES[ctx.frequency])
def output(pred,ctx,**metadata):
    a=np.asarray(pred,dtype=float).ravel()
    if a.shape!=(ctx.horizon,) or not np.isfinite(a).all():raise ValueError('El módulo no produjo un pronóstico finito del horizonte requerido.')
    return {'prediction':a.tolist(),'metadata':json_safe(metadata)}
def params(ctx,variant,specs):return {**specs[variant-1].defaults,**ctx.parameters}
def inner_cut(n):
    cut=n-max(8,n//5)
    if cut<16:raise ValueError('Se requieren al menos 24 observaciones para selección interna.')
    return cut

def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value
