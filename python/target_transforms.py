"""Objetivos de red y reconstrucción causal de cotizaciones."""
import numpy as np
MODES=('level','difference','log_return')
def validate_transform(mode):
    if mode not in MODES:raise ValueError('Objetivo neuronal: level, difference o log_return.')
    return mode

def transform(values,mode='level'):
    validate_transform(mode);y=np.asarray(values,dtype=float)
    if mode=='level':return y.copy()
    if mode=='log_return' and np.any(y<=0):raise ValueError('Log-retorno requiere precios estrictamente positivos. Usa diferencia para series con ceros/negativos.')
    return np.r_[np.nan,np.diff(y) if mode=='difference' else np.log(y[1:]/y[:-1])]

def restore(predicted,previous,mode='level'):
    validate_transform(mode);v=np.asarray(predicted,dtype=float);p=np.asarray(previous,dtype=float)
    return v if mode=='level' else p+v if mode=='difference' else p*np.exp(v)
