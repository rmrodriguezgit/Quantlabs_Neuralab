"""Despacho explícito de tareas de las nueve infografías aportadas."""
from features import bounded_int
from financial_lab import config


def run(request):
    unit=bounded_int(request.get('unit',2),2,10,'Unidad');raw=request.get('config') or {};cfg=config(raw);datasets=request.get('datasets',[])
    cfg['projectInput']=bool(raw.get('projectInput',unit in (2,3,4,10)))
    if 'architecture' not in raw and unit in (6,7):
        from neural import architecture
        cfg['architecture']=architecture({'inputUnits':32,'inputActivation':'linear','layers':[{'units':32,'activation':'relu','kernelSize':3},{'units':64,'activation':'relu','kernelSize':3}]} if unit==6 else {'inputUnits':32,'inputActivation':'linear','layers':[{'units':32,'activation':'tanh'}]})
    if not isinstance(datasets,list) or not 1<=len(datasets)<=10:raise ValueError('Selecciona de uno a diez datasets OHLC.')
    if unit<=7:
        from unit_supervised import run as execute
        return execute(datasets,cfg,unit)
    if unit==8:
        from unit_autoencoders import run as execute
    elif unit==9:
        from unit_generative import run as execute
    else:
        from unit_trading import run as execute
    return execute(datasets,cfg)
