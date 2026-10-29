"""Paquetes de inferencia portables, sin pickle ni claves de intermediario."""
import json,uuid,zipfile,hashlib,shutil,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
from financial_lab import ohlc_frame


def export_bundle(root,family,kind,cfg,datasets,scalers,models=None,extras=None,validation=None,test=None,lookback=1):
    artifact_id=uuid.uuid4().hex;base=Path(root);folder=base/artifact_id;folder.mkdir(parents=True);models=models or {};extras=extras or {};manifest={'schemaVersion':'1.0','family':family,'kind':kind,'config':{k:v for k,v in cfg.items() if not k.startswith('_')},'seed':cfg['seed'],'lookback':lookback,'features':cfg.get('selectedFeatures',[]),'scalers':scalers,'validationMetrics':validation,'testMetrics':test,'selectionUses':'validation only; test after family winner selection','target':'10000*log(next Close / next Open)' if kind in ('predictive','latent','ridge') else 'direction = next intraday log return > 0' if kind in ('classifier','logistic') else kind,'assets':[],'runtime':{'python':platform.python_version(),'tensorflow':version('tensorflow') if models else None,'numpy':version('numpy'),'pandas':version('pandas')},'weights':{},'mode':'paper','optimizerStateIncluded':False}
    for i,ds in enumerate(datasets):
        file=f'context_{i}.csv';f=ohlc_frame(ds);f.index.name='date';f.to_csv(folder/file);manifest['assets'].append({'name':ds['name'],'id':ds['id'],'contextFile':file,'lastObservedDate':f.index[-1].isoformat()})
    if models:import tensorflow as tf
    for name,model in models.items():
        clone=tf.keras.models.clone_model(model);clone.set_weights(model.get_weights());clone.save(folder/f'{name}.keras');(folder/f'{name}.architecture.json').write_text(model.to_json(),encoding='utf-8');np.savez_compressed(folder/f'{name}.weights.npz',**{f'weight_{i}':w for i,w in enumerate(model.get_weights())});manifest['weights'][name]=[{'name':str(w.name),'path':str(getattr(w,'path',w.name)),'shape':list(w.shape),'dtype':str(w.dtype)} for w in model.weights]
    for name,value in extras.items():(folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    source=Path(__file__).parent
    for name in ['bot_infer.py','financial_lab.py','neural.py','features.py','training_control.py','generative_conditions.py','scaling.py','target_transforms.py']:shutil.copy2(source/name,folder/name)
    (folder/'requirements.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in (['numpy','pandas']+(['tensorflow'] if models else [])))+'\n',encoding='utf-8')
    (folder/'risk.json').write_text(json.dumps({'mode':'paper','liveOrdersEnabled':False,'requiresClosedDailyBars':True,'maxAbsolutePosition':1,'costBpsPerUnit':cfg.get('costBps',5),'credentialsIncluded':False},indent=2),encoding='utf-8')
    (folder/'README.md').write_text('''# Paquete de inferencia financiera\n\nContiene arquitectura, modelos Keras sin estado de optimizador, pesos NPZ, escalas, configuración, métricas y contexto OHLC para reproducir indicadores. No contiene claves ni envía órdenes.\n\n```bash\npython -m venv .venv\n# Activar .venv según tu sistema\npip install -r requirements.txt\npython bot_infer.py --bundle . --output signal.json\npython bot_infer.py --bundle . --csv barras_nuevas.csv --output signal.json\n```\n\nCSV: date ISO y Open, High, Low, Close. Usa barras diarias cerradas y las mismas convenciones de ajuste de precios. El contexto es histórico para indicadores; no implica que los pesos se hayan entrenado con test. `--asof FECHA` permite inferencia retrospectiva con contexto cortado. `--asset 0` elige el primer activo; `--position -1/0/1` informa la posición al modelo de trading. Generadores aceptan `--scenarios 32`.\n\n`.keras` conserva arquitectura y pesos; `*.weights.npz` incluye los arrays en el orden descrito por el manifest. Ridge/PCA utilizan JSON numérico, sin pickle. Los pesos de TensorFlow necesitan adaptación de disposición si se reconstruyen en otro framework. El paquete está preparado para inferencia, no para reanudar el optimizador en el estado exacto del checkpoint.\n\nUna integración con broker debe implementar tamaño, calendario, controles de riesgo, ejecución, persistencia, observabilidad y pruebas en paper. Las señales y escenarios son experimentales; sus métricas históricas no garantizan rentabilidad.\n''',encoding='utf-8')
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()};(folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    archive=base/f'{artifact_id}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in folder.iterdir():z.write(file,file.name)
    return {'id':artifact_id,'filename':family+'-inference.zip','sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'contains':['manifest JSON']+(['modelos Keras','pesos NPZ'] if models else ['coeficientes numéricos JSON'])+['escaladores','contexto OHLC','script de inferencia','controles paper']}


def bundle_artifacts(root,ids,report=None):
    import re
    if not isinstance(ids,list) or not 1<=len(ids)<=20 or any(not re.fullmatch('[a-f0-9]{32}',str(i)) for i in ids):raise ValueError('Identificadores de paquete inválidos.')
    base=Path(root);artifact_id=uuid.uuid4().hex;archive=base/f'{artifact_id}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for artifact in ids:
            with zipfile.ZipFile(base/f'{artifact}.zip') as source:
                manifest=json.loads(source.read('manifest.json'));family=manifest['family']
                if not re.fullmatch('[a-z0-9_]+',family):raise ValueError('Familia de paquete inválida.')
                for name in source.namelist():
                    if Path(name).name!=name:raise ValueError('Ruta de paquete inválida.')
                    output.writestr(family+'/'+name,source.read(name))
        output.writestr('selection_report.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        output.writestr('README.md','# Ganadores por familia\n\nCada subcarpeta es un paquete independiente de inferencia. Ver su README y manifest. selection_report.json conserva variantes, validación y test reservado. Los scripts funcionan en modo paper y no envían órdenes.\n')
    return {'id':artifact_id,'filename':'forecast-winners.zip','sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size}
