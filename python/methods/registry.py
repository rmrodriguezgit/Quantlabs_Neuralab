"""Registro declarativo: agregar módulos no requiere editar el orquestador."""
from importlib import import_module
MODULES=('regression','anova','garch','monte_carlo','rnn','cnn','lstm_study','prophet','rl')
def registry():
    result={}
    for module in MODULES:
        study=import_module('methods.'+module).STUDY
        if study.key in result:raise ValueError('Clave de módulo duplicada: '+study.key)
        if [v.number for v in study.variants]!=[1,2,3]:raise ValueError('Cada estudio debe registrar V1, V2 y V3.')
        result[study.key]=study
    return result

def catalog():
    import importlib.util
    return [{'key':s.key,'title':s.title,'lesson':s.lesson,'dependency':s.dependency,'available':not s.dependency or importlib.util.find_spec(s.dependency) is not None,'theorySteps':list(s.theory),'variants':[{'number':v.number,'title':v.title,'description':v.description,'parameters':v.defaults} for v in s.variants]} for s in registry().values()]
