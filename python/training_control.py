"""Parada temprana explícita, independiente de la selección del checkpoint."""
import math
from features import bounded_int

MAX_EPOCHS=5000

def settings(raw=None):
    raw=raw or {}
    enabled=raw.get('earlyStopping',False)
    if not isinstance(enabled,bool):raise ValueError('Early Stopping debe ser true o false.')
    def number(key,default,low,high):
        try:value=float(raw.get(key,default))
        except (TypeError,ValueError):raise ValueError(f'{key}: número inválido.')
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{key}: rango {low}–{high}.')
        return value
    return {'earlyStopping':enabled,'patience':bounded_int(raw.get('patience',30),1,1000,'Paciencia'),'minDelta':number('minDelta',1e-5,0,1e6),'warmupEpochs':bounded_int(raw.get('warmupEpochs',10),0,MAX_EPOCHS,'Épocas de calentamiento')}

class TrainingControl:
    def __init__(self,raw=None,mode='min'):
        self.settings=settings(raw);self.mode=mode;self.best=float('inf') if mode=='min' else -float('inf');self.best_epoch=0;self.progress=None;self.wait=0;self.epochs=0;self.stopped=False
    def observe(self,epoch,value):
        value=float(value)
        if not math.isfinite(value):raise ValueError('Métrica de validación no finita.')
        self.epochs=epoch
        improved=value<self.best if self.mode=='min' else value>self.best
        if improved:self.best=value;self.best_epoch=epoch
        if epoch<=self.settings['warmupEpochs']:return improved,False
        meaningful=self.progress is None or (value<self.progress-self.settings['minDelta'] if self.mode=='min' else value>self.progress+self.settings['minDelta'])
        if meaningful:self.progress=value;self.wait=0
        else:self.wait+=1
        self.stopped=self.settings['earlyStopping'] and self.wait>=self.settings['patience']
        return improved,self.stopped
    def report(self,requested,criterion):
        return {'enabled':self.settings['earlyStopping'],'requestedEpochs':requested,'actualEpochs':self.epochs,'bestEpoch':self.best_epoch,'bestValue':self.best,'criterion':criterion,'patience':self.settings['patience'],'minDelta':self.settings['minDelta'],'warmupEpochs':self.settings['warmupEpochs'],'stopReason':'early_stopping' if self.stopped else 'epoch_limit','restoredBestWeights':True}


def checkpoint_callback(tf,control,train_score,validation_score):
    class Checkpoint(tf.keras.callbacks.Callback):
        def __init__(self):super().__init__();self.history=[];self.best_weights=None
        def on_epoch_end(self,epoch,logs=None):
            tr=float(train_score());va=float(validation_score())
            if not math.isfinite(tr):raise ValueError('Pérdida de entrenamiento no finita.')
            self.history.append({'epoch':epoch+1,'train':tr,'validation':va})
            improved,stop=control.observe(epoch+1,va)
            if improved:self.best_weights=[w.copy() for w in self.model.get_weights()]
            if stop:self.model.stop_training=True
    return Checkpoint()
