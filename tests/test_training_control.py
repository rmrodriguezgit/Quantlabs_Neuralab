import unittest,sys,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from training_control import TrainingControl,settings
from neural import network_config,prepared_validation,neural_forecast,gan_forecast
from financial_lab import demo,config,panel
from unit_supervised import fit,run

class ControlTests(unittest.TestCase):
 def test_on_off_patience(self):
  for enabled in (True,False):
   c=TrainingControl({'earlyStopping':enabled,'patience':2,'warmupEpochs':0,'minDelta':.01});stops=[]
   for i,x in enumerate([1.,1.,1.,1.,1.],1):
    improved,stop=c.observe(i,x);stops.append(stop)
   self.assertEqual(stops[2],enabled);self.assertEqual(c.best_epoch,1)
 def test_min_delta_separate_from_best_checkpoint(self):
  c=TrainingControl({'earlyStopping':True,'patience':2,'warmupEpochs':0,'minDelta':.1});c.observe(1,1.);improved,stop=c.observe(2,.99);self.assertTrue(improved);self.assertFalse(stop);improved,stop=c.observe(3,.98);self.assertTrue(improved);self.assertTrue(stop);self.assertEqual(c.best_epoch,3)
 def test_warmup(self):
  c=TrainingControl({'earlyStopping':True,'patience':2,'warmupEpochs':2});stop=[c.observe(i,1)[1] for i in range(1,6)];self.assertEqual(stop,[False,False,False,False,True])
 def test_1500_and_limits(self):
  self.assertEqual(config({'epochs':1500})['epochs'],1500);self.assertEqual(network_config('lstm',{'networks':{'lstm':{'epochs':1500}}})['epochs'],1500)
  with self.assertRaises(ValueError):config({'epochs':5001})
  with self.assertRaises(ValueError):settings({'earlyStopping':'false'})
 def test_scalers_exclude_internal_validation(self):
  cfg=network_config('lstm',{'networks':{'lstm':{'lookback':4}}});y=np.arange(100.)+100;a=prepared_validation(y,cfg);mutated=y.copy();mutated[80:]+=1000;b=prepared_validation(mutated,cfg);np.testing.assert_equal(a['featureMean'],b['featureMean']);self.assertEqual(a['targetMean'],b['targetMean']);self.assertEqual(a['validationSamples'],20)

@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow opcional no instalado')
class ActualTrainingTests(unittest.TestCase):
 def test_1500_ceiling_stops_and_off_runs_all(self):
  ds=demo({'records':1565});arch={'inputUnits':4,'inputActivation':'relu','layers':[{'units':4,'activation':'relu'}]}
  for enabled,epochs,actual in [(True,1500,3),(False,4,4)]:
   cfg=config({'architecture':arch,'epochs':epochs,'earlyStopping':enabled,'patience':2,'warmupEpochs':0,'minDelta':1000000});p=panel([ds],cfg,1);r=fit(p,cfg);control=r['trainingControl'];self.assertEqual(control['actualEpochs'],actual);self.assertEqual(control['enabled'],enabled);self.assertEqual(control['stopReason'],'early_stopping' if enabled else 'epoch_limit');pred=r['predict'](p['masks']['validation']);raw=(pred-r['scale']['targetMean'])/r['scale']['targetScale'];actual_loss=float(np.mean((raw-r['scale']['y'][p['masks']['validation']])**2));self.assertAlmostEqual(actual_loss,control['bestValue'],places=5)
 def test_general_neural_and_gan_switch(self):
  y=100+np.arange(72)*.2+np.sin(np.arange(72));base={'epochs':1500,'earlyStopping':True,'patience':2,'warmupEpochs':0,'minDelta':1000000,'inputUnits':4,'layers':[{'units':4,'activation':'tanh'}],'lookback':4,'draws':8}
  for name in ('lstm','gan'):
   options={'networks':{name:base}};r=neural_forecast(name,y,2,42,options) if name!='gan' else gan_forecast(y,2,42,options);self.assertEqual(r['trainingControl']['actualEpochs'],3);self.assertTrue(r['trainingControl']['restoredBestWeights']);self.assertGreater(r['network']['validationSamples'],0)
 def test_u3_toggle_and_u4_variant_can_be_excluded(self):
  ds=demo({'records':1565});raw={'epochs':4,'earlyStopping':True,'patience':1,'warmupEpochs':0,'minDelta':1000000,'architecture':{'inputUnits':4,'layers':[{'units':4,'activation':'relu'}]}}
  r=run([ds],config(raw),3);self.assertTrue(all(c['trainingControl']['actualEpochs']==2 for c in r['comparison'] if not c.get('reference')))
  r=run([ds],config({**raw,'epochs':1,'includeEarlyStopping':False}),4);self.assertNotIn('EarlyStopping',[c['model'] for c in r['comparison']]);self.assertTrue(all(not c['trainingControl']['enabled'] for c in r['comparison'] if not c.get('reference')))

if __name__=='__main__':unittest.main()
