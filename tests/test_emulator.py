import unittest,sys,tempfile,json,zipfile,importlib.util
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from emulator import run
from financial_lab import demo,indicators,config,ohlc_frame,panel,regression_metrics
from bot_infer import infer
from model_exports import bundle_artifacts

@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow opcional no instalado')
class EmulatorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ds=demo({'records':1565})
 def request(self,root,mode,families,trials=1):return {'mode':mode,'families':families,'trialsPerFamily':trials,'datasets':[self.ds],'artifactRoot':root,'config':{'epochs':1,'earlyStopping':False,'batchSize':64,'horizon':8,'latentDim':2,'episodes':1,'replayBatch':16,'replayWarmup':32,'trainEvery':32,'targetSync':2},'search':{'widths':[4],'depths':[1],'lookbacks':[4],'latentDims':[2],'optimizers':['adam'],'dropouts':[0],'lrLogMin':-3,'lrLogMax':-3}}
 def unpack(self,root,artifact):
  dest=Path(root)/('extracted_'+artifact['id']);dest.mkdir()
  with zipfile.ZipFile(Path(root)/(artifact['id']+'.zip')) as z:self.assertIsNone(z.testzip());z.extractall(dest)
  return dest
 def test_predictive_variants_rank_and_exports_reload(self):
  with tempfile.TemporaryDirectory() as root:
   result=run(self.request(root,'predictive',['ridge','mlp','lstm','ae_ridge','pca_ridge'],2));self.assertEqual(len(result['winners']),5)
   for winner in result['winners']:
    variants=[r for r in result['variants'] if r['family']==winner['family']];self.assertEqual(winner['validation']['mse'],min(v['validation']['mse'] for v in variants));self.assertEqual(winner['validation']['r2'],max(v['validation']['r2'] for v in variants));folder=self.unpack(root,winner['artifact']);signal=infer(folder);self.assertTrue(np.isfinite(signal['predictedIntradayBps']));self.assertFalse(signal['ordersSubmitted']);self.assertTrue((folder/'ridge.json').exists() or (folder/'predictor.weights.npz').exists());manifest=json.loads((folder/'manifest.json').read_text());self.assertEqual(manifest['validationMetrics'],winner['validation']);self.assertFalse(manifest['optimizerStateIncluded'])
   bundle=bundle_artifacts(root,[w['artifact']['id'] for w in result['winners']],result)
   with zipfile.ZipFile(Path(root)/(bundle['id']+'.zip')) as z:self.assertIn('selection_report.json',z.namelist());self.assertIn('mlp/predictor.keras',z.namelist())
 def test_remaining_predictive_families(self):
  with tempfile.TemporaryDirectory() as root:
   families=['rnn','gru','bilstm','bigru','dae_ridge'];result=run(self.request(root,'predictive',families));self.assertEqual({w['family'] for w in result['winners']},set(families))
   for winner in result['winners']:
    self.assertTrue(np.isfinite(winner['validation']['mse']));folder=self.unpack(root,winner['artifact']);self.assertTrue(np.isfinite(infer(folder)['predictedIntradayBps']))
 def test_neural_package_has_same_test_metrics(self):
  import tensorflow as tf
  with tempfile.TemporaryDirectory() as root:
   req=self.request(root,'predictive',['cnn']);result=run(req);winner=result['winners'][0];folder=self.unpack(root,winner['artifact']);manifest=json.loads((folder/'manifest.json').read_text());p=panel([self.ds],config(req['config']),4);s=manifest['scalers'];X=((p['X']-np.asarray(s['featureMean']))/s['featureScale']).astype(np.float32);model=tf.keras.models.load_model(folder/'predictor.keras',compile=False);te=p['masks']['test'];pred=model(X[te],training=False).numpy().ravel()*s['targetScale']+s['targetMean'];metrics=regression_metrics(p['y'][te],pred);self.assertAlmostEqual(metrics['mse'],winner['test']['mse'],places=2);self.assertAlmostEqual(metrics['r2'],winner['test']['r2'],places=5)
 def test_classification_packages(self):
  with tempfile.TemporaryDirectory() as root:
   result=run(self.request(root,'classification',['logistic','mlp']));self.assertEqual(len(result['winners']),2)
   for winner in result['winners']:
    folder=self.unpack(root,winner['artifact']);signal=infer(folder);self.assertTrue(0<=signal['probabilityUp']<=1);self.assertIn('bce',winner['test']);self.assertFalse(signal['ordersSubmitted'])
 def test_generation_packages(self):
  with tempfile.TemporaryDirectory() as root:
   result=run(self.request(root,'generation',['cvae','cgan']));self.assertEqual(len(result['winners']),2)
   for winner in result['winners']:
    folder=self.unpack(root,winner['artifact']);signal=infer(folder,scenarios=8);self.assertEqual(np.asarray(signal['returnScenarios']).shape,(8,8));self.assertTrue(np.isfinite(signal['returnScenarios']).all());self.assertIn('wasserstein',winner['test'])
 def test_trading_packages(self):
  with tempfile.TemporaryDirectory() as root:
   result=run(self.request(root,'trading',['dqn','ddqn']));self.assertEqual(len(result['winners']),2)
   for winner in result['winners']:
    folder=self.unpack(root,winner['artifact']);signal=infer(folder,position=0);self.assertEqual(len(signal['qValues']),3);self.assertIn(signal['targetPosition'],[-1,0,1]);self.assertTrue((folder/'target.keras').exists());self.assertIn('netReturn',winner['test'])
 def test_test_metrics_only_for_family_winners(self):
  calls=[]
  def record(actual,predicted):calls.append(len(actual));return regression_metrics(actual,predicted)
  with tempfile.TemporaryDirectory() as root,patch('emulator.regression_metrics',side_effect=record):
   req=self.request(root,'predictive',['ridge','mlp'],2);req.pop('artifactRoot');result=run(req);p=panel([self.ds],config(req['config']),4);self.assertEqual(calls.count(int(p['masks']['validation'].sum())),4);self.assertEqual(calls.count(int(p['masks']['test'].sum())),2)
 def test_tamper_detection(self):
  with tempfile.TemporaryDirectory() as root:
   result=run(self.request(root,'predictive',['ridge']));folder=self.unpack(root,result['winners'][0]['artifact']);(folder/'ridge.json').write_text('{}')
   with self.assertRaises(ValueError):infer(folder)

if __name__=='__main__':unittest.main()
