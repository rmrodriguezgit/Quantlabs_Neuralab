import unittest,sys,tempfile,zipfile,json,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from financial_lab import config,demo,ohlc_frame
from generative_conditions import observation,inference_condition
from unit_generative import windows,run,selection_score,distribution
from emulator import run as emulate
from bot_infer import infer
class ConditionsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ds=demo({'records':1565})
 def test_causal_conditions(self):
  f=ohlc_frame(self.ds);cfg={'conditionFeatures':['volatility','momentum','regime','asset']};a=observation(f,100,cfg);f.iloc[100:]*=10;np.testing.assert_array_equal(a,observation(f,100,cfg))
 def test_train_only_schema(self):
  cfg=config({'conditionFeatures':['volatility','momentum','regime','asset']});p=windows([self.ds],cfg);expected=p['rawConditions'][p['masks']['train']].mean(axis=0);np.testing.assert_allclose(p['conditionSchema']['mean'][:2],expected[:2]);self.assertEqual(p['conditionSchema']['categories'],{'2':3,'3':1})
 def test_volume_missing(self):
  ds=copy.deepcopy(self.ds);ds['columns']=[c for c in ds['columns'] if c.lower()!='volume'];ds['rows']=[{k:v for k,v in r.items() if k.lower()!='volume'} for r in ds['rows']]
  with self.assertRaisesRegex(ValueError,'Volume'):windows([ds],config({'conditionFeatures':['volume']}))
 def test_regime_override(self):
  cfg=config({'conditionFeatures':['volatility','regime']});p=windows([self.ds],cfg);a,_=inference_condition(ohlc_frame(self.ds),cfg,p['conditionSchema'],regime=0);b,_=inference_condition(ohlc_frame(self.ds),cfg,p['conditionSchema'],regime=2);self.assertEqual(a[0],b[0]);self.assertEqual(b[1]-a[1],2)
 def test_temporal_score_penalizes_order(self):
  rng=np.random.default_rng(4);a=np.sort(rng.normal(size=(100,16)),axis=1);b=a[:,rng.permutation(16)];self.assertAlmostEqual(selection_score(a,b,0),0);self.assertGreater(selection_score(a,b,1),.1)
 def test_diversity_diagnostics(self):
  p=distribution(np.random.default_rng(2).normal(size=(30,16)),np.zeros((30,16)));self.assertTrue(p['diversity']['lowDiversity']);self.assertIn('squaredReturnAcf5',p['real'])
class NetworkTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ds=demo({'records':1565})
 def cfg(self,**kw):return config({'epochs':1,'batchSize':128,'horizon':7,'latentDim':2,'scenarioCount':8,'generativeModels':['cdcgan'],'conditionFeatures':['volatility','momentum','asset','regime'],'ganBatchNorm':True,'ganLeaky':True,'conditionFusion':'multiply','generator':{'inputUnits':4,'layers':[{'units':4,'kernelSize':3,'stride':2}]},'discriminator':{'layers':[{'units':4,'kernelSize':3,'stride':2},{'units':4,'stride':2}]},**kw})
 def test_transpose_and_upsampling_serialization(self):
  import tensorflow as tf
  for method in ['transpose','upsample']:
   r=run([self.ds],self.cfg(ganUpsampling=method),evaluate_test=False);g=r['_generator'];self.assertEqual(g.output_shape,(None,7));p=r['_prepared'];x=np.c_[np.zeros((2,2)),p['c'][:2]]
   with tempfile.TemporaryDirectory() as d:
    file=Path(d)/'g.keras';g.save(file);restored=tf.keras.models.load_model(file,compile=False);np.testing.assert_allclose(g(x).numpy(),restored(x).numpy(),atol=1e-6)
 def test_full_report(self):
  r=run([self.ds],self.cfg());self.assertEqual(r['winner'],'CDCGAN');self.assertEqual(np.shape(r['scenarios']),(8,7));self.assertEqual(len(r['test']['metrics']['byRegime']),3);self.assertEqual(len(r['scenarioComparisons']),3)
 def test_export_and_inference(self):
  with tempfile.TemporaryDirectory() as d:
   r=emulate({'mode':'generation','families':['cdcgan'],'trialsPerFamily':1,'datasets':[self.ds],'config':self.cfg(),'search':{'widths':[4],'depths':[1],'latentDims':[2]},'artifactRoot':d});w=r['winners'][0];target=Path(d)/'unpack'
   with zipfile.ZipFile(Path(d)/(w['artifact']['id']+'.zip')) as z:z.extractall(target)
   output=infer(target,scenarios=8,regime=0);self.assertEqual(np.shape(output['returnScenarios']),(8,7));self.assertTrue(np.isfinite(output['returnScenarios']).all());self.assertFalse(output['ordersSubmitted']);self.assertEqual(output['conditions']['values'][-1],0)
if __name__=='__main__':unittest.main()
