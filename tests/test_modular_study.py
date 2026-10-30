import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from domain.contracts import StudyContext,MethodSpec
from domain.decomposition import causal,descriptive
from domain.metrics import regression
from methods.registry import registry,catalog
from methods.rl import td_targets
from engine import serialize,clean,dispatch
from study_runner import run,preview
class ModularStudyTests(unittest.TestCase):
 def context(self,n=160):
  rng=np.random.default_rng(42);y=100*np.exp(np.cumsum(rng.normal(.0002,.008,n)))+np.sin(np.arange(n)*2*np.pi/5)
  index=pd.bdate_range('2020-01-01',periods=n);frame=pd.DataFrame({'Close':y,'Other':y*.8+rng.normal(size=n)},index=index)
  return StudyContext(y,index,frame,'Close',4,5,42,{'epochs':1,'batchSize':32,'earlyStopping':False,'patience':2,'minDelta':0,'warmupEpochs':0},{},'business')
 def request(self):
  c=self.context(200);d=serialize(c.frame,'Estudio sintético','financial','business')
  return {'dataset':d,'column':'Close','methods':['regression'],'horizons':[0],'config':{'folds':2,'period':5,'epochs':1,'warmupEpochs':0,'allowPersistence':False}}
 def test_catalog_and_causal_prefix(self):
  self.assertEqual(len(catalog()),9)
  for m in catalog():self.assertEqual([v['number'] for v in m['variants']],[1,2,3])
  c=self.context();np.testing.assert_allclose(causal(c.values,5)[:80],causal(c.values[:80],5))
  d=descriptive(c.values,5);np.testing.assert_allclose(np.array(d['trend'])+d['seasonal']+d['residual'],c.values)
 def test_metrics_and_double_targets(self):
  self.assertEqual(regression([1,2,3],[1,2,3])['r2'],1)
  self.assertLess(regression([1,2,3],[9,9,9])['r2'],0)
  self.assertIsNone(regression([1,1],[1,1])['r2'])
  online=np.array([[3,1],[1,3]]);target=np.array([[2,9],[7,4]])
  np.testing.assert_allclose(td_targets(online,target,np.array([1,2]),np.array([0,1]),.5,True),[2,2])
  np.testing.assert_allclose(td_targets(online,target,np.array([1,2]),np.array([0,1]),.5,False),[5.5,2])
 def test_all_real_modules_three_variants(self):
  from dataclasses import replace
  c=self.context()
  for key,spec in registry().items():
   for v in (1,2,3):
    with self.subTest(method=key,variant=v):
     params={**spec.variants[v-1].defaults,'paths':100,'simulations':100,'episodes':1,'units':[4,4],'lookback':5,'kernels':[3],'trendPriors':[.05],'seasonPriors':[1],'trainEvery':128,'maxDegree':3}
     out=spec.forecast(replace(c,parameters=params),v)
     self.assertEqual(len(out['prediction']),4);self.assertTrue(np.isfinite(out['prediction']).all());json.dumps(clean(out),allow_nan=False)
     types=[l['type'] for l in out['metadata'].get('layers',[])]
     if key in ('rnn','lstm') and v==3:self.assertIn('Bidirectional',types)
     if key=='lstm' and v==3:self.assertIn('MultiHeadAttention',types)
     if key=='cnn' and v==2:self.assertIn('AveragePooling1D',types)
     if key=='cnn' and v==3:self.assertIn('Add',types)
 def test_outer_selection_never_sees_test_labels(self):
  req=self.request();seen=[];real=registry()['regression']
  def fake(c,v):
   seen.append((v,len(c.values),len(c.frame)));return {'prediction':np.repeat(c.values[-1],c.horizon).tolist(),'metadata':{}}
  spec=MethodSpec(real.key,real.title,real.lesson,real.variants,fake,real.theory)
  with patch('study_runner.registry',return_value={'regression':spec}):a=run(req)
  h=a['reports'][0]['horizon']['steps'];changed=copy.deepcopy(req)
  for row in changed['dataset']['rows'][-h:]:row['Close']+=50
  with patch('study_runner.registry',return_value={'regression':spec}):b=run(changed)
  self.assertEqual(a['reports'][0]['winner'],b['reports'][0]['winner']);self.assertNotEqual(a['reports'][0]['test']['metrics'],b['reports'][0]['test']['metrics'])
  self.assertTrue(all(n==f for _,n,f in seen));self.assertEqual(sum(n==200-h for _,n,_ in seen),2)
 def test_full_regression_pipeline_and_invalid_plan(self):
  r=dispatch({'action':'study_run',**self.request()});self.assertEqual(len(r['reports'][0]['comparisons']),3);self.assertEqual(r['failures'],[]);json.dumps(clean(r),allow_nan=False)
  req=self.request();req['config']['folds']=5;req['horizons']=[2]
  with self.assertRaisesRegex(ValueError,'insuficiente'):preview(req)
 def test_multimodule_multihorizon_and_parameter_validation(self):
  req=self.request();req['methods']=['regression','anova'];req['horizons']=[0,1]
  r=run(req);self.assertEqual(len(r['reports']),2);self.assertEqual([len(h['comparisons']) for h in r['reports']],[6,6]);self.assertFalse(r['failures'])
  req['config']['variantParameters']={'lstm':{'4':{}}}
  with self.assertRaisesRegex(ValueError,'variantes'):preview(req)
 def test_test_refit_failure_keeps_validation_report(self):
  req=self.request();real=registry()['regression'];h=preview(req)['horizons'][0]['steps']
  def fake(c,v):
   if len(c.values)==200-h:raise ValueError('fallo en ajuste final')
   return {'prediction':np.repeat(c.values[-1],c.horizon).tolist(),'metadata':{}}
  spec=MethodSpec(real.key,real.title,real.lesson,real.variants,fake,real.theory)
  with patch('study_runner.registry',return_value={'regression':spec}):r=run(req)
  self.assertEqual(len(r['reports'][0]['comparisons']),3);self.assertIn('ajuste final',r['reports'][0]['testError']);self.assertEqual(r['failures'][0]['stage'],'test_winner')
if __name__=='__main__':unittest.main()
