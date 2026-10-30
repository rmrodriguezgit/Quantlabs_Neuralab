import unittest,sys,copy,json
from unittest.mock import patch
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from engine import serialize
from strategy_search import preview,run,choose_weight
class StrategyTests(unittest.TestCase):
 def request(self,constant=False):
  values=np.ones(500)*1.1 if constant else 1.1+np.arange(500)*.001
  d=serialize(pd.DataFrame({'Close':values},index=pd.bdate_range('2020-01-01',periods=500)),'EURUSD=X','financial','business')
  return {'dataset':d,'column':'Close','config':{'methods':['cnn','rnn'],'histories':[100],'targets':['level','log_return'],'lookbacks':[4],'featureSets':['minimal'],'scalers':['standard'],'units':[4],'depths':[1],'dropouts':[0],'learningRates':[.001],'seeds':[42,43],'horizon':10,'folds':4,'epochs':1,'screenLimit':2,'refineLimit':1,'finalists':1}}
 def fake(self,name,values,steps,*args):
  return {'prediction':(values[-1]+.001*np.arange(1,steps+1)).tolist(),'network':{'historyOffset':0,'scaling':{'featureFitLastIndex':len(values)-2}}}
 def test_preview_no_training(self):
  with patch('strategy_search.predict',side_effect=AssertionError('No debe entrenar')):p=preview(self.request())
  self.assertEqual(p['test']['samples'],10);self.assertEqual(p['searchSamples'],20);self.assertLess(p['folds'][-1]['validationLast'],p['test']['firstDate'])
 def test_winner_invariant_to_test_labels(self):
  req=self.request()
  with patch('strategy_search.predict',self.fake):a=run(req)
  changed=copy.deepcopy(req)
  for row in changed['dataset']['rows'][-10:]:row['Close']+=20
  with patch('strategy_search.predict',self.fake):b=run(changed)
  self.assertEqual(a['selected'],b['selected']);self.assertEqual(a['finalValidation'],b['finalValidation']);self.assertNotEqual(a['test']['metrics'],b['test']['metrics']);self.assertEqual(a['selected']['modelWeight'],1)
 def test_no_test_fit_for_losing_candidates(self):
  req=self.request();seen=[]
  def observe(name,values,steps,*args):seen.append((name,len(values)));return self.fake(name,values,steps,*args)
  with patch('strategy_search.predict',observe):r=run(req)
  final=[name for name,n in seen if n==490];self.assertEqual(len(final),2);self.assertEqual(set(final),{r['selected']['method']});self.assertEqual(sum(n==500 for _,n in seen),2)
 def test_fallback_constant_target_and_serialization(self):
  req=self.request(constant=True)
  with patch('strategy_search.predict',self.fake):r=run(req)
  self.assertTrue(r['fallbackToPersistence']);self.assertEqual(r['selected']['method'],'naive');self.assertEqual(r['test']['metrics']['mse'],0);json.dumps(r,allow_nan=False)
 def test_freeze_ensemble_weight_on_search(self):
  y=np.array([1.,2.]);model=np.array([2.,4.]);base=np.zeros(2);self.assertEqual(choose_weight(y,model,base,[0,.25,.5,.75,1]),.5)
 def test_reject_duplicate_single_seed_and_budget(self):
  req=self.request();req['config']['seeds']=[42,42]
  with self.assertRaisesRegex(ValueError,'semillas'):preview(req)
  req=self.request();req['config'].update(screenLimit=81,refineLimit=40,folds=12,seeds=[1,2,3,4,5],finalists=5,histories=[80],targets=['level'])
  # Small input cannot support all folds for these settings before training.
  req['config']['horizon']=40
  with self.assertRaises(ValueError):preview(req)
 def test_all_network_failures_reported_not_hidden(self):
  with patch('strategy_search.predict',side_effect=ValueError('fallo controlado')):r=run(self.request())
  self.assertTrue(r['allCandidatesFailed']);self.assertGreater(len(r['failures']),0);self.assertEqual(r['selected']['method'],'naive')
 def test_history_caps_not_silently_equivalent(self):
  req=self.request();req['config']['histories']=[100,1000];p=preview(req);self.assertEqual(p['config']['histories'],[100]);self.assertTrue(any('omiten' in w for w in p['warnings']))
if __name__=='__main__':unittest.main()
