import unittest,sys,copy,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from financial_lab import demo,config,ohlc_frame,indicators,panel,scale_panel,trade_path
from unit_generative import windows
from unit_trading import double_targets,TradingEnv
from unit_lab import run

class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ds=demo({'records':1565});cls.cfg=config()
 def test_signal_target_and_purge(self):
  p=panel([self.ds],self.cfg,20);val=pd.Timestamp(self.cfg['validationStart']);test=pd.Timestamp(self.cfg['testStart']);self.assertGreater(p['purged'],0)
  for row in p['meta']:
   a,b=pd.Timestamp(row['signalDate']),pd.Timestamp(row['targetDate']);self.assertLess(a,b)
   if row['split']=='train':self.assertLess(b,val)
   elif row['split']=='validation':self.assertGreaterEqual(a,val);self.assertLess(b,test)
   else:self.assertGreaterEqual(a,test)
 def test_target_is_next_intraday_bps(self):
  p=panel([self.ds],self.cfg,1);f=ohlc_frame(self.ds);row=p['meta'][20];target=pd.Timestamp(row['targetDate']);self.assertAlmostEqual(p['y'][20],10000*np.log(f.loc[target,'Close']/f.loc[target,'Open']))
 def test_indicator_prefix_causality(self):
  f=ohlc_frame(self.ds);a=indicators(f,self.cfg);mutated=f.copy();mutated.iloc[800:]*=5;b=indicators(mutated,self.cfg);np.testing.assert_allclose(a.iloc[:800],b.iloc[:800],equal_nan=True)
 def test_scale_fits_only_train(self):
  p=panel([self.ds],self.cfg,20);a=scale_panel(p);q=copy.deepcopy(p);q['X'][~q['masks']['train']]+=10000;q['y'][~q['masks']['train']]+=10000;b=scale_panel(q);np.testing.assert_equal(a['featureMean'],b['featureMean']);self.assertEqual(a['targetMean'],b['targetMean'])
 def test_no_windows_across_assets(self):
  second=copy.deepcopy(self.ds);second['id']='other';second['name']='Otro';single=panel([self.ds],self.cfg,20);both=panel([self.ds,second],self.cfg,20);self.assertEqual(len(both['X']),2*len(single['X']));np.testing.assert_equal(both['X'][:len(single['X'])],single['X']);np.testing.assert_equal(both['X'][len(single['X']):],single['X'])
 def test_generation_purges_whole_windows(self):
  p=windows([self.ds],self.cfg);self.assertEqual(p['X'].shape[1],16);self.assertGreater(p['purged'],2)
  f=ohlc_frame(self.ds);returns=np.log(f.Close/f.Close.shift(1));first=p['meta'][0];start=f.index.get_loc(pd.Timestamp(first['start']));expected=np.log(max(returns.iloc[start-20:start].std(ddof=1),1e-8));recovered=float(p['c'][0,0]*p['conditionScale']+p['conditionMean']);self.assertAlmostEqual(recovered,expected,places=5)
  for row in p['meta']:
   if row['split']=='train':self.assertLess(pd.Timestamp(row['end']),pd.Timestamp(self.cfg['validationStart']))
   if row['split']=='validation':self.assertGreaterEqual(pd.Timestamp(row['start']),pd.Timestamp(self.cfg['validationStart']));self.assertLess(pd.Timestamp(row['end']),pd.Timestamp(self.cfg['testStart']))
 def test_double_q_online_selects_target_evaluates(self):
  actual=double_targets(np.array([1.,2.]),np.array([False,True]),np.array([[10.,1.,0.],[0.,1.,5.]]),np.array([[3.,100.,200.],[1.,2.,300.]]),.95);np.testing.assert_allclose(actual,[1+.95*3,2])
 def test_execution_gap_intraday_costs_terminal(self):
  meta=[{'assetId':'A','asset':'A','targetDate':'2025-01-02','gap':.01,'intraday':.02},{'assetId':'A','asset':'A','targetDate':'2025-01-03','gap':.03,'intraday':-.01}];env=TradingEnv(np.zeros((2,11)),meta,5,True);state=env.reset();self.assertEqual(len(state),12)
  state,reward,done,_=env.step(2);self.assertFalse(done);expected=(1-.0005)*(1+.02)-1;self.assertAlmostEqual(reward,100*expected);self.assertEqual(state[-1],1)
  state,reward,done,_=env.step(0);self.assertTrue(done);expected2=(1+.03)*(1-.001)*(1+.01)*(1-.0005)-1;self.assertAlmostEqual(reward,100*expected2)
  path=trade_path(meta,[1,-1],5,True);self.assertAlmostEqual(path['path'][0]['return'],expected);self.assertAlmostEqual(path['path'][1]['return'],expected2)
 def test_intraday_strategy_is_flat_overnight(self):
  meta=[{'assetId':'A','asset':'A','targetDate':'2025-01-02','gap':.9,'intraday':.02}];a=trade_path(meta,[1],5,overnight=False);expected=(1-.0005)*(1+.02)*(1-.0005)-1;self.assertAlmostEqual(a['metrics']['netReturn'],expected)
 def test_test_end_and_train_start_exclusion(self):
  cfg=config({'trainStart':'2021-01-01','validationStart':'2023-01-01','testStart':'2024-01-01','testEnd':'2025-01-01'});p=panel([self.ds],cfg,1);self.assertGreater(p['excluded'],0);self.assertTrue(all(pd.Timestamp(r['signalDate'])>=pd.Timestamp('2021-01-01') and pd.Timestamp(r['targetDate'])<pd.Timestamp('2025-01-01') for r in p['meta']))
 def test_no_ohlc_or_wrong_frequency(self):
  ds=copy.deepcopy(self.ds);ds['frequency']='hourly'
  with self.assertRaises(ValueError):ohlc_frame(ds)
  ds=copy.deepcopy(self.ds);ds['columns']=['Close'];ds['rows']=[{'date':r['date'],'Close':r['Close']} for r in ds['rows']]
  with self.assertRaises(ValueError):ohlc_frame(ds)

@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow opcional no instalado')
class UnitExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ds=demo({'records':1565});cls.arch={'inputUnits':8,'inputActivation':'relu','layers':[{'units':4,'activation':'relu','kernelSize':3}],'optimizer':'adam','learningRate':.001}
 def run_unit(self,unit,patch=None):
  cfg={'epochs':1,'architecture':self.arch,'encoder':self.arch,'decoder':self.arch,'generator':self.arch,'discriminator':self.arch,'lookback':8,'trials':1,'widths':[8],'depths':[1],'foldYears':[2022,2023],'recurrentModels':['lstm','gru','bilstm','bigru'],'latentDim':2,'scenarioCount':8,'episodes':1,'replayBatch':16,'replayWarmup':32,'trainEvery':32,'targetSync':2,**(patch or {})}
  return run({'unit':unit,'datasets':[self.ds],'config':cfg})
 def test_u2_regression_and_classification(self):
  for task in ('regression','classification'):
   result=self.run_unit(2,{'task':task});self.assertEqual(result['task'],task);self.assertEqual(result['winner'],'MLP');self.assertTrue(np.isfinite(result['test']['metrics']['rmse' if task=='regression' else 'bce']));self.assertTrue(result['test']['predictions'])
 def test_u3_fair_optimizers(self):
  r=self.run_unit(3);self.assertTrue({'SGD','RMSPROP','ADAMW'}.issubset({c['model'] for c in r['comparison']}));self.assertTrue(all(len(c['history'])==1 for c in r['comparison'] if c['history']))
 def test_u4_variants_and_data_loss(self):
  r=self.run_unit(4);self.assertEqual(r['task'],'classification');names={v['model'] for v in r['comparison']};self.assertTrue({'Base','Dropout','L1','L2','BatchNorm','EarlyStopping'}.issubset(names));self.assertEqual(r['criterion'],'validation bce')
 def test_u5_nested_folds(self):
  r=self.run_unit(5);trial=r['trials'][0];self.assertEqual([f['year'] for f in trial['folds']],[2022,2023]);self.assertAlmostEqual(trial['meanRmseBps'],np.mean([f['rmseBps'] for f in trial['folds']]));self.assertNotEqual(trial['folds'][0]['scalerMean'],trial['folds'][1]['scalerMean'])
  with self.assertRaises(ValueError):self.run_unit(5,{'foldYears':[2024]})
 def test_u6_cnn_and_flatten_reference(self):
  r=self.run_unit(6);self.assertTrue({'CNN 1D','MLP ventana'}.issubset({c['model'] for c in r['comparison']}))
 def test_u7_all_recurrent_families(self):
  r=self.run_unit(7,{'architecture':{**self.arch,'layers':[{'units':4,'activation':'tanh'}]}});self.assertTrue({'LSTM','GRU','BILSTM','BIGRU'}.issubset({c['model'] for c in r['comparison']}))
 def test_u8_reconstruction_and_downstream(self):
  r=self.run_unit(8);self.assertEqual(r['task'],'reconstruction');self.assertEqual(len(r['testModels']),3);self.assertEqual(len(r['latent'][0]),2);self.assertTrue(all(np.isfinite(m['reconstructionMse']) and np.isfinite(m['downstream']['rmse']) for m in r['testModels']))
 def test_u9_conditional_vae_gan(self):
  r=self.run_unit(9);self.assertEqual(r['task'],'generation');self.assertEqual(np.asarray(r['scenarios']).shape,(8,16));self.assertTrue(np.isfinite(r['test']['metrics']['wasserstein']));self.assertEqual({m['model'] for m in r['comparison']},{'CVAE','CGAN'})
 def test_u10_replay_and_costed_baselines(self):
  r=self.run_unit(10);self.assertEqual(r['network']['inputShape'],[12]);self.assertEqual(r['network']['outputShape'],[3]);self.assertEqual(r['training']['steps'],r['splits']['train']['samples']);self.assertGreater(r['training']['updates'],0);self.assertEqual(r['training']['replayContains'],'solo train');self.assertEqual(len(r['testStrategies']),4);cash=next(t for t in r['testStrategies'] if t['model']=='Efectivo');self.assertEqual(cash['metrics']['netReturn'],0)
 def test_dense_initial_weights_match_regularized_variants(self):
  import tensorflow as tf
  from unit_supervised import model
  cfg=config({'architecture':self.arch});base=model(tf,'mlp',(1,11),cfg,'classification',{});bn=model(tf,'mlp',(1,11),cfg,'classification',{'batchNorm':True})
  for name in ('dense_0','dense_1','output'):
   np.testing.assert_equal(base.get_layer(name).get_weights()[0],bn.get_layer(name).get_weights()[0])

if __name__=='__main__':unittest.main()
