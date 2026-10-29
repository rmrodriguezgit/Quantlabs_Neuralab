import sys,unittest,copy,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from engine import metrics,analyze,simulate
from financial_lab import demo,config,panel,scale_panel,split_summary,indicators,ohlc_frame
from neural import network_config,prepared_validation,neural_forecast
from target_transforms import transform,restore
from experiment_diagnostics import data_audit
class MetricsAuditTests(unittest.TestCase):
 def test_small_nonzero_mse(self):
  r=metrics([1.1,1.1001,1.1002],[1.1001]*3);self.assertGreater(r['mse'],0);self.assertLess(r['mse'],1e-7);self.assertAlmostEqual(r['r2'],0)
 def test_r2_scaling_invariance(self):
  y=np.array([1.,2.,3.,4.]);p=y+np.array([.2,-.3,.1,.4]);self.assertAlmostEqual(metrics(y,p)['r2'],metrics((y-2)/1e8,(p-2)/1e8)['r2'])
 def test_bad_shapes_rejected(self):
  for y,p in [([1,2],[[1],[2]]),([],[]),([1,float('nan')],[1,2])]:
   with self.assertRaises(ValueError):metrics(y,p)
 def test_affine_scalers_fit_only_train(self):
  ds=demo({'records':1565})
  for method in ['standard','robust','minmax','none']:
   c=config({'featureScaler':method,'targetScaler':method});p=panel([ds],c,20);a=scale_panel(p);q=copy.deepcopy(p);q['X'][~q['masks']['train']]+=1000;q['y'][~q['masks']['train']]-=999;b=scale_panel(q);np.testing.assert_array_equal(a['featureMean'],b['featureMean']);self.assertEqual(a['targetScale'],b['targetScale']);np.testing.assert_allclose(a['y']*a['targetScale']+a['targetMean'],p['y'],rtol=1e-5,atol=2e-5);self.assertLess(a['fitFeatureRows'],int(p['masks']['train'].sum())*20)
 def test_transforms_and_past_only_target_scaling(self):
  y=np.exp(np.linspace(0,.3,120)+np.sin(np.arange(120))*.01)
  for mode in ['level','difference','log_return']:
   values=transform(y,mode);np.testing.assert_allclose(restore(values[1:],y[:-1],mode),y[1:]);c=network_config('lstm',{'networks':{'lstm':{'targetTransform':mode}}});p=prepared_validation(y,c);other=y.copy();other[96:]*=1.2;q=prepared_validation(other,c);self.assertEqual(p['targetMean'],q['targetMean']);np.testing.assert_array_equal(p['featureMean'],q['featureMean'])
 def test_current_features_causal(self):
  d=demo({'records':1565});f=ohlc_frame(d);c=config();a=indicators(f,c);f.iloc[100:]*=2;b=indicators(f,c);np.testing.assert_allclose(a.iloc[:100],b.iloc[:100]);self.assertAlmostEqual(a.intraday_current.iloc[70],np.log(f.Close.iloc[70]/f.Open.iloc[70]))
 def test_split_ranges_multiple_assets(self):
  a=demo({'records':1565});b=demo({'records':1565});b['id']='other';b['rows']=b['rows'][100:];p=panel([b,a],config(),1);rows=split_summary(p);self.assertEqual(rows['train']['firstSignal'],min(r['signalDate'] for r in p['meta'] if r['split']=='train'))
 def test_noise_demo_warning(self):
  d=demo({'records':1565});c=config();p=panel([d],c,1);a=data_audit(p,c,datasets=[d]);self.assertTrue(a['chronological']);self.assertTrue(any('ruido' in s for s in a['warnings']))
 def test_controlled_signal_learnable(self):
  from sklearn.linear_model import Ridge
  d=demo({'records':1565,'demoMode':'signal'});c=config({'selectedFeatures':['intraday_current']});p=panel([d],c,1);s=scale_panel(p);tr=p['masks']['train'];va=p['masks']['validation'];m=Ridge(alpha=1).fit(s['X'][tr,0],s['y'][tr]);r=metrics(p['y'][va],m.predict(s['X'][va,0])*s['targetScale']+s['targetMean']);self.assertGreater(r['r2'],.4)
 def test_general_fold_audit(self):
  d=simulate({'records':96,'frequency':'monthly','columns':['price']});d['name']='EURUSD=X';r=analyze({'dataset':d,'column':'price','methods':['naive'],'folds':2,'validationSteps':6,'forecastSteps':2});a=r['diagnostics'];self.assertEqual(len(a['folds']),2);self.assertEqual(a['validation'][0]['skillVsPersistence'],0);self.assertLess(a['folds'][-1]['validationLast'],a['holdout']['firstDate'])
 def test_mlp_respects_hidden_activation_without_projection(self):
  import tensorflow as tf
  from unit_supervised import model
  c=config({'projectInput':False,'architecture':{'inputActivation':'linear','layers':[{'units':4,'activation':'tanh','dropout':.1}]}});m=model(tf,'mlp',(1,11),c);acts=[l.activation.__name__ for l in m.layers if isinstance(l,tf.keras.layers.Activation)];self.assertEqual(acts,['tanh']);self.assertTrue(any(isinstance(l,tf.keras.layers.Dropout) for l in m.layers))
 def test_custom_holdout_forex_pips(self):
  import pandas as pd
  from engine import serialize
  d=serialize(pd.DataFrame({'Close':1.1+.01*np.sin(np.arange(120)/9)},index=pd.bdate_range('2024-01-01',periods=120)),'EURUSD=X','financial','business');r=analyze({'dataset':d,'column':'Close','methods':['naive'],'folds':3,'validationSteps':6,'testSteps':12,'forecastSteps':2,'options':{'evaluateExtraHorizons':False}});a=r['diagnostics'];self.assertEqual(a['holdout']['samples'],12);self.assertEqual(a['pipSize'],.0001);self.assertAlmostEqual(a['test']['rmsePips'],r['test']['metrics']['rmse']/.0001);self.assertEqual(len(a['folds']),3);self.assertEqual(r['reports'],[])
 def test_daily_download_excludes_current_utc_date(self):
  import pandas as pd
  from unittest.mock import patch
  from engine import finance
  today=pd.Timestamp.now(tz='UTC').normalize().tz_localize(None);index=pd.date_range(end=today,periods=50);frame=pd.DataFrame({'Close':1.1,'Open':1.1,'High':1.11,'Low':1.09,'Volume':0},index=index)
  with patch('yfinance.download',return_value=frame):d=finance({'ticker':'EURUSD=X'})
  self.assertEqual(len(d['rows']),49);self.assertLess(pd.Timestamp(d['rows'][-1]['date']),today)
 def test_log_return_rejects_nonpositive(self):
  with self.assertRaises(ValueError):transform([1.,0.,2.],'log_return')
 def test_level_restored_forecast(self):
  y=1.1+np.sin(np.arange(100)/15)*.01
  for mode in ['difference','log_return']:
   r=neural_forecast('rnn',y,3,42,{'networks':{'rnn':{'epochs':1,'layers':[{'units':4,'activation':'tanh'}],'targetTransform':mode,'featureScaler':'robust','maxTrainSamples':80}}});self.assertTrue(np.isfinite(r['prediction']).all());self.assertGreater(min(r['prediction']),.9);self.assertEqual(r['network']['historyOffset'],20);self.assertEqual(r['network']['scaling']['targetTransform'],mode)
if __name__=='__main__':unittest.main()
