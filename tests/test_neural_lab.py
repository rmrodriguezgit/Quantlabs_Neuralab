import unittest,sys,copy,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from features import FeatureStream,feature_matrix,training_windows
from neural import network_config,build_network,architecture,gan_forecast,neural_forecast,dqn_forecast
import engine


def features():return {'lags':[1,3],'ema':[5,12],'sma':[5],'rsi':True,'rsiPeriod':5,'macd':True,'macdFast':4,'macdSlow':8,'macdSignal':3,'returns':True,'volatility':True,'volatilityPeriod':5}
def options(name,depth=2):
 return {'epochs':1,'episodes':10,'networks':{name:{'inputUnits':7,'inputActivation':'tanh','layers':[{'units':6+i,'activation':'tanh' if name in ('rnn','lstm') else 'relu','kernelSize':2,'dropout':.1,'recurrentActivation':'sigmoid'} for i in range(depth)],'optimizer':'rmsprop','learningRate':.002,'lookback':4,'epochs':1,'batchSize':16,'features':features(),'latentDim':3,'draws':8,'reconstructionWeight':2,'discriminator':{'inputUnits':5,'inputActivation':'relu','layers':[{'units':4,'activation':'tanh'},{'units':3,'activation':'relu'}],'optimizer':'sgd','learningRate':.003}}}}
def series():return 100+np.arange(72)*.2+np.sin(np.arange(72)*2*np.pi/12)+np.random.default_rng(4).normal(0,.3,72)

class FeatureTests(unittest.TestCase):
 def test_feature_prefix_invariance(self):
  y=series();config=features();a,stream=feature_matrix(y,config);b,_=feature_matrix(np.r_[y[:50],np.repeat(9999.,22)],config)
  np.testing.assert_allclose(a[:50],b[:50],equal_nan=True)
  self.assertEqual(a.shape[1],len(stream.names));self.assertTrue(np.isfinite(a[stream.warmup:]).all())
 def test_known_lags_ema_sma_macd(self):
  y=np.arange(1.,31.);matrix,s=feature_matrix(y,{'lags':[1,3],'ema':[3],'sma':[3],'macd':True,'macdFast':3,'macdSlow':5,'macdSignal':2})
  import pandas as pd
  expected_ema=pd.Series(y).ewm(span=3,adjust=False).mean().to_numpy()
  expected_macd=expected_ema-pd.Series(y).ewm(span=5,adjust=False).mean().to_numpy()
  i=20;mapping=dict(zip(s.names,matrix[i]));self.assertEqual(mapping['lag_1'],y[i-1]);self.assertEqual(mapping['lag_3'],y[i-3]);self.assertAlmostEqual(mapping['ema_3'],expected_ema[i]);self.assertAlmostEqual(mapping['sma_3'],y[i-2:i+1].mean());self.assertAlmostEqual(mapping['macd'],expected_macd[i]);self.assertAlmostEqual(mapping['macd_signal'],pd.Series(expected_macd).ewm(span=2,adjust=False).mean().iloc[i])
 def test_rsi_up_down_flat(self):
  for values,expected in [(np.arange(30.),100),(np.arange(30.)[::-1],0),(np.ones(30),50)]:
   matrix,s=feature_matrix(values,{'lags':[],'rsi':True,'rsiPeriod':5});self.assertEqual(matrix[-1,s.names.index('rsi')],expected)
 def test_scalers_and_windows_exclude_target(self):
  y=series();p=training_windows(y,features(),4);raw=p['matrix'];i=p['start']
  np.testing.assert_allclose(p['X'][0],((raw[i-4:i]-p['featureMean'])/p['featureScale']),rtol=1e-5,atol=1e-5)
  self.assertAlmostEqual(float(p['target'][0]),(y[i]-p['targetMean'])/p['targetScale'],places=5)
  np.testing.assert_allclose(p['featureMean'],raw[p['stream'].warmup:-1].mean(axis=0))
 def test_recursive_stream_matches_batch(self):
  values=series();matrix,s=feature_matrix(values[:50],features());tail=np.asarray([s.append(v) for v in values[50:]]);whole,_=feature_matrix(values,features());np.testing.assert_allclose(tail,whole[50:])
 def test_rolling_volatility(self):
  values=series();matrix,s=feature_matrix(values,{'lags':[],'volatility':True,'volatilityPeriod':5});self.assertAlmostEqual(matrix[-1,-1],np.diff(values)[-5:].std(ddof=1))
 def test_invalid_configs(self):
  for raw in ({'macd':True,'macdFast':26,'macdSlow':12},{'lags':[0]},{'lags':[1,1]},{'ema':[1]},{'rsiPeriod':2.5}):
   with self.subTest(raw=raw),self.assertRaises(ValueError):FeatureStream(raw)
  with self.assertRaises(ValueError):training_windows(np.arange(40.),{'ema':[50]},8)
  for patch in ({'layers':[]},{'layers':[{'units':4}]*6},{'optimizer':'unknown'},{'inputActivation':'unknown'},{'learningRate':0},{'inputUnits':257}):
   with self.subTest(patch=patch),self.assertRaises(ValueError):network_config('lstm',{'networks':{'lstm':patch}})

@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow opcional no instalado')
class NetworkLabTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import tensorflow as tf
  cls.tf=tf
 def test_stacked_architectures_all_depths(self):
  tf=self.tf
  for name in ('rnn','lstm','cnn','dqn'):
   for depth in (1,3,5):
    with self.subTest(name=name,depth=depth):
     tf.keras.backend.clear_session();cfg=network_config(name,options(name,depth));shape=(4,12) if name!='dqn' else (48,);model=build_network(tf,name,cfg,shape,outputs=3 if name=='dqn' else 1)
     self.assertEqual(len([l for l in model.layers if l.name.startswith('intermedia')]),depth);self.assertEqual(model.get_layer('proyeccion_entrada').units,7);self.assertEqual(model.get_layer('salida').activation.__name__,'linear');self.assertEqual(type(model.optimizer).__name__,'RMSprop')
     output=model(np.ones((2,*shape),dtype=np.float32),training=False).numpy();self.assertEqual(output.shape,(2,3 if name=='dqn' else 1))
 def test_custom_forecast_features(self):
  for name in ('rnn','lstm','cnn'):
   with self.subTest(name=name):
    output=neural_forecast(name,series(),3,42,options(name));self.assertEqual(len(output['prediction']),3);self.assertTrue(np.isfinite(output['prediction']).all());self.assertIn('macd',output['network']['features']);self.assertEqual(output['network']['config']['layers'][1]['units'],7)
 def test_all_optimizers(self):
  tf=self.tf
  for name in ('adam','rmsprop','sgd','adamax','nadam','adagrad'):
   tf.keras.backend.clear_session();cfg=architecture({'optimizer':name});model=build_network(tf,'dense',cfg,(5,));loss=model.train_on_batch(np.ones((8,5),np.float32),np.ones((8,1),np.float32));self.assertTrue(np.isfinite(loss))
 def test_conditional_gan_separate_networks(self):
  output=gan_forecast(series(),4,42,options('gan'));self.assertEqual(len(output['prediction']),4);self.assertTrue(np.isfinite(output['prediction']).all());self.assertTrue(np.all(np.asarray(output['lower'])<=output['upper']));self.assertEqual(output['network']['layers'][-1]['activation'],'linear');self.assertEqual(output['network']['discriminator']['layers'][-1]['activation'],'sigmoid');self.assertEqual(output['network']['discriminator']['config']['optimizer'],'sgd');self.assertEqual(len(output['ganLoss']),1);self.assertEqual(output['network']['inputShape'][0],4*len(output['network']['features'])+3)
 def test_custom_dqn(self):
  output=dqn_forecast(series(),3,42,options('dqn'));self.assertEqual(len(output['prediction']),3);self.assertEqual(output['network']['outputShape'],[3]);self.assertEqual(output['network']['config']['optimizer'],'rmsprop')
 def test_gan_in_ranking_and_export_metadata(self):
  ds=engine.simulate({'records':72,'frequency':'annual','columns':['Ventas'],'seed':42});config=options('gan');result=engine.analyze({'dataset':ds,'column':'Ventas','methods':['gan'],'folds':1,'validationSteps':2,'forecastSteps':2,'options':config})
  self.assertEqual(result['winner'],'gan');self.assertTrue(np.isfinite(result['test']['metrics']['rmse']));self.assertIn('gan',result['experiment']['options']['networks']);self.assertIn('network',result['future']);self.assertEqual(len(result['future']['prediction']),2)

if __name__=='__main__':unittest.main()
