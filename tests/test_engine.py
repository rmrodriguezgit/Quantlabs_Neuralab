import unittest,sys,json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import engine
from models import predict

class EngineTests(unittest.TestCase):
    def dataset(self,n=180):return engine.simulate({'records':n,'frequency':'monthly','columns':['Ventas'],'seed':42})
    def test_reproducible_simulation(self):
        a=self.dataset();b=self.dataset();self.assertEqual(a['rows'],b['rows']);self.assertEqual(len(a['rows']),180)
    def test_decomposition_identity(self):
        ds=self.dataset();y=np.array([r['Ventas'] for r in ds['rows']]);d=engine.decompose(y,12)
        np.testing.assert_allclose(np.array(d['trend'])+d['seasonal']+d['residual'],y,atol=1e-8)
    def test_metrics_and_constant_r2(self):
        m=engine.metrics([1,2,3],[2,2,2]);self.assertAlmostEqual(m['mse'],2/3);self.assertEqual(m['mae'],2/3);self.assertEqual(m['r2'],0)
        self.assertIsNone(engine.metrics([4,4],[4,4])['r2'])
    def test_holdout_never_used_in_selection(self):
        ds=self.dataset(120);seen=[]
        def recording(name,values,steps,*args):
            seen.append(len(values));return {'prediction':np.repeat(values[-1],steps).tolist()}
        with patch.object(engine,'predict',recording):
            r=engine.analyze({'dataset':ds,'column':'Ventas','methods':['naive'],'validationSteps':4,'folds':2,'forecastSteps':7})
        self.assertEqual(seen[:2],[88,92]);self.assertEqual(seen[-2:],[96,120]);self.assertEqual(len(r['test']['actual']),24)
        self.assertEqual(len(r['future']['dates']),7)
    def test_insufficient_horizon_is_explicit(self):
        r=engine.analyze({'dataset':self.dataset(48),'column':'Ventas','methods':['linear'],'validationSteps':3,'folds':2,'forecastSteps':3})
        self.assertIsNone(r['reports'][-1]['winner']);self.assertTrue(r['reports'][-1]['errors'])
    def test_financial_horizons(self):
        ds=self.dataset();ds['kind']='financial';labels=[v[0] for v in engine.horizons(ds,pd.bdate_range('2020-01-01',periods=252))]
        self.assertEqual(labels,['15 días naturales','30 días naturales','60 días naturales'])
    def test_basic_models(self):
        y=100+np.arange(100)*.5+np.sin(np.arange(100)*2*np.pi/12)+np.random.default_rng(42).normal(0,.4,100)
        for method in ('naive','linear','polynomial','arima','sarima','garch','monte_carlo'):
            with self.subTest(method=method):
                r=predict(method,y,8,12);self.assertEqual(len(r['prediction']),8);self.assertTrue(np.isfinite(r['prediction']).all())
    def test_export_import_csv_xlsx(self):
        ds=self.dataset(48)
        with tempfile.TemporaryDirectory() as directory:
            for fmt in ('csv','xlsx'):
                path=Path(directory)/('test.'+fmt)
                engine.dispatch({'action':'export','path':str(path),'format':fmt,'dataset':ds})
                result=engine.import_files({'paths':[str(path)]})[0]
                self.assertNotIn('error',result);self.assertEqual(result['dataset']['columns'],['Ventas']);self.assertEqual(len(result['dataset']['rows']),48)
    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):engine.simulate({'records':2})
        with self.assertRaises(ValueError):engine.simulate({'columns':['date']})
        ds=self.dataset();ds['rows'][0]['Ventas']=None
        with self.assertRaises(ValueError):engine.analyze({'dataset':ds,'column':'Ventas'})
    def test_json_protocol_stdout(self):
        p=subprocess.run([sys.executable,str(Path(engine.__file__))],input=json.dumps({'action':'simulate','records':24}),text=True,capture_output=True)
        r=json.loads(p.stdout);self.assertTrue(r['ok']);self.assertEqual(len(r['data']['rows']),24)
    def test_import_partial_failure(self):
        result=engine.import_files({'paths':['/missing/not-a-real-file.csv']});self.assertIn('error',result[0])

if __name__=='__main__':unittest.main()
