import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from experiment_preview import preview
from financial_lab import demo,config,panel,split_summary
class PreviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=demo({'records':1565})
 def request(self,**kw):return {'datasets':[self.data],'config':{'epochs':30},**kw}
 def test_counts_match_training_panel(self):
  p=preview(self.request());self.assertEqual(p['splits'],split_summary(panel([self.data],config({'epochs':30}),1)));self.assertTrue(p['valid']);self.assertEqual(p['maxEpochs'],30)
 def test_missing_data(self):
  with self.assertRaisesRegex(ValueError,'Paso 2'):preview(self.request(datasets=[]))
 def test_chronological_dates(self):
  with self.assertRaises(ValueError):preview(self.request(config={'validationStart':'2025-01-01','testStart':'2024-01-01'}))
 def test_multiple_assets_never_silently_dropped(self):
  with self.assertRaisesRegex(ValueError,'exactamente un activo'):preview(self.request(unit=10,datasets=[self.data,self.data]))
 def test_generation_counts(self):
  p=preview(self.request(unit=9));self.assertGreater(p['splits']['validation']['samples'],8)
 def test_bank_budget(self):
  p=preview(self.request(kind='bank',families=['ridge','mlp'],trialsPerFamily=3));self.assertEqual(p['fits'],6)
 def test_bank_rejects_empty_family(self):
  with self.assertRaisesRegex(ValueError,'familia'):preview(self.request(kind='bank',families=[]))
 def test_cnn_short_window(self):
  with self.assertRaisesRegex(ValueError,'CNN requiere'):preview(self.request(unit=6,config={'lookback':1}))
 def test_classification_criterion(self):self.assertIn('BCE',preview(self.request(config={'task':'classification'}))['criterion'])
 def test_early_stop_warning(self):
  p=preview(self.request(config={'epochs':2,'warmupEpochs':10,'earlyStopping':True}));self.assertTrue(any('calentamiento' in w for w in p['warnings']))
 def test_fold_validation(self):
  with self.assertRaisesRegex(ValueError,'folds internos'):preview(self.request(unit=5,config={'foldYears':[2024]}))
if __name__=='__main__':unittest.main()
