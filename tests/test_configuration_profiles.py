import json,sys,subprocess,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from experiment_preview import preview
from financial_lab import demo
class RecommendedProfiles(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path(__file__).resolve().parents[1];cls.fixtures=json.loads(subprocess.check_output(['node',str(root/'tests/test-profiles.cjs'),'--fixtures'],cwd=root));cls.ds=demo({'records':1565})
 def test_unit_presets_use_real_preflight(self):
  for entry in self.fixtures['units']:
   with self.subTest(unit=entry['unit'],profile=entry['config']['configurationProfile']['id']):self.assertTrue(preview({'unit':entry['unit'],'config':entry['config'],'datasets':[self.ds]})['valid'])
 def test_bank_presets_use_real_preflight(self):
  for row in self.fixtures['banks']:
   with self.subTest(mode=row['mode'],family=row['families']):
    req={'kind':'bank','mode':row['mode'],'families':row['families'],'trialsPerFamily':row['trials'],'config':row['cfg'],'datasets':[self.ds],'search':{k:row[k] for k in ['widths','depths','lookbacks','latentDims','optimizers','dropouts']}};req['search'].update(lrLogMin=row['low'],lrLogMax=row['high']);self.assertTrue(preview(req)['valid'])
 def test_neural_metadata_preserved(self):
  from neural import network_config
  profile={'id':'network-lstm','mode':'recommended','label':'LSTM','values':{}}
  self.assertEqual(network_config('lstm',{'networks':{'lstm':{'configurationProfile':profile}}})['configurationProfile'],profile)
