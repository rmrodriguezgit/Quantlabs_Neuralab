import unittest,importlib.util,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from models import predict
class AdvancedTests(unittest.TestCase):
    def run_model(self,method,requirements):
        missing=[m for m in requirements if importlib.util.find_spec(m) is None]
        if missing:self.skipTest('Dependencias opcionales ausentes: '+', '.join(missing))
        y=100+np.arange(64)*.2+np.sin(np.arange(64)*2*np.pi/12)+np.random.default_rng(4).normal(0,.3,64)
        output=predict(method,y,5,12,42,{'epochs':1,'episodes':10})
        self.assertEqual(len(output['prediction']),5)
        self.assertTrue(np.isfinite(output['prediction']).all())
    def test_prophet(self):self.run_model('prophet',['prophet'])
    def test_rnn(self):self.run_model('rnn',['tensorflow'])
    def test_cnn(self):self.run_model('cnn',['tensorflow'])
    def test_lstm(self):self.run_model('lstm',['tensorflow'])
    def test_qlearning(self):self.run_model('qlearning',['gymnasium'])
    def test_dqn(self):self.run_model('dqn',['tensorflow','gymnasium'])
if __name__=='__main__':unittest.main()
