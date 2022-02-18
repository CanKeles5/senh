import numpy as np
import math
import unittest
from metrics import ComputeMetrics


class test_metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1 = np.ones(16000*5000)/2 + np.random.rand(16000*5000)
        cls.s2 = np.random.rand(16000*5000)
        
        cls.metrics_res_rand = ComputeMetrics(s1, s2)
        cls.metrics_res_ = ComputeMetrics()
        cls.metrics_res_ = ComputeMetrics()
    
    def test_SDR(self):
        res = ComputeMetrics(np.ones(16000*50), np.expand_dims(np.ones(16000*50), axis=1))
        self.assertEqual(res.SDR, None)
    
    def test_SIR(self):
        
        pass
        
    def test_SAR(self):
        pass


if __name__ == '__main__':
    unittest.main()
    
