"""
Unit test utils.
"""

import unittest
import utils


class TestUtils(unittest.TestCase):
    def test_convert_to_int16(self):
        pass
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6, "")
    
    def test_open_audio(self):
        pass
    
    def test_add_noise(self):
        rand = np.random(1, 10000)
        zeros = np.zeros(1, 10000)
        
        assertEqual(utils.add_noise(rand, zeros), rand, "")
    
    def test_add_noise_wave(self):
        rand1 = np.ones(1, 10000)
        rand2 = np.zeros(1, 10000)
        
        assertLess(utils.add_noise_wave(radn1, rand2), utils.add_noise(rand1, rand2), "")
        
    def test_add_reverb(self):
        rand = np.random(1, 10000)
        
        assertNotEqual(utils.add_reverb(rand), rand, "")

if __name__ == '__main__':
    unittest.main()
