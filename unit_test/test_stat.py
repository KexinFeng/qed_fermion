import unittest
import matlab
import numpy as np
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.utils.stat import init_convex_seq_estimator

class TestInitConvexSeqEstimator(unittest.TestCase):

    def setUp(self):
        # Initialize MATLAB engine mock
        # self.eng = matlab.engine.start_matlab()
        # self.eng.addpath('/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/utils/init_seq_matlab')
        np.random.seed(42)

    def tearDown(self):
        # Quit MATLAB engine
        # self.eng.quit()
        pass

    def test_init_convex_seq_estimator_shape(self):
        array = np.random.rand(10, 5, 3)
        result = init_convex_seq_estimator(array)
        self.assertEqual(result.shape, array[0].shape)

    def test_init_convex_seq_estimator_values(self):
        array = np.random.rand(10, 5, 3)
        result = init_convex_seq_estimator(array)
        self.assertTrue(np.all(result >= 0))


if __name__ == '__main__':
    unittest.main()