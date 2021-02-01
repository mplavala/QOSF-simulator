import unittest
import numpy as np
import numpy.testing as npt

from measurement import *


class TestState(unittest.TestCase):
    def test_get_ground_state(self):
        # wrong shape of state
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([[1.], [0.]])
                          )
        # wrong shape of state
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([
                              [[1., 0.], [0., 1.]],
                              [[1., 0.], [0., 1.]]
                          ])
                          )
        # wrong shape of state
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([[1., 0.], [0., 0.]])
                          )
        # wrong length of state, not 2**n
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([1., 0., 0.])
                          )
        # wrong length of state, not 2**n
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([0., 0., 0., 1., 0.])
                          )
        # state not normalized
        self.assertRaises(ValueError,
                          get_measurement_outcome,
                          np.array([1., 0., 0., 1.])
                          )
        # state not normalized
        self.assertRaises(ValueError,
                          get_measurement_outcome,
                          np.array([1., -10. + 8.j, 5., 1.])
                          )
        self.assertEqual(get_measurement_outcome(np.array([0., 1.])),
                         "1"
                         )
        self.assertEqual(get_measurement_outcome(np.array([1., 0., 0., 0.])),
                         "00"
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 1., 0., 0., 0., 0., 0.])),
                         "010"
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 0., 1., 0., 0., 0., 0.])),
                         "011"
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 0., 0., 0., 1., 0., 0.])),
                         "101"
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 0., 0., 0., 0., 0., 1.])),
                         "111"
                         )


if __name__ == '__main__':
    unittest.main()
