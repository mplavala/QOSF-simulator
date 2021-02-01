import unittest
import numpy as np
import numpy.testing as npt

from measurement import *


class TestState(unittest.TestCase):
    def test_get_ground_state(self):
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([[1.], [0.]])
                          )
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([
                              [[1., 0.], [0., 1.]],
                              [[1., 0.], [0., 1.]]
                          ])
                          )
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([[1., 0.], [0., 0.]])
                          )
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([1., 0., 0.])
                          )
        self.assertRaises(TypeError,
                          get_measurement_outcome,
                          np.array([0., 0., 0., 1., 0.])
                          )
        self.assertRaises(ValueError,
                          get_measurement_outcome,
                          np.array([1., 0., 0., 1.])
                          )
        self.assertRaises(ValueError,
                          get_measurement_outcome,
                          np.array([1., -10. + 8.j, 5., 1.])
                          )
        self.assertEqual(get_measurement_outcome(np.array([0., 1.])),
                         1
                         )
        self.assertEqual(get_measurement_outcome(np.array([1., 0., 0., 0.])),
                         0
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 1., 0., 0., 0., 0., 0.])),
                         2
                         )
        self.assertEqual(get_measurement_outcome(np.array([0., 0., 0., 0., 0., 1., 0., 0.])),
                         5
                         )


if __name__ == '__main__':
    unittest.main()
