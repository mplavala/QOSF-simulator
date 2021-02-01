import unittest
import numpy as np
import numpy.testing as npt

from state import *


class TestState(unittest.TestCase):
    def test_get_ground_state(self):
        npt.assert_array_equal(get_initial_state(1),
                               [1., 0.])
        npt.assert_array_equal(get_initial_state(2),
                               [1., 0., 0., 0.])
        npt.assert_array_equal(get_initial_state(3),
                               [1., 0., 0., 0., 0., 0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
