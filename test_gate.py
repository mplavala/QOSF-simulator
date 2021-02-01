import unittest
import numpy as np
import numpy.testing as npt

from gate import *


class TestState(unittest.TestCase):
    def test_gate_x(self):
        npt.assert_array_equal(gate_x(),
                               np.array([[0., 1.],
                                         [1., 0.]])
                               )

    def test_gate_y(self):
        npt.assert_array_equal(gate_y(),
                               np.array([[0., 0. - 1.j],
                                         [0. + 1.j, 0.]])
                               )

    def test_gate_z(self):
        npt.assert_array_equal(gate_z(),
                               np.array([[1., 0.],
                                         [0., -1.]])
                               )

    def test_gate_h(self):
        npt.assert_array_equal(gate_h(),
                               np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                                         [1 / np.sqrt(2), -1 / np.sqrt(2)]])
                               )

    def test_get_single_qubit_unitary(self):
        self.assertRaises(IndexError,
                          get_single_qubit_unitary,
                          1,
                          gate_x(),
                          1
                          )
        self.assertRaises(IndexError,
                          get_single_qubit_unitary,
                          5,
                          np.identity(2),
                          7
                          )
        self.assertRaises(TypeError,
                          get_single_qubit_unitary,
                          1,
                          np.identity(3),
                          0
                          )
        self.assertRaises(ValueError,
                          get_single_qubit_unitary,
                          1,
                          2 * np.identity(2),
                          0
                          )
        self.assertRaises(ValueError,
                          get_single_qubit_unitary,
                          1,
                          np.array([[1., 0. + 1.j],
                                    [0. + 1.j, 0.]]),
                          0
                          )
        npt.assert_array_equal(get_single_qubit_unitary(1,
                                                        gate_z(),
                                                        0),
                               gate_z()
                               )
        npt.assert_array_equal(get_single_qubit_unitary(2,
                                                        gate_x(),
                                                        0),
                               np.array([[0., 0., 1., 0.],
                                         [0., 0., 0., 1.],
                                         [1., 0., 0., 0.],
                                         [0., 1., 0., 0.]])
                               )
        npt.assert_array_equal(get_single_qubit_unitary(3,
                                                        gate_h(),
                                                        2),
                               np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0., 0., 0., 0., 0., 0.],
                                         [1 / np.sqrt(2), -1 / np.sqrt(2), 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1 / np.sqrt(2), 1 / np.sqrt(2), 0., 0., 0., 0.],
                                         [0., 0., 1 / np.sqrt(2), -1 / np.sqrt(2), 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 1 / np.sqrt(2), 1 / np.sqrt(2), 0., 0.],
                                         [0., 0., 0., 0., 1 / np.sqrt(2), -1 / np.sqrt(2), 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 1 / np.sqrt(2), 1 / np.sqrt(2)],
                                         [0., 0., 0., 0., 0., 0., 1 / np.sqrt(2), -1 / np.sqrt(2)]])
                               )

    def test_get_cx_unitary(self):
        self.assertRaises(IndexError,
                          get_cnot_unitary,
                          2,
                          0,
                          2
                          )
        self.assertRaises(IndexError,
                          get_cnot_unitary,
                          7,
                          8,
                          3
                          )
        self.assertRaises(IndexError,
                          get_cnot_unitary,
                          2,
                          1,
                          1
                          )
        npt.assert_array_equal(get_cnot_unitary(2,
                                                0,
                                                1),
                               np.array([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 0., 1.],
                                         [0., 0., 1., 0.]])
                               )
        npt.assert_array_equal(get_cnot_unitary(2,
                                                1,
                                                0),
                               np.array([[1., 0., 0., 0.],
                                         [0., 0., 0., 1.],
                                         [0., 0., 1., 0.],
                                         [0., 1., 0., 0.]])
                               )
        npt.assert_array_equal(get_cnot_unitary(3,
                                                0,
                                                2),
                               np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                         [0., 1., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 1., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 1., 0., 0.],
                                         [0., 0., 0., 0., 1., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0., 1.],
                                         [0., 0., 0., 0., 0., 0., 1., 0.]])
                               )
        npt.assert_array_equal(get_cnot_unitary(3,
                                                2,
                                                0),
                               np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 1., 0., 0.],
                                         [0., 0., 1., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0., 1.],
                                         [0., 0., 0., 0., 1., 0., 0., 0.],
                                         [0., 1., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 1., 0.],
                                         [0., 0., 0., 1., 0., 0., 0., 0.]])
                               )


if __name__ == '__main__':
    unittest.main()
