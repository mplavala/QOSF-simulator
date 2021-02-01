#!/usr/bin/python3
import numpy as np


def gate_x() -> np.ndarray:
    """Return the unitary operator of the X gate."""
    return np.array([[0., 1.],
                     [1., 0.]])


def gate_y() -> np.ndarray:
    """Return the unitary operator of the Y gate."""
    return np.array([[0., 0. - 1.j],
                     [0. + 1.j, 0.]])


def gate_z() -> np.ndarray:
    """Return the unitary operator of the Z gate."""
    return np.array([[1., 0.],
                     [0., -1.]])


def gate_h() -> np.ndarray:
    """Return the unitary operator of the H gate."""
    return np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                     [1 / np.sqrt(2), -1 / np.sqrt(2)]])


def get_single_qubit_unitary(num_qubits: int,
                             gate_unitary: np.ndarray,
                             target_qubit: int) -> np.ndarray:
    """Return single qubit unitary operator of size 2**n x 2**n for given gate and target qubits."""
    if target_qubit >= num_qubits:
        raise IndexError("Target qubit is outside of qubits array.")

    if np.shape(gate_unitary) != (2, 2):
        raise TypeError("Gate must be 2x2 array.")

    if not np.allclose(np.dot(gate_unitary, gate_unitary.conjugate().transpose()), np.identity(2)):
        raise ValueError("Gate must be unitary.")

    # initialize the matrix
    gate = np.array([1])

    for i in range(num_qubits):
        if i == target_qubit:
            # target qubit, apply unitary
            gate = np.kron(gate, gate_unitary)
        else:
            # not target, apply identity
            gate = np.kron(gate, np.identity(2))

    return gate


def get_cnot_unitary(num_qubits: int,
                     control_qubit: int,
                     target_qubit: int) -> np.ndarray:
    """Return control-X unitary operator of size 2**n x 2**n for given control and target qubits."""
    if control_qubit >= num_qubits:
        raise IndexError("Control qubit is outside of qubits array.")

    if target_qubit >= num_qubits:
        raise IndexError("Target qubit is outside of qubits array.")

    if target_qubit == control_qubit:
        raise IndexError("Target qubit and control qubit must be different.")

    # initialize two matrices
    # gate_0 is the |0><0| part
    gate_0 = np.array([1])
    # gate_1 is the |1><1| part
    gate_1 = np.array([1])

    for i in range(num_qubits):
        if i == control_qubit:
            # control qubit, apply |0><0| and |1><1|
            gate_0 = np.kron(gate_0, np.array([[1., 0.],
                                               [0., 0.]]))
            gate_1 = np.kron(gate_1, np.array([[0., 0.],
                                               [0., 1.]]))
        elif i == target_qubit:
            # target qubit, apply identity to gate_0, unitary to gate_1
            gate_0 = np.kron(gate_0, np.identity(2))
            gate_1 = np.kron(gate_1, gate_x())
        else:
            # not target, not control, apply identities
            gate_0 = np.kron(gate_0, np.identity(2))
            gate_1 = np.kron(gate_1, np.identity(2))

    # we return sum of the matrices
    return np.add(gate_0, gate_1)


if __name__ == '__main__':
    n = 2
    c = 1
    t = 0

    print(get_single_qubit_unitary(n, gate_y(), t))
    print(get_cnot_unitary(n, c, t))
