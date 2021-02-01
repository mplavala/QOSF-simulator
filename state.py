#!/usr/bin/python3
import numpy as np


def get_initial_state(num_qubits: int) -> np.array:
    """Return vector of size 2**num_qubits with all zeroes except first element which is 1."""

    # initialize zeros
    state = np.zeros((2**num_qubits))
    # change first element to 1
    state[0] = 1.

    return state


if __name__ == '__main__':
    n = 3
    print(get_initial_state(n))
    print(np.shape(get_initial_state(n)))
