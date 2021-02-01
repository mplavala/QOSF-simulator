#!/usr/bin/python3
import numpy as np
import random


def get_measurement_outcome(state: np.ndarray) -> str:
    """Return random outcome of measurement in computational basis."""
    if len(np.shape(state)) != 1:
        raise TypeError("State vector must be shape (n,).")

    if np.modf(np.log2(np.shape(state)[0]))[0] != 0:
        raise TypeError("Length of state vector must be power of 2.")

    if not np.allclose(np.inner(state.conjugate(), state), 1.):
        raise ValueError("State vector must be normalized.")

    # output probability distribution
    prob = np.real(np.multiply(state.conjugate(), state))
    # dimension of the Hilbert space
    dim = np.shape(state)[0]
    # number of qubits
    n = int(np.modf(np.log2(dim))[1])
    # tuple of all possible outcomes
    outcomes = tuple(range(dim))

    # random package used to sample the outcome of the measurement
    result = random.choices(outcomes, prob)[0]

    return bin(result)[2:].zfill(n)


if __name__ == '__main__':
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    # psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(4) * (1 - 1j)])
    # psi = np.array([1., 0., 0., 0.])

    print(get_measurement_outcome(psi))
