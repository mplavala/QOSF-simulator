#!/usr/bin/python3
import numpy as np
import random


def get_measurement_outcome(state: np.ndarray) -> int:
    """Return random outcome of measurement in computational basis."""
    if len(np.shape(state)) != 1:
        raise TypeError("State vector must be shape (n,).")

    if np.modf(np.log2(np.shape(state)[0]))[0] != 0:
        raise TypeError("Length of state vector must be power of 2.")

    if not np.allclose(np.inner(state.conjugate(), state), 1.):
        raise ValueError("State vector must be normalized.")

    prob = np.real(np.multiply(state.conjugate(), state))
    n = np.shape(state)[0]
    outcomes = tuple(range(n))

    return random.choices(outcomes, prob)[0]


if __name__ == '__main__':
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    # psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(4) * (1 - 1j)])
    # psi = np.array([1., 0., 0., 0.])

    print(get_measurement_outcome(psi))
