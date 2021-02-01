#!/usr/bin/python3
import numpy as np
from state import *


if __name__ == '__main__':
    numberOfQubits = 1
    X = np.array([[0., 1.],
                  [1., 0.]])

    q0 = get_initial_state(1)

    q0 = np.dot(X, q0)

    print(q0)
