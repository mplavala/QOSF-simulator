#!/usr/bin/python3
import time

from simulator import *

if __name__ == '__main__':
    # starting timer to calculate run time
    start_time = time.time()

    # Number of qubits in the circuit. Everything works up to 8 qubits.
    qubits = 3

    # number of shots to execute
    shots = 1000

    # Gates to perform. Single qubit gates are either inputted as string, or as matrix.
    # Single qubit gates ignore the control parameter.
    gate_array = [
        {"gate": "H", "target": 0},
        {"gate": "CNOT", "target": 1, "control": 0},
        {"gate": [[np.cos(np.pi / 10), np.sin(np.pi / 10) * (0. + 1.j)],
                  [np.sin(np.pi / 10) * (0. + 1.j), np.cos(np.pi / 10)]],
         "target": 2}
    ]

    result = simulate(qubits, gate_array, shots)

    # the following part is only about displaying the results nicely
    keys = list(result.keys())
    keys.sort()

    for i in keys:
        print(f'{i}: {result[i]}')

    # displays run time of the script
    print('\n')
    print(f'run time: {time.time() - start_time} seconds')
