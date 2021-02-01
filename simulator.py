import numpy as np
from state import *
from gate import *
from measurement import *


def simulate(num_qubits: int, gate_array: list, shots: int) -> dict:
    """Simulate quantum circuit with given number of qubits,
    given gate array and given number of shots to run the simulation. """
    gate_dict = {
        "X": gate_x(),
        "Y": gate_y(),
        "Z": gate_z(),
        "H": gate_h(),
    }

    state = get_initial_state(num_qubits)

    for gate_spec in gate_array:
        if gate_spec["gate"] == "CNOT":
            full_unitary = get_cnot_unitary(num_qubits, gate_spec["control"], gate_spec["target"])
        else:
            if isinstance(gate_spec["gate"], str):
                if gate_spec["gate"] in gate_dict:
                    unitary = gate_dict[gate_spec["gate"]]
                else:
                    raise ValueError(
                        f'Gate {gate_spec["gate"]} not defined. Available options are {", ".join(gate_dict.keys())}.')
            else:
                unitary = np.array(gate_spec["gate"])

            full_unitary = get_single_qubit_unitary(num_qubits, unitary, gate_spec["target"])

        state = np.dot(state, full_unitary)

    results = {}

    for i in range(shots):
        result = get_measurement_outcome(state)
        if result in results:
            results[result] += 1
        else:
            results[result] = 1

    return results
