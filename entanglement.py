import stim
import galois
import numpy as np

def circuit_to_tableau(circuit: stim.Circuit) -> stim.Tableau:
    s = stim.TableauSimulator()
    s.do_circuit(circuit)
    return s.current_inverse_tableau() ** -1

def entanglement_entropy(circ: stim.Circuit, qubits: list[int]) -> int:
    tableau = circuit_to_tableau(circ)
    tab_np = tableau.to_numpy()
    x_z_trucated = np.concatenate((tab_np[2][:, qubits], tab_np[3][:, qubits]), axis=1)
    # calculate the rank of the x_z matrix over the field with 2 elements
    x_z_gf2 = galois.GF(2)(x_z_trucated.astype(int))
    return np.linalg.matrix_rank(x_z_gf2) - len(qubits)

def num_logical_qubits(circ, qubits: list[int]) -> int:
    circ_from_maximally_mixed = stim.Circuit()
    for q in qubits:
        circ_from_maximally_mixed.append_operation('H', [q])
        circ_from_maximally_mixed.append_operation('CNOT', [q, q+circ.num_qubits])
    circ_from_maximally_mixed += circ
    return entanglement_entropy(circ_from_maximally_mixed, qubits)

if __name__ == '__main__':
    # perpare the repetition code
    circ = stim.Circuit()
    circ.append_operation('MZZ', [0, 1])
    circ.append_operation('MZZ', [1, 2])

    print(num_logical_qubits(circ, qubits=range(circ.num_qubits)))