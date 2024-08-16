from copy import copy

import numpy as np
from qiskit.quantum_info import Pauli
from typing import Union

class PauliMeasurement:
    def __init__(self, pauli: Pauli, index: Union[int, list[int]]):
        self.pauli = pauli
        if isinstance(index, list):
            self.index = index
        else:
            self.index = [index]

    def __mul__(self, other):
        pauli = self.pauli.compose(other.pauli)
        # the index is the union of the indices minus the intersection
        index = list(set(self.index) | set(other.index))
        return PauliMeasurement(pauli, index)

class StabilizerGroup:
    def __init__(self, paulis:list[PauliMeasurement] = None):
        if paulis is None:
            paulis = []
        self.paulis = paulis

    def x_z(self):
        return np.array([np.concatenate((p.pauli.x, p.pauli.z), axis=0) for p in self.paulis])

    def measure_pauli(self, pauli: PauliMeasurement):
        # assuming the new measurement is not already in the stabilizer
        # find all the stabilizers that anti-commute with the new measurement
        # and multiply them by the first stabilizer that anti-commutes with the new measurement
        anti_commuting_index = [i for i, p in enumerate(self.paulis) if p.pauli.anticommutes(pauli.pauli)]
        if len(anti_commuting_index) > 1:
            for i in anti_commuting_index[1:]:
                self.paulis[i] = self.paulis[i] * self.paulis[anti_commuting_index[0]]
        # remove the first stabilizer that anti-commutes with the new measurement
        if len(anti_commuting_index) > 0:
            self.paulis.pop(anti_commuting_index[0])
        # add the new measurement
        self.paulis.append(pauli)

        # reduce the stabilizer by gaussian elimination
        pivot = 0
        for i in list(range(2 * self.paulis[0].pauli.num_qubits))[::(-1 if reversed else 1)]:
            for j in range(pivot, len(self.paulis)):
                if self.x_z()[j][i] != 0:
                    temp_row = copy(self.paulis[j])
                    self.paulis[j] = self.paulis[pivot]
                    self.paulis[pivot] = temp_row
                    # if j == len(self) - 1:
                    #     continue
                    for jj in range(len(self.paulis)):
                        if jj == pivot:
                            continue
                        if self.x_z()[jj][i] != 0:
                            self.paulis[jj].pauli = self.paulis[jj].pauli.dot(self.paulis[pivot].pauli)
                    pivot += 1
                    break
        self.paulis = self.paulis[:pivot]
        # remove rows after the pivot
        return self