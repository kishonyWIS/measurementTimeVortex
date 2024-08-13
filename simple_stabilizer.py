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
        return self