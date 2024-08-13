import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from stabilizer import StabilizerGroup
from typing import Optional
from matplotlib import pyplot as plt
import stim


class StabilizerMeasurementSequence:
    def __init__(self, measurement_sequence: list[Pauli]):
        self.measurement_sequence = measurement_sequence
        self.stabilizer = StabilizerGroup(StabilizerGroup('I'*len(measurement_sequence[0].pauli))[:0])
        self.current_measurement_index = 0

    def __iter__(self):
        return self

    def __next__(self, verb=True):
        pauli_to_measure = self.measurement_sequence[self.current_measurement_index]
        if verb:
            print('measuring: ',pauli_to_measure)
        self.stabilizer = self.stabilizer.measure_pauli(pauli_to_measure)
        if self.current_measurement_index == len(self.measurement_sequence) - 1:
            self.current_measurement_index = 0
        else:
            self.current_measurement_index += 1
        return self.stabilizer

    @classmethod
    def from_stim_circuit(cls, stim_circuit: stim.Circuit):
        return cls([Pauli.from_label(stim_circuit[p].to_pauli_string()) for p in range(stim_circuit.num_moments)])


if __name__ == '__main__':
    circ = stim.Circuit()
    circ.append('MZZ', [0, 1])
    circ.append('MZZ', [1, 2])
    measurement_sequence = StabilizerMeasurementSequence.from_stim_circuit(circ)
    print()