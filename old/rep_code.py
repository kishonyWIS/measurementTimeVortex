import stim
import numpy as np
from matplotlib import pyplot as plt
import pymatching


def get_circuit(reps=10, d=3, bit_flip_rate=None):
    circ = stim.Circuit()

    i_meas = 0
    for rep in range(reps):
        if bit_flip_rate is not None and rep >= 2:
            circ.append("X_ERROR", range(d), bit_flip_rate)
        for i in range(d-1):
            qubits = [i, i+1]
            circ.append('MZZ', qubits)
            i_meas += 1
            if rep > 1:
                record_targets = [stim.target_rec(-1), stim.target_rec(-1 - (d-1))]
                circ.append_operation("DETECTOR", record_targets, [i, rep])

    # Finish circuit with data measurements.
    qubits_for_observable = range(d)
    circ.append_operation("M", qubits_for_observable)
    circ.append_operation("OBSERVABLE_INCLUDE",
                                  [stim.target_rec(i - len(qubits_for_observable)) for i in range(len(qubits_for_observable))],
                                  0)
    return circ

circ = get_circuit(d=21, reps=30, bit_flip_rate=0.1)

model = circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

print(circ)

sampler = circ.compile_detector_sampler()
shots = 100000
syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

predicted_observables = matching.decode_batch(syndrome)
num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print("logical error_rate", num_errors/shots)
