import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt


def get_rep_code_circuit(rounds=25, d=9, before_round_data_depolarization=0.04, before_measure_flip_probability=0.01):
    # even qubits are data qubits, odd qubits are ancilla qubits, with periodic boundary conditions
    # add depolarization to the data qubits
    noisy_syndrome_measurement = stim.Circuit()
    n_qubits = d*2
    # depolarization on the data qubits
    for q in range(0, n_qubits, 2):
        noisy_syndrome_measurement.append_operation("DEPOLARIZE1", [q], before_round_data_depolarization)
    # cx gates between 2i and 2i+1
    for q in range(0, n_qubits, 2):
        noisy_syndrome_measurement.append_operation("CX", [q, q+1])
    # cx gates between 2i and 2i-1
    for q in range(0, n_qubits, 2):
        noisy_syndrome_measurement.append_operation("CX", [q, (q-1) % n_qubits])
    # x error on ancilla qubits
    for q in range(1, n_qubits, 2):
        noisy_syndrome_measurement.append_operation("X_ERROR", [q], before_measure_flip_probability)
    # measure and reset ancilla qubits
    for q in range(1, n_qubits, 2):
        noisy_syndrome_measurement.append_operation("MR", [q])

    circ = stim.Circuit()
    # first round
    circ += noisy_syndrome_measurement
    # add detectors
    for n_det in range(d):
        circ.append_operation("DETECTOR", [stim.target_rec(-d+n_det)], [0, n_det])
    for r in range(1, rounds):
        circ += noisy_syndrome_measurement
        for n_det in range(d):
            circ.append_operation("DETECTOR", [stim.target_rec(-d+n_det), stim.target_rec(-2*d+n_det)], [r, n_det])
    # measure the data qubits
    for q in range(0, n_qubits, 2):
        circ.append_operation("X_ERROR", [q], before_measure_flip_probability)
    for q in range(0, n_qubits, 2):
        circ.append_operation("M", [q])
    for n_det in range(d):
        circ.append_operation("DETECTOR", [stim.target_rec(-d+n_det), stim.target_rec(-d+(n_det+1)%d), stim.target_rec(-2*d+n_det)],
                              [rounds, n_det])
    circ.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-1)])
    return circ

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


class Gate:
    def __init__(self, name, qubits, time):
        self.name = name
        self.qubits = qubits
        self.time = time

    def __repr__(self):
        return f"Gate(name={self.name}, qubits={self.qubits}, time={self.time})"

def analyze_stim_circuit(circuit: stim.Circuit):
    num_qubits = circuit.num_qubits
    qubit_depths = [0] * num_qubits

    gates = []
    detectors = []
    observables = []

    measurement_to_gate = {}

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            op = instruction.name
            targets = [t.value for t in instruction.targets_copy() if t.is_qubit_target or t.is_measurement_record_target]

            if op in {"H", "CNOT", "CZ", "X", "Y", "Z", "RX", "RY", "RZ"}:
                gate_time = max(qubit_depths[q] for q in targets if q < num_qubits)
                for q in targets:
                    if q < num_qubits:
                        qubit_depths[q] = gate_time + 1

                gate = Gate(op, targets, gate_time)
                gates.append(gate)

            elif op in ["M", "MR"]:
                measurement_targets = [t.value for t in instruction.targets_copy() if t.is_qubit_target]
                gate_time = max(qubit_depths[q] for q in measurement_targets if q < num_qubits)
                for q in measurement_targets:
                    if q < num_qubits:
                        qubit_depths[q] = gate_time + 1

                gate = Gate(op, measurement_targets, gate_time)
                gates.append(gate)

                for idx in measurement_targets:
                    measurement_to_gate[idx] = gate

        elif isinstance(instruction, stim.CircuitRepeatBlock):
            block_analysis = analyze_stim_circuit(instruction.body_copy())
            gates.extend(block_analysis['gates'])
            detectors.extend(block_analysis['detectors'])
            observables.extend(block_analysis['observables'])

        elif instruction.name == "DETECTOR":
            detector_targets = [t.value for t in instruction.targets_copy() if t.is_measurement_record_target]
            linked_gates = [measurement_to_gate[idx] for idx in detector_targets if idx in measurement_to_gate]
            detectors.append((instruction.args_copy(), linked_gates))

        elif instruction.name == "OBSERVABLE_INCLUDE":
            observable_targets = [t.value for t in instruction.targets_copy() if t.is_measurement_record_target]
            linked_gates = [measurement_to_gate[idx] for idx in observable_targets if idx in measurement_to_gate]
            observables.append((instruction.args_copy(), linked_gates))

    return {
        "gates": gates,
        "detectors": detectors,
        "observables": observables
    }

def reconstruct_stim_circuit(data: dict) -> stim.Circuit:
    circuit = stim.Circuit()

    for gate in data["gates"]:
        circuit.append_operation(gate.name, gate.qubits)

    for detector_args, linked_gates in data["detectors"]:
        measurement_targets = [stim.target_rec(-gate.time - 1) for gate in linked_gates]
        circuit.append_operation("DETECTOR", measurement_targets, detector_args)

    for observable_args, linked_gates in data["observables"]:
        measurement_targets = [stim.target_rec(-gate.time - 1) for gate in linked_gates]
        circuit.append_operation("OBSERVABLE_INCLUDE", measurement_targets, observable_args)

    return circuit


if __name__ == '__main__':
    circuit = get_rep_code_circuit(d=4, rounds=3)
    data = analyze_stim_circuit(circuit)


    num_shots = 10_000
    for d in [3, 5, 7]:
        xs = []
        ys = []
        for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
            circuit = get_rep_code_circuit(d=d, before_round_data_depolarization=noise)
            num_errors_sampled = count_logical_errors(circuit, num_shots)
            xs.append(noise)
            ys.append(num_errors_sampled / num_shots)
        # print the graphlike distance
        print("d=", d, 'graphlike distance=', len(circuit.shortest_graphlike_error()))
        plt.plot(xs, ys, label="d=" + str(d))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()
