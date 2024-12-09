import stim

from color_code import *


def idle_qubits(circ: stim.Circuit, phys_err_rate: float, add_depolarization: bool = True, return_idle_time: bool = False):
    qubit_depths = np.zeros(circ.num_qubits, dtype=int)
    qubit_idle_time = np.zeros(circ.num_qubits, dtype=int)
    new_circ = stim.Circuit()
    for op in circ:
        targets = op.targets_copy()
        if op.name in ["DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS"]:
            new_circ.append_operation(op.name, targets, op.gate_args_copy())
            continue
        elif stim.gate_data(op.name).is_single_qubit_gate:
            num_targets_per_gate = 1
        elif stim.gate_data(op.name).is_two_qubit_gate:
            num_targets_per_gate = 2
        elif op.name == 'MPP':
            num_targets_per_gate = 3
        else:
            raise ValueError(f'Unknown gate: {op.name}')
        # split into separate gates
        for i in range(len(targets) // num_targets_per_gate):
            sub_targets = targets[i*num_targets_per_gate:(i+1)*num_targets_per_gate]
            qubits = [tar.qubit_value for tar in sub_targets]
            qubits = [q for q in qubits if q is not None]
            args = op.gate_args_copy()
            op_depth = max(qubit_depths[qubits]) + 1
            for q in qubits:
                if qubit_depths[q] == 0:
                    qubit_depths[q] = op_depth-1
                for _ in range(op_depth - qubit_depths[q] - 1):
                    if add_depolarization:
                        new_circ.append_operation('DEPOLARIZE1', [q], phys_err_rate)
                    if return_idle_time:
                        qubit_idle_time[q] += 1
                qubit_depths[q] = op_depth
            new_circ.append_operation(op.name, sub_targets, args)
    print('depth: ', max(qubit_depths))
    return new_circ, qubit_idle_time if return_idle_time else new_circ


if __name__ == '__main__':
    circ = stim.Circuit()
    circ.append_operation('H', [0])
    circ.append_operation('CX', [0, 1])
    circ.append_operation('CX', [1, 2])
    circ.append_operation('M', [0])
    print(idle_qubits(circ, 0.01, add_depolarization=True, return_idle_time=True))

    # small MPP example
    circ = stim.Circuit()
    circ.append_operation('MPP', [stim.target_x(0), stim.target_combiner(), stim.target_z(1)])
    circ.append_operation('MPP', [stim.target_x(1), stim.target_combiner(), stim.target_z(2)])
    circ.append_operation('MPP', [stim.target_x(2), stim.target_combiner(), stim.target_z(3)])
    circ.append_operation('MPP', [stim.target_x(1), stim.target_combiner(), stim.target_z(2)])
    circ.append_operation('MPP', [stim.target_x(0), stim.target_combiner(), stim.target_z(1)])
    # circ.append_operation('MPP', [stim.target_x(1), stim.target_combiner(), stim.target_z(3)])

    print(idle_qubits(circ, 0.01, add_depolarization=True, return_idle_time=True))

    # test on color code circuit
    lat = HexagonalLatticeGidney((4,6))
    reps_without_noise = 5
    reps_with_noise = 0
    code = FloquetCode(lat, num_vortexes=(1,0), detectors=('X',))
    circ, _, _ = code.get_circuit(reps=reps_with_noise+2*reps_without_noise,
                                  reps_without_noise=reps_without_noise,
                                  noise_model = get_noise_model('DEPOLARIZE1', 0.1),
                                  logical_operator_pauli_type='X',
                                  logical_op_directions=('x','y'),
                                  detector_indexes=None, detector_args=None,
                                  draw=False, return_num_logical_qubits=False)

    # keep only MPP gates
    new_circ = stim.Circuit()
    for op in circ:
        if op.name == 'MPP':
            new_circ.append_operation(op.name, op.targets_copy(), op.gate_args_copy())
    circ = new_circ.without_noise()

    new_circ, qubit_idle_time = idle_qubits(circ, 0.01, add_depolarization=True, return_idle_time=True)
    print(qubit_idle_time)