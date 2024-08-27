import dataclasses
from functools import partial
from typing import Optional, Dict, Set, Tuple, Callable

import stim

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    idle: float
    measure_reset_idle: float
    noisy_gates: Dict[str, float]
    any_clifford_1: Optional[float] = None
    any_clifford_2: Optional[float] = None
    parity_measurement_noise: Callable[[stim.GateTarget, stim.GateTarget, int, float], stim.Circuit] = None

    @staticmethod
    def SD6(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=0,
            idle=0,
            measure_reset_idle=0,
            parity_measurement_noise=parity_measurement_with_uncorrelated_measurement_noise,
            noisy_gates={
                "MPP": p,
            },
        )

    @staticmethod
    def EM3_v2(p: float) -> 'NoiseModel':
        """EM3 with measurement flip errors correlated with measurement target depolarization error."""
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=0,
            measure_reset_idle=0,
            parity_measurement_noise=parity_measurement_with_correlated_measurement_noise,
            noisy_gates={
                "MPP": p,
            },
        )

    @staticmethod
    def Noise_before_parity_meas(p: float, noise_type:str = 'DEPOLARIZE2') -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=0,
            measure_reset_idle=0,
            parity_measurement_noise=partial(parity_measurement_with_noise_before, noise_type=noise_type),
            noisy_gates={
                "MPP": p,
            },
        )


    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> stim.Circuit:
        result = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name == "MPP":
                assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
                assert args == [] or args == [0]

                for k in range(0, len(targets), 3):
                    result += self.parity_measurement_noise(
                        t1=targets[k],
                        t2=targets[k + 2],
                        ancilla=ancilla,
                        p=p)
            else:
                raise NotImplementedError(repr(op))
        else:
            result.append_operation(op.name, targets, args)
        return result

    def noisy_circuit(self, circuit: stim.Circuit) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits

        for op in circuit:
            result += self.noisy_op(op, self.noisy_gates.get(op.name, 0), ancilla)

        return result


def mix_probability_to_independent_component_probability(mix_probability: float, n: float) -> float:
    """Converts the probability of applying a full mixing channel to independent component probabilities.

    If each component is applied independently with the returned component probability, the overall effect
    is identical to, with probability `mix_probability`, uniformly picking one of the components to apply.

    Not that, unlike in other places in the code, the all-identity case is one of the components that can
    be picked when applying the error case.
    """
    return 0.5 - 0.5 * (1 - mix_probability) ** (1 / 2 ** (n - 1))


def parity_measurement_with_correlated_measurement_noise(
        *,
        t1: stim.GateTarget,
        t2: stim.GateTarget,
        ancilla: int,
        p: float) -> stim.Circuit:
    """Performs a noisy parity measurement.

    With probability mix_probability, applies a random element from

        {I1,X1,Y1,Z1}*{I2,X2,Y2,Z2}*{no flip, flip}

    Note that, unlike in other places in the code, the all-identity term is one of the possible
    samples when the error occurs.
    """

    ind_p = mix_probability_to_independent_component_probability(p, 5)

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets
    # with last element corresponding to measure qubit.
    circuit = stim.Circuit()
    circuit.append_operation('R', [ancilla])
    if t1.is_x_target:
        circuit.append_operation('XCX', [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append_operation('YCX', [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append_operation('ZCX', [t1.value, ancilla])
    if t2.is_x_target:
        circuit.append_operation('XCX', [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append_operation('YCX', [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append_operation('ZCX', [t2.value, ancilla])

    first_targets = ["I", stim.target_x(t1.value), stim.target_y(t1.value), stim.target_z(t1.value)]
    second_targets = ["I", stim.target_x(t2.value), stim.target_y(t2.value), stim.target_z(t2.value)]
    measure_targets = ["I", stim.target_x(ancilla)]

    errors = []
    for first_target in first_targets:
        for second_target in second_targets:
            for measure_target in measure_targets:
                error = []
                if first_target != "I":
                    error.append(first_target)
                if second_target != "I":
                    error.append(second_target)
                if measure_target != "I":
                    error.append(measure_target)

                if len(error) > 0:
                    errors.append(error)

    for error in errors:
        circuit.append_operation("CORRELATED_ERROR", error, ind_p)

    circuit.append_operation('M', [ancilla])

    return circuit


def parity_measurement_with_uncorrelated_measurement_noise(
        *,
        t1: stim.GateTarget,
        t2: stim.GateTarget,
        ancilla: int,
        p: float) -> stim.Circuit:

    circuit = stim.Circuit()
    circuit.append_operation('R', [ancilla])
    circuit.append_operation('X_ERROR', [ancilla], p)

    if t1.is_x_target:
        circuit.append_operation('XCX', [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append_operation('YCX', [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append_operation('ZCX', [t1.value, ancilla])
    circuit.append_operation('DEPOLARIZE2', [t1.value, ancilla], p)

    if t2.is_x_target:
        circuit.append_operation('XCX', [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append_operation('YCX', [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append_operation('ZCX', [t2.value, ancilla])
    circuit.append_operation('DEPOLARIZE2', [t2.value, ancilla], p)

    circuit.append_operation('X_ERROR', [ancilla], p)
    circuit.append_operation('M', [ancilla])

    return circuit

def parity_measurement_with_noise_before(
        *,
        t1: stim.GateTarget,
        t2: stim.GateTarget,
        ancilla: int,
        p: float,
        noise_type: str = 'DEPOLARIZE2') -> stim.Circuit:

    circuit = stim.Circuit()
    circuit.append_operation('R', [ancilla])
    circuit.append_operation(noise_type, [t1.value, t2.value], p)

    if t1.is_x_target:
        circuit.append_operation('XCX', [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append_operation('YCX', [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append_operation('ZCX', [t1.value, ancilla])

    if t2.is_x_target:
        circuit.append_operation('XCX', [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append_operation('YCX', [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append_operation('ZCX', [t2.value, ancilla])

    # circuit.append_operation('X_ERROR', [ancilla], p)
    circuit.append_operation('M', [ancilla])
    # circuit.append_operation(noise_type, [t1.value, t2.value], p)

    return circuit


def get_noise_model(noise_type: str, physical_error_rate: float) -> NoiseModel:
    if noise_type == 'SD6':
        return NoiseModel.SD6(physical_error_rate)
    if noise_type == 'EM3_v2':
        return NoiseModel.EM3_v2(physical_error_rate)
    if noise_type == 'DEPOLARIZE2':
        return NoiseModel.Noise_before_parity_meas(physical_error_rate, noise_type='DEPOLARIZE2')
    if noise_type == 'DEPOLARIZE1':
        return NoiseModel.Noise_before_parity_meas(physical_error_rate, noise_type='DEPOLARIZE1')
    raise NotImplementedError(f"Unknown noise type: {noise_type}")
