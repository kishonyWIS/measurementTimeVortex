from color_code import *
from lattice import *

lat = HexagonalLatticeGidney((2,6))
logical_op_directions = ('y',)
detectors = ('Z',)
logical_operator_pauli_type = 'Z'
num_vortexes = (0, 0)  # (0,1)
reps_without_noise = 1
reps_with_noise = 1
shots = 10000000
phys_err_rate = 0.1#*np.sqrt(10)
noise_type = 'EM3_v2'#'EM3_v2'#'DEPOLARIZE1'

code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)
circ, _, _, num_logicals = code.get_circuit(reps=reps_with_noise + 2 * reps_without_noise, reps_without_noise=reps_without_noise,
                                            noise_model=get_noise_model(noise_type, phys_err_rate),
                                            logical_operator_pauli_type=logical_operator_pauli_type,
                                            logical_op_directions=logical_op_directions,
                                            detector_indexes=None, detector_args=None, draw=False,
                                            return_num_logical_qubits=True)

print(len(circ.shortest_graphlike_error()))

model = circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
sampler = circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)
perfect_syndrome_mask = np.all(syndrome == False, axis=1)
actual_observables_trivial_syndrome = actual_observables[perfect_syndrome_mask]

print(np.mean(actual_observables_trivial_syndrome))