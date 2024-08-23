from color_code import *
from geometry import *

d_list = [(6,6)]
phys_err_rate_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
shots = 100000
reps_without_noise = 2
noise_type = 'DEPOLARIZE2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
boundary_conditions = ('periodic', 'periodic')
geometry = SymmetricTorus
logical_op_directions = ('y')
detectors = ('X',)

for num_vortexes in [(0,0)]:  # , (1, 0)
    for logical_operator_pauli_type in ['X']:
        for id, d in enumerate(d_list):
            dx = d[0]
            dy = d[1]
            try:
                simulate_vs_noise_rate(dx, dy, phys_err_rate_list, shots, reps_without_noise,
                                       noise_type, logical_operator_pauli_type,
                                       logical_op_directions, boundary_conditions, num_vortexes,
                                       get_reps_by_graph_dist=True, geometry=geometry, detectors=detectors)
            except:
                print(f'Failed to simulate for dx:{dx},dy:{dy} and num_vortexes={num_vortexes}')
                continue