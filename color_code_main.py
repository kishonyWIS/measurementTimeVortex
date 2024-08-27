from color_code import *
from lattice import *

d_list = [(2,3)]
phys_err_rate_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
shots = 100000
reps_without_noise = 2
noise_type = 'DEPOLARIZE2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x')
detectors = ('X',)
logical_operator_pauli_type = 'X'

for num_vortexes in product([0,1],[0,1]):  # , (1, 0)
    for d in d_list:
        lat = HexagonalLatticeGidney(d)
        # try:
        simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=True)
        # except:
        #     print(f'Failed to simulate for dx:{d[0]},dy:{d[1]} and num_vortexes={num_vortexes}')
        #     continue