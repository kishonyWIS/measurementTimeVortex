from color_code import *
from lattice import *

phys_err_rate_list = [0.005]#, 0.01, 0.015, 0.02, 0.025, 0.03]
shots = 1000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'

d = (4, 5)
num_vortexes = (0, 0)
print(f'Running for dx:{d[0]},dy:{d[1]}')

lat = HexagonalLatticeGidney(d)

simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                   logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                   detectors=detectors, draw=True, color_bonds_by_delay=False, csv_path='data/color_code_fractional_size.csv')
