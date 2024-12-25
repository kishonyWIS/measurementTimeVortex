from color_code import *
from lattice import *
import pandas as pd
import ast

phys_err_rate_list = np.logspace(-4,-1,31)#[10:-7]
shots = 1000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'


for L1, L2 in [((0, 6, -6), (6, 3, 6)), ((0, 6, -6), (6, 0, 6)), ((0, 6, -6), (6, 0, -12)), ((5, 2, 6), (3, -6, -12)), ((4, 4, 18), (3, -6, -12)), ((2, 5, -6), (6, -3, -12)), ((4, 4, -18), (6, -3, -12)), ((4, 4, -6), (3, -6, 12)), ((3, 3, -12), (6, -6, -6)), ((4, 4, 6), (6, -3, 12)), ((3, 3, 12), (6, -6, -6)), ((0, 6, 12), (6, 0, 6)), ((6, 0, 6), (3, 6, 18)), ((0, 6, -6), (6, 3, -18)), ((6, 0, 6), (0, 6, -6)), ((2, 5, -6), (6, -3, 12)), ((3, 3, -12), (5, -7, -6)), ((3, 3, 12), (5, -7, -6)), ((6, 0, -12), (0, 6, -6)), ((6, 0, 6), (3, 6, -6)), ((5, 2, 6), (3, -6, 12)), ((3, 3, -12), (7, -5, -6)), ((6, 0, 6), (0, 6, 12)), ((3, 3, 12), (7, -5, -6))]:
    num_vortexes = (int(np.round(-L1[-1]/6)), int(np.round(-L2[-1]/6)))
    print(f'L1:{L1}, L2:{L2}, num_vortices:{num_vortexes}')
    lat = HexagonalLattice(L1[:-1], L2[:-1])
    simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=False, color_bonds_by_delay=False, csv_path='data/data_threshold_same_N_D.csv')