from color_code import *
from lattice import *

L_list = [5]#[1,2,3,4]
phys_err_rate_list = np.logspace(-4,-1,31)[23:]
shots = 1000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'

for L in L_list:
    d = (2*L, 3*L)
    print(f'Running for dx:{d[0]},dy:{d[1]}')
    for num_vortexes in [(0,0)]: #(0,0),(0,2*L-1)
        lat = HexagonalLatticeGidney(d)
        simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=False, color_bonds_by_delay=False, csv_path='data/data_threshold.csv')
