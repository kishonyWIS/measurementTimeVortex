from color_code import *
from lattice import *

phys_err_rate_list = np.logspace(-4,-1,31)#[10:-7]
shots = 1000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'

L_vx_vy_list = [(2,1,2), (3,1,4), (4,1,6), (4,2,4), (4,3,3)]

for L, vx, vy in L_vx_vy_list:
    d = (2*L, 3*L)
    num_vortexes = (vx, vy)
    # vx_list = np.arange(2 * L)
    # vy_list = np.arange(-L + 1, 2 * L)
    # for num_vortexes in [(vx, vy) for vx in vx_list for vy in vy_list]:
    print(f'Running for dx:{d[0]},dy:{d[1]}, num_vortexes={num_vortexes}')
    lat = HexagonalLatticeGidney(d)
    try:
        simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=False, color_bonds_by_delay=False, csv_path='data/data_threshold_generic_lattice.csv')
    except:
        print(f'Failed to simulate for dx:{d[0]},dy:{d[1]} and num_vortexes={num_vortexes}')
        continue