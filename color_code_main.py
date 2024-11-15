from color_code import *
from lattice import *

N_list = [9,16,25]
phys_err_rate_list = [0.005]#, 0.01, 0.015, 0.02, 0.025, 0.03]
shots = 1000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'

for N in N_list:
    for Lx in range(1, N+1):
        if N % Lx != 0:
            continue
        Ly = N // Lx
        d = (2*Lx, 3*Ly)
        print(f'Running for dx:{d[0]},dy:{d[1]}')

        vx_list = np.arange(2 * Lx)
        vy_list = np.arange(-Ly + 1, 2 * Ly)
        for num_vortexes in [(vx, vy) for vx in vx_list for vy in vy_list]:
            lat = HexagonalLatticeGidney(d)
            try:
                simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                                   logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                                   detectors=detectors, draw=False, color_bonds_by_delay=False)
            except:
                print(f'Failed to simulate for dx:{d[0]},dy:{d[1]} and num_vortexes={num_vortexes}')
                continue