from color_code import *

d_list = [6, 9]
phys_err_rate_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
shots = 100000
reps_without_noise = 2
noise_type = 'parity_measurement_with_correlated_measurement_noise'  # 'DEPOLARIZE2'
boundary_conditions = ('periodic', 'periodic')
logical_op_directions = ['x']

for num_vortexes in [(1,0)]:  # , (1, 0)
    for logical_operator_pauli_type in ['X']:
        for id, d in enumerate(d_list):
            dx = d
            dy = d
            simulate_vs_noise_rate(dx, dy, phys_err_rate_list, shots, reps_without_noise,
                                   noise_type, logical_operator_pauli_type,
                                   logical_op_directions, boundary_conditions, num_vortexes,
                                   get_reps_by_graph_dist=True)