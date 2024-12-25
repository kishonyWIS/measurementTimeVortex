from color_code import *
from lattice import *
import pandas as pd
import ast

phys_err_rate_list = np.logspace(-4,-1,31)#[10:-7]
shots = 10000000
reps_without_noise = 1
noise_type = 'EM3_v2'  # 'parity_measurement_with_correlated_measurement_noise', 'DEPOLARIZE2', 'DEPOLARIZE1', 'Z_ERROR', 'SD6', 'EM3_v2'
logical_op_directions = ('x','y')
detectors = ('X',)
logical_operator_pauli_type = 'X'

# read the data from the csv file

# Custom converter function
def tuple_int_converter(value):
    if not value or pd.isna(value):  # Check for empty or NaN values
        return ()  # Return an empty tuple or another default value
    try:
        return tuple(map(int, ast.literal_eval(value)))
    except (ValueError, SyntaxError):
        # Handle cases where the value is not properly formatted
        raise ValueError(f"Invalid tuple format: {value}")

df = pd.read_csv('data/torus_configurations_without_vortices.csv', converters={'L1': tuple_int_converter, 'L2': tuple_int_converter})

unique_L1_L2_dist = set()
for irow, row in df.iterrows():
    L1, L2, dist = row['L1'], row['L2'], row['distance']
    if len(L1) == 0 or len(L2) == 0:
        continue
    unique_L1_L2_dist.add((L1, L2, dist))

# sort by abs(L1 x L2)
unique_L1_L2 = sorted(unique_L1_L2_dist, key=lambda x: abs(x[0][0]*x[1][1] - x[0][1]*x[1][0]))
# for L1, L2, dist in unique_L1_L2:
for L1, L2, dist in [((1, 1, 0), (2, -1, 0), 1), ((3, 0, 0), (0, 3, 0), 2), ((4, 1, 0), (1, -5, 0), 3), ((0, 6, 0), (6, 0, 0), 4),
                     ((3, 0, -6), (1, -5, 0), 3), ((1, 4, 12), (5, -1, 6), 4), ((4, 4, -18), (6, -3, -12), 5), ((1, 7, -12), (7, 1, 6), 6)]:

    num_vortexes = (int(np.round(-L1[-1]/6)), int(np.round(-L2[-1]/6)))
    print(f'L1:{L1}, L2:{L2}, num_vortices:{num_vortexes}, distance:{dist}')
    lat = HexagonalLattice(L1[:-1], L2[:-1])
    simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=False, color_bonds_by_delay=False, csv_path='data/data_threshold.csv')