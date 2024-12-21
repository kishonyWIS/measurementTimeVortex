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

df = pd.read_csv('data/best_torus_for_distance.csv', converters={'L1': tuple_int_converter, 'L2': tuple_int_converter, 'L1_no_vortex': tuple_int_converter, 'L2_no_vortex': tuple_int_converter})
# columns = ['distance', 'num_qubits', 'L1', 'L2', 'num_qubits_no_vortex', 'L1_no_vortex', 'L2_no_vortex']
# iterate over the rows of the dataframe
# simulate L1, L2 and then L1_no_vortex, L2_no_vortex

for irow, row in df.iterrows():
    for L1, L2 in [(row['L1'], row['L2']), (row['L1_no_vortex'], row['L2_no_vortex'])]:
        L1= (3, 0, -6)
        L2= (1, -5, 0)
        print(f'L1:{L1}, L2:{L2}, distance:{row["distance"]}')
        lat = HexagonalLattice(L1[:-1], L2[:-1])
        num_vortexes = (L1[-1], L2[-1])
        simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat, get_reps_by_graph_dist=True,
                           detectors=detectors, draw=True, color_bonds_by_delay=True, csv_path='data/data_threshold_generic_lattice.csv')