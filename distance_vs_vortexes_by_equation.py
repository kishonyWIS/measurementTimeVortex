import numpy as np
from qiskit.pulse import num_qubits

from color_code import FloquetCode
from lattice import *
from noise import get_noise_model
from matplotlib import pyplot as plt
from idle import idle_qubits
import stim
import pandas as pd
import os

def is_allowed_vortex_numbers(Lx, Ly, vx, vy):
    condition1 = -vy - (3*Ly - 2)/(2*Lx - 1) * abs(vx) + 2*Ly - 1#-vy - (3*Ly - 2)/(2*Lx - 1) * abs(vx) + 2*Ly - 1
    condition2 = vy + Ly
    print(f'condition1={condition1}, condition2={condition2}')
    return condition1>=0 and condition2>0

def compute_dist(Lx, Ly, vx, vy, logical_direction):
    if logical_direction == 'y': # error in x direction
        return 2*Lx + abs(vx)
    elif logical_direction == 'x': # error in y direction
        return 2*Ly + np.ceil(3/4 * abs(vy) - 1/4 * vy)

# for logical_op_directions in [('edge_to_hole',), ('around_hole',)]:
#     if logical_op_directions[0] == 'edge_to_hole':
#         detectors = ('X',)
#         logical_operator_pauli_type = 'X'
#     elif logical_op_directions[0] == 'around_hole':
#         detectors = ('Z',)
#         logical_operator_pauli_type = 'Z'
#     else:
#         raise ValueError(f'Unknown logical_op_directions: {logical_op_directions}')


for logical_op_directions in [('y',), ('x',)]:
    if logical_op_directions[0] == 'x':
        detectors = ('Z',)
        logical_operator_pauli_type = 'Z'
    elif logical_op_directions[0] == 'y':
        detectors = ('X',)
        logical_operator_pauli_type = 'X'
    else:
        raise ValueError(f'Unknown logical_op_directions: {logical_op_directions}')

    noise_type = 'EM3_v2'
    lattice_type = HexagonalLatticeGidney
    reps_without_noise = 1
    draw = False

    for N in range(1, 41):
        for Lx in range(1, N+1):
            if N % Lx != 0:
                continue
            Ly = N // Lx
            # for Lx in [1,2,3,4]:
            #     for Ly in [1,2,3,4]:
            dx = Lx*2
            dy = Ly*3
            vx_list = np.arange(2*Lx)
            vy_list = np.arange(-Ly+1, 2*Ly)
            dists = np.zeros((len(vx_list), len(vy_list)))
            dists[:] = np.nan

            for i, vx in enumerate(vx_list):
                for j, vy in enumerate(vy_list):
                    num_vortexes = (vx, vy)

                    lat = lattice_type((dx, dy))

                    # try:
                    if is_allowed_vortex_numbers(Lx, Ly, vx, vy):
                        dist = compute_dist(Lx, Ly, vx, vy, logical_op_directions[0])
                        n_qubits = dx*dy*4
                        df = pd.DataFrame({'Lx': [Lx], 'Ly': [Ly], 'vx': [vx], 'vy': [vy], 'logical_op_direction': logical_op_directions[0], 'dist': [dist], 'n_qubits': [n_qubits]})
                        # add header if file does not exist
                        df.to_csv('distance_vs_vortices_by_equation.csv', mode='a', header=not os.path.exists('distance_vs_vortices_by_equation.csv'), index=False)
