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
        detectors = ('Z')
        logical_operator_pauli_type = 'Z'
    elif logical_op_directions[0] == 'y':
        detectors = ('X')
        logical_operator_pauli_type = 'X'
    else:
        raise ValueError(f'Unknown logical_op_directions: {logical_op_directions}')

    noise_type = 'EM3_v2'
    lattice_type = HexagonalLatticeGidney
    reps_without_noise = 1
    draw = False


    for N in range(1, 10):
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
            # vx_list = [0,1,2,3,4,5]#[2]#[-5,-4,-3,-2,-1,0,1,2,3,4,5]
            # vy_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5]#[0]#[-5,-4,-3,-2,-1,0,1,2,3,4,5]
            dists = np.zeros((len(vx_list), len(vy_list)))
            dists[:] = np.nan

            for i, vx in enumerate(vx_list):
                for j, vy in enumerate(vy_list):
                    num_vortexes = (vx, vy)

                    lat = lattice_type((dx, dy))

                    # try:
                    code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)
                    circ, _, _, num_logicals = code.get_circuit(reps=2+2*reps_without_noise, reps_without_noise=reps_without_noise,
                        noise_model = get_noise_model(noise_type, 0.1),
                        logical_operator_pauli_type=logical_operator_pauli_type,
                        logical_op_directions=logical_op_directions,
                        detector_indexes=None, detector_args=None, draw=draw, return_num_logical_qubits=True)
                    # print(idle_qubits(circ, phys_err_rate=0.1, add_depolarization=True, return_idle_time=True))
                    try:
                        dist = len(circ.shortest_graphlike_error())
                        print(f'dist={dist}')
                        if num_logicals != 2:
                            continue
                        dists[i,j] = dist
                        # save to csv with pandas Lx, Ly, vx, vy, logical_op_directions[0], dist, num_qubits
                        n_qubits = dx*dy*4
                        df = pd.DataFrame({'Lx': [Lx], 'Ly': [Ly], 'vx': [vx], 'vy': [vy], 'logical_op_direction': logical_op_directions[0], 'dist': [dist], 'n_qubits': [n_qubits]})
                        # add header if file does not exist
                        df.to_csv('distance_vs_vortices.csv', mode='a', header=not os.path.exists('distance_vs_vortices.csv'), index=False)
                    except:
                        print(f'Failed to simulate for dx:{dx},dy:{dy} and num_vortexes={num_vortexes}')

            plt.figure()
            plt.pcolor(dists.T)
            plt.xticks(np.arange(len(vx_list))+0.5, vx_list)
            plt.yticks(np.arange(len(vy_list))+0.5, vy_list)
            plt.xlabel('vx')
            plt.ylabel('vy')
            plt.colorbar()
            plt.savefig(f'figures/distance_vs_vortexes_lattice_{lattice_type.__name__}_noise_{noise_type}_logical_direction_{logical_op_directions[0]}_Lx{dx}_Ly{dy}.pdf')
