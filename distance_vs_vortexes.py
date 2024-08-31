import numpy as np

from color_code import FloquetCode
from lattice import *
from noise import get_noise_model
from matplotlib import pyplot as plt
from idle import idle_qubits
import stim

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

    noise_type = 'DEPOLARIZE1'
    lattice_type = HexagonalLatticeGidney
    reps_without_noise = 1
    draw = True


    for dx, dy in [(6,9)]:#zip([3,6,9,12], [3,6,9,12]):#zip([2,4,6,8,10], [3,6,9,12,15]):
        vx_list = [1]#[-5,-4,-3,-2,-1,0,1,2,3,4,5]
        vy_list = [0]#[-5,-4,-3,-2,-1,0,1,2,3,4,5]
        dists = np.zeros((len(vx_list), len(vy_list)))
        dists[:] = np.nan

        for i, vx in enumerate(vx_list):
            for j, vy in enumerate(vy_list):
                num_vortexes = (vx, vy)

                lat = lattice_type((dx, dy))

                # try:
                code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)
                circ, _, _, num_logicals = code.get_circuit(reps=4+2*reps_without_noise, reps_without_noise=reps_without_noise,
                    noise_model = get_noise_model(noise_type, 0.1),
                    logical_operator_pauli_type=logical_operator_pauli_type,
                    logical_op_directions=logical_op_directions,
                    detector_indexes=None, detector_args=None, draw=draw, return_num_logical_qubits=True)
                print(idle_qubits(circ, phys_err_rate=0.1, add_depolarization=True, return_idle_time=True))
                dist = len(circ.shortest_graphlike_error())
                print(f'dist={dist}')
                if num_logicals != 2:
                    continue
                dists[i,j] = dist
                # except:
                #     print(f'Failed to simulate for dx:{dx},dy:{dy} and num_vortexes={num_vortexes}')

        plt.figure()
        plt.pcolor(dists.T)
        plt.xticks(np.arange(len(vx_list))+0.5, vx_list)
        plt.yticks(np.arange(len(vy_list))+0.5, vy_list)
        plt.xlabel('vx')
        plt.ylabel('vy')
        plt.colorbar()
        plt.savefig(f'figures/distance_vs_vortexes_lattice_{lattice_type.__name__}_logical_direction_{logical_op_directions[0]}_Lx{dx}_Ly{dy}.pdf')
