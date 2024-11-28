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


detectors = ('Z', 'X')
logical_operator_pauli_type = 'Z'
noise_type = 'EM3_v2'
lattice_type = HexagonalLatticeGidney
reps_without_noise = 1
draw = False
Lx = 2
Ly = 2
dx = Lx * 2
dy = Ly * 3
for logical_op_directions in [('y',), ('x',), ('x', 'y')]:
    vx_list = np.arange(2*Lx)
    vy_list = np.arange(-Ly+1, 2*Ly)
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
plt.show()