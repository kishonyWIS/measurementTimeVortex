from color_code import FloquetCode
from lattice import *
from noise import get_noise_model
from matplotlib import pyplot as plt

logical_op_directions = ('x',)
reps_without_noise = 2
if logical_op_directions[0] == 'x':
    detectors = ('Z',)
    logical_operator_pauli_type = 'Z'
else:
    detectors = ('X',)
    logical_operator_pauli_type = 'X'
noise_type = 'DEPOLARIZE1'
draw = False

# num_vortexes = (0, 0)
# dx_list, dy_list = [2,4,6], [3,6,9] # dx_list, dy_list = [3,6,9], [3,6,9]
# dists = np.zeros((len(dx_list), len(dy_list)))
# for i, dx in enumerate(dx_list):
#     for j, dy in enumerate(dy_list):

dx = 4
dy = 6
vx_list = [-1,0,1]
vy_list = [-1,0,1]
dists = np.zeros((len(vx_list), len(vy_list)))

for i, vx in enumerate(vx_list):
    for j, vy in enumerate(vy_list):
        num_vortexes = (vx, vy)
        lat = HexagonalLatticeGidney((dx, dy))

        try:
            code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)
            circ, _, _ = code.get_circuit(reps=1+2*reps_without_noise, reps_without_noise=reps_without_noise,
                noise_model = get_noise_model(noise_type, 0.1),
                logical_operator_pauli_type=logical_operator_pauli_type,
                logical_op_directions=logical_op_directions,
                detector_indexes=None, detector_args=None, draw=draw)
            dist = len(circ.shortest_graphlike_error())
            print(f'dist={dist}')
            dists[i,j] = dist
        except:
            print(f'Failed to simulate for dx:{dx},dy:{dy} and num_vortexes={num_vortexes}')

# plt.pcolor(dists.T)
# plt.xticks(np.arange(len(dx_list))+0.5, dx_list)
# plt.yticks(np.arange(len(dy_list))+0.5, dy_list)
# plt.xlabel('dx')
# plt.ylabel('dy')
# plt.colorbar()


plt.pcolor(dists.T)
plt.xticks(np.arange(len(vx_list))+0.5, vx_list)
plt.yticks(np.arange(len(vy_list))+0.5, vy_list)
plt.xlabel('vx')
plt.ylabel('vy')
plt.colorbar()

plt.show()
