import numpy as np
from color_code import FloquetCode
from lattice import *
from noise import get_noise_model
from matplotlib import pyplot as plt
from idle import idle_qubits
import stim



noise_type = 'EM3_v2'
lattice_type = HexagonalLatticeGidney
reps_without_noise = 1
draw = False

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

for _ in range(100):
    logical_op_directions = np.random.choice(['y', 'x'], size=1)
    logical_op_directions = (logical_op_directions[0],)
    if logical_op_directions[0] == 'x':
        detectors = ('Z',)
        logical_operator_pauli_type = 'Z'
    elif logical_op_directions[0] == 'y':
        detectors = ('X',)
        logical_operator_pauli_type = 'X'
    else:
        raise ValueError(f'Unknown logical_op_directions: {logical_op_directions}')

    Lx = np.random.randint(1, 4+1)
    Ly = np.random.randint(1, 4+1)

    dx = 2*Lx
    dy = 3*Ly

    vx = np.random.randint(-5, 5+1)
    vy = np.random.randint(-5, 5+1)
    num_vortexes = (vx, vy)

    lat = lattice_type((dx, dy))

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
        print(f'Lx={Lx}, Ly={Ly}, vx={vx}, vy={vy}, logical_op_directions={logical_op_directions[0]}')
        computed_dist = compute_dist(Lx, Ly, vx, vy, logical_op_directions[0])
        print(f'computed_dist={computed_dist}, dist={dist}')
        assert dist == computed_dist
        if num_logicals != 2:
            configuration_works = False
        else:
            configuration_works = True
    except:
        configuration_works = False

    print(f'Lx={Lx}, Ly={Ly}, vx={vx}, vy={vy}, logical_op_directions={logical_op_directions[0]}')
    print('configuration_works=', configuration_works)
    print('is_allowed_vortex_numbers=', is_allowed_vortex_numbers(Lx, Ly, vx, vy))
    assert configuration_works == is_allowed_vortex_numbers(Lx, Ly, vx, vy)