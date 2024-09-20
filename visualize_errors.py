from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from color_code import *
from lattice import *
from noise import get_noise_model
import matplotlib.pyplot as plt


def draw_shortest_error(circ: stim.Circuit, draw_graph=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    if draw_graph:
        dem = circ.detector_error_model()
        for dem_element in dem:
            if dem_element.type == 'error' and len(dem_element.targets_copy()) == 2:
                det1 = dem_element.targets_copy()[0].val
                det2 = dem_element.targets_copy()[1].val
                cord1 = dem.get_detector_coordinates(det1)[det1]
                cord2 = dem.get_detector_coordinates(det2)[det2]
                ax.plot([cord1[0], cord2[0]], [cord1[1], cord2[1]], zs=[cord1[2], cord2[2]],
                        color='r', linewidth=1)
    for error in circ.shortest_graphlike_error():
        coords = [detector.coords[:3] for detector in error.dem_error_terms]
        coords = [c for c in coords if len(c)>0]
        coords = np.array(coords)
        # coords[:,2] = (coords[:,2] - min(coords[:,2]))/max(coords[:,2])
        ax.plot(coords[:,0], coords[:,1], zs=coords[:,2], linewidth=4)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('time')
    corner = lat.coords_to_pos((lat.size[0]-1, lat.size[1]-1, 1))
    ax.set_box_aspect([corner[0]/corner[1], 1, 1])  # x:y:z ratio
    lat.draw_3d(zplane=min(coords[:,2]), ax=ax)


if __name__ == '__main__':
    lat = HexagonalLatticeGidney((4,6))
    logical_op_directions = ('x',)
    detectors = ('Z',)
    logical_operator_pauli_type = 'Z'
    num_vortexes = (0, 0)  # (0,1)
    reps_without_noise = 1
    reps_with_noise = 1
    noise_type = 'DEPOLARIZE1'

    code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)
    circ, _, _, num_logicals = code.get_circuit(reps=reps_with_noise + 2 * reps_without_noise, reps_without_noise=reps_without_noise,
                                                noise_model=get_noise_model(noise_type, 0.1),
                                                logical_operator_pauli_type=logical_operator_pauli_type,
                                                logical_op_directions=logical_op_directions,
                                                detector_indexes=None, detector_args=None, draw=False,
                                                return_num_logical_qubits=True)

    print(len(circ.shortest_graphlike_error()))
    draw_shortest_error(circ, draw_graph=True)
    # plt.axis('equal')
    plt.show()