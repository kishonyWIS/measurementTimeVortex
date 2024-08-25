from color_code import *
from geometry import *
from noise import get_noise_model
import matplotlib.pyplot as plt


def draw_pauli_3d(code, ax=None, z_plane=0):
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Draw plaquettes as shaded polygons
    for plaquette in code.plaquettes:
        if plaquette.pauli_label == 'Z':
            continue
        sites, was_shifted = code.geometry.sites_unwrap_periodic(plaquette.sites, return_was_shifted=True)
        if was_shifted:
            continue
        points = list(map(code.geometry.site_to_physical_location, sites))
        points_3d = [(x, y, z_plane) for x, y in points]  # Add constant z-plane
        color = code.geometry.get_plaquette_color(plaquette.coords)

        # Create a 3D polygon
        polygon = Poly3DCollection([points_3d], color=color, alpha=0.2)
        ax.add_collection3d(polygon)

    # Draw bonds
    for bond in code.bonds:
        sites = copy(bond.sites)
        sites = code.geometry.sites_unwrap_periodic(sites)
        points = [code.geometry.site_to_physical_location(site) for site in sites]
        xs, ys = zip(*points)
        zs = [z_plane] * len(xs)  # Set z to the constant plane
        ax.plot(xs, ys, zs, 'k')

        # Add bond label
        x = np.mean(xs)
        y = np.mean(ys)
        z = z_plane
        fontsize = 10
        y = y + (bond.pauli_label == 'XX') * 0.2 - (bond.pauli_label == 'ZZ') * 0.2
        # ax.text(x, y, z, '{:.1f}'.format(bond.order * 6) + bond.pauli_label, fontsize=fontsize, ha='center', va='center')



def draw_shortest_error(circ: stim.Circuit):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    for error in circ.shortest_graphlike_error():
        coords = [detector.coords[:3] for detector in error.dem_error_terms]
        coords = [c for c in coords if len(c)>0]
        coords = np.array(coords)
        # coords[:,2] = (coords[:,2] - min(coords[:,2]))/max(coords[:,2])
        ax.plot(coords[:,0], coords[:,1], zs=coords[:,2], linewidth=4)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('time')
    corner = code.geometry.site_to_physical_location([dx,dy,2])
    ax.set_box_aspect([corner[0]/corner[1], 1, 0.2])  # x:y:z ratio
    draw_pauli_3d(code, ax=ax, z_plane=min(coords[:,2]))


if __name__ == '__main__':
    boundary_conditions = ('periodic', 'periodic')
    geometry = SymmetricTorus
    logical_op_directions = ('x')
    detectors = ('Z',)
    logical_operator_pauli_type = 'Z'
    num_vortexes = (0, 1)  # (0,1)
    dx, dy = 12, 6
    reps_without_noise = 1
    reps_with_noise = 12
    noise_type = 'DEPOLARIZE1'

    code = FloquetCode(dx, dy, boundary_conditions=boundary_conditions,
                       num_vortexes=num_vortexes, geometry=geometry, detectors=detectors)

    circ, _, _ = code.get_circuit(
        reps=reps_with_noise + 2 * reps_without_noise, reps_without_noise=reps_without_noise,
        noise_model=get_noise_model(noise_type, 0.1),
        logical_operator_pauli_type=logical_operator_pauli_type,
        logical_op_directions=logical_op_directions,
        detector_indexes=None, detector_args=None)

    print(len(circ.shortest_graphlike_error()))
    draw_shortest_error(circ)
    # plt.axis('equal')
    plt.show()
