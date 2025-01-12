import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BLUE = (0/255, 101/255, 183/255)


def draw_matching_graph_rep_code(n_qubits, n_rounds, n_vortices):
    """
    Draws a grid of vertices (blue squares) and edges (black lines) representing a repetition code.
    The lattice has periodic boundary conditions in the qubit axis and open boundary conditions in the time axis.
    The vertices are sheared based on the number of vortices.

    Parameters:
        n_qubits (int): Number of qubits.
        n_rounds (int): Number of rounds of error correction.
        n_vortices (int): Number of vortices, which introduces a shear in the lattice.
    """
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    marker_size = 10
    linewidth = 2

    # Loop over the points in the grid
    for j in range(n_rounds):
        for i in range(n_qubits):
            # Compute the sheared position of the vertex
            x = i
            y = j + i / n_qubits * n_vortices

            # Draw the vertex as a blue square with a black edge
            ax.plot(x, y, marker='s', color=BLUE, markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=3)

            # Draw the edges connecting to the next round (time axis)
            if j < n_rounds - 1:
                next_y = j + 1 + i / n_qubits * n_vortices
                ax.plot([x, x], [y, next_y], color='black', zorder=2, linewidth=linewidth)

            # Draw the edges connecting to the next qubit (space axis)
            next_x = (i + 1)
            next_y = j + (i + 1) / n_qubits * n_vortices

            linestyle = '--' if i == n_qubits - 1 else 'solid'
            ax.plot([x, next_x], [y, next_y], color='black', linestyle=linestyle, zorder=2, linewidth=linewidth)

            # draw diagonal edges going up one in the time direction and one in the space direction
            if j < n_rounds - 1:
                next_y = j + 1 + (i + 1) / n_qubits * n_vortices
                ax.plot([x, next_x], [y, next_y], color='lightgrey', linestyle=linestyle, zorder=2, linewidth=linewidth)

    # Set limits and aspect ratio
    ax.set_xlim(-1, n_qubits + 1)
    ax.set_ylim(-1, n_rounds + n_qubits / n_qubits * n_vortices)
    ax.set_aspect('equal')

    # Add labels and grid
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Rounds")
    ax.axis('off')
    ax.grid(False)

    plt.savefig('figures/rep_code_matching_graph.pdf', bbox_inches='tight', pad_inches=0)
    # Show the plot
    plt.show()

def plot_blue_cube(ax, size, x_y_z):
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])
    vertices = vertices - 0.5  # Center the cube at the origin
    vertices = vertices * size  # Scale the cube
    vertices = vertices + x_y_z  # Translate the cube

    # Define the 6 faces of the cube (as sets of vertices)
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
             [vertices[4], vertices[5], vertices[6], vertices[7]],
             [vertices[0], vertices[1], vertices[5], vertices[4]],
             [vertices[2], vertices[3], vertices[7], vertices[6]],
             [vertices[1], vertices[2], vertices[6], vertices[5]],
             [vertices[4], vertices[7], vertices[3], vertices[0]]]

    # Plot the faces with blue color
    ax.add_collection3d(Poly3DCollection(faces, facecolors=BLUE, linewidths=1, edgecolors='black'))#, alpha=1))



def draw_matching_graph_surface_code(n_qubits_x, n_qubits_y, n_rounds):
    """
    Draws a 3D grid of vertices and edges representing a surface code.
    The lattice has periodic boundary conditions in the qubit axes and open boundary conditions in the time axis.
    The vertices are sheared based on the number of vortices.

    Parameters:
        n_qubits_x (int): Number of qubits in the x direction.
        n_qubits_y (int): Number of qubits in the y direction.
        n_rounds (int): Number of rounds of error correction.
        n_vortices (int): Number of vortices, which introduces a shear in the lattice.
    """
    # Initialize the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    linewidth = 2

    # Loop over the points in the grid
    for z in range(n_rounds):
        for x in range(n_qubits_x):
            for y in range(n_qubits_y):
                # Compute the sheared position of the vertex

                # Draw the vertex as a blue square
                plot_blue_cube(ax,0.1, (x, y, z))

                # Draw edges along the x direction (space axis)
                if x < n_qubits_x - 1:
                    ax.plot([x, x + 1], [y, y], [z, z], color='black', linewidth=linewidth)
                else:
                    # ax.plot([x, x + 1], [y, y], [z, z], color='black', linewidth=linewidth, linestyle='--')
                    pass

                # Draw edges along the y direction (space axis)
                if y < n_qubits_y - 1:
                    ax.plot([x, x], [y, y + 1], [z, z], color='black', linewidth=linewidth)
                else:
                    # ax.plot([x, x], [y, y + 1], [z, z], color='black', linewidth=linewidth, linestyle='--')
                    pass

                # Draw edges along the t direction (time axis)
                if z < n_rounds - 1:
                    ax.plot([x, x], [y, y], [z, z+1], color='black', linewidth=linewidth)

                    # diagonals x+1, z+1 and y+1, z+1 and x+1, y+1, z+1
                    if x < n_qubits_x - 1:
                        ax.plot([x, x + 1], [y, y], [z, z + 1], color='black', linewidth=linewidth)
                    else:
                        # ax.plot([x, x + 1], [y, y], [z, z + 1], color='black', linewidth=linewidth, linestyle='--')
                        pass
                    if y < n_qubits_y - 1:
                        ax.plot([x, x], [y, y + 1], [z, z + 1], color='black', linewidth=linewidth)
                    else:
                        #ax.plot([x, x], [y, y + 1], [z, z + 1], color='black', linewidth=linewidth, linestyle='--')
                        pass
                    if x < n_qubits_x - 1 and y < n_qubits_y - 1:
                        ax.plot([x, x + 1], [y, y + 1], [z, z + 1], color='black', linewidth=linewidth)
                    else:
                        # ax.plot([x, x + 1], [y, y + 1], [z, z + 1], color='black', linewidth=linewidth, linestyle='--')
                        pass



    # remove the axis so that there's just a white background
    ax.axis('off')
    ax.grid(False)

    # Set the aspect ratio
    ax.set_box_aspect([n_qubits_x-1, n_qubits_y-1, n_rounds-1])  # Equal aspect for x, y, z axes

    # set elevation and azimuth
    ax.view_init(elev=36, azim=-67)

    plt.savefig('figures/surface_code_matching_graph.pdf', bbox_inches='tight', pad_inches=0)
    # Show the plot
    plt.show()


# Example usage
draw_matching_graph_rep_code(n_qubits=5, n_rounds=5, n_vortices=1)
draw_matching_graph_surface_code(n_qubits_x=3, n_qubits_y=3, n_rounds=2)
