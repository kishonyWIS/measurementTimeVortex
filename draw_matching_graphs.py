import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.patches as patches

# BLUE_MPL = (0/255, 101/255, 183/255)
YELLOW_MPL = (255/255, 255/255, 0/255)
# BLUE = "rgb(0, 101, 183)"
YELLOW = "rgb(255, 255, 0)"

def draw_matching_graph_rep_code(n_qubits, n_rounds, n_vortices, filename='rep_code_matching_graph'):
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
    marker_size = 16
    linewidth = 3
    square_len = 0.25

    # Loop over the points in the grid
    for j in range(n_rounds):
        for i in range(n_qubits):
            # Compute the sheared position of the vertex
            x = i
            y = j + i / n_qubits * n_vortices

            # Draw the vertex as a blue square with a black edge
            # ax.plot(x, y, marker='s', color=YELLOW_MPL, markersize=marker_size, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            rect = patches.Rectangle((x-square_len/2, y-square_len/2), square_len, square_len, linewidth=1.5, edgecolor='k', facecolor=YELLOW_MPL, zorder=3)
            ax.add_patch(rect)

            # Draw the edges connecting to the next round (time axis)
            if j < n_rounds - 1:
                next_y = j + 1 + i / n_qubits * n_vortices
                ax.plot([x, x], [y, next_y], color='black', zorder=2, linewidth=linewidth)

            # Draw the edges connecting to the next qubit (space axis)
            next_x = (i + 1)
            next_y = j + (i + 1) / n_qubits * n_vortices

            linestyle = 'solid'#'--' if i == n_qubits - 1 else 'solid'
            ax.plot([x, next_x], [y, next_y], color='black', linestyle=linestyle, zorder=2, linewidth=linewidth)

            # draw diagonal edges going up one in the time direction and one in the space direction
            if j < n_rounds - 1:
                next_y = j + 1 + (i + 1) / n_qubits * n_vortices
                ax.plot([x, next_x], [y, next_y], color='lightgrey', linestyle=linestyle, zorder=2, linewidth=linewidth)

    # draw an extra line of squares at x=n_qubits, with alpha=0.15
    for j in range(n_rounds):
        x = n_qubits
        y = j + n_vortices
        rect = patches.Rectangle((x-square_len/2, y-square_len/2), square_len, square_len, linewidth=1.5, edgecolor='k', facecolor=YELLOW_MPL, zorder=3, alpha=0.15)
        ax.add_patch(rect)

    # Set limits and aspect ratio
    ax.set_xlim(-1, n_qubits + 1)
    ax.set_ylim(-1, n_rounds + n_qubits / n_qubits * n_vortices)
    ax.set_aspect('equal')

    # Add labels and grid
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Rounds")
    ax.axis('off')
    ax.grid(False)

    plt.savefig('figures/'+filename+'.pdf', bbox_inches='tight', pad_inches=0)

# Function to draw a cube with edges
def add_cube(x, y, z, half_size, edgewidth, fig):
    fig.add_trace(go.Mesh3d(
        # 8 vertices of a cube
        x=[x - half_size, x - half_size, x + half_size, x + half_size, x - half_size, x - half_size, x + half_size, x + half_size],
        y=[y - half_size, y + half_size, y + half_size, y - half_size, y - half_size, y + half_size, y + half_size, y - half_size],
        z=[z - half_size, z - half_size, z - half_size, z - half_size, z + half_size, z + half_size, z + half_size, z + half_size],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=YELLOW,
        opacity=1.0,
        flatshading=True,
        showlegend=False
    ))

    vertices = [
        (x - half_size, y - half_size, z - half_size),  # 0
        (x + half_size, y - half_size, z - half_size),  # 1
        (x + half_size, y + half_size, z - half_size),  # 2
        (x - half_size, y + half_size, z - half_size),  # 3
        (x - half_size, y - half_size, z + half_size),  # 4
        (x + half_size, y - half_size, z + half_size),  # 5
        (x + half_size, y + half_size, z + half_size),  # 6
        (x - half_size, y + half_size, z + half_size),  # 7
    ]

    # Cube edges (pairs of vertex indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    # Draw cube edges
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[vertices[edge[0]][0], vertices[edge[1]][0]],
            y=[vertices[edge[0]][1], vertices[edge[1]][1]],
            z=[vertices[edge[0]][2], vertices[edge[1]][2]],
            mode="lines",
            line=dict(color="black", width=edgewidth),
            showlegend=False
        ))

def clean_fig(fig, eye):
    fig.update_layout(
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color of the whole paper
        xaxis=dict(showline=False, showgrid=False, zeroline=False),  # Hide x-axis line, grid, and zero line
        yaxis=dict(showline=False, showgrid=False, zeroline=False),  # Hide y-axis line, grid, and zero line
        scene=dict(
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            zaxis=dict(visible=False),  # Hide z-axis
            aspectmode="data"
        ),
        showlegend=False
    )
    fig.update_layout(
        scene_camera=dict(projection=dict(type="orthographic")),
    )
    # set camera angle
    fig.update_layout(scene_camera=dict(eye=eye))
    # remove margins
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

def draw_matching_graph_surface_code(n_qubits_x, n_qubits_y, n_rounds, scale=1.0):
    """
    Draws a 3D grid of vertices (scaled blue cubes with black edges) and edges (black lines) representing a surface code.
    """
    fig = go.Figure()

    linewidth = 8
    half_size = 0.075 * scale

    # at z=0 draw red squares between (x+0.5,y), (x,y+0.5), (x-0.5,y), (x,y-0.5) for x,y in [0,1,2]
    for x in range(n_qubits_x):
        for y in range(n_qubits_y):
            fig.add_trace(go.Mesh3d(
                x=[x - 0.5, x, x + 0.5, x, x],
                y=[y, y + 0.5, y, y - 0.5, y],
                z=[0]*5,
                i=[0, 1, 2, 3, 4],
                j=[1, 2, 3, 0, 4],
                k=[2, 3, 0, 1, 4],
                color='blue',
                opacity=0.15,
                flatshading=True,
                showlegend=False
            ))
    for x in np.arange(n_qubits_x-1)+0.5:
        for y in np.arange(n_qubits_y-1)+0.5:
            fig.add_trace(go.Mesh3d(
                x=[x - 0.5, x, x + 0.5, x, x],
                y=[y, y + 0.5, y, y - 0.5, y],
                z=[0] * 5,
                i=[0, 1, 2, 3, 4],
                j=[1, 2, 3, 0, 4],
                k=[2, 3, 0, 1, 4],
                color='red',
                opacity=0.15,
                flatshading=True,
                showlegend=False
            ))

    # Loop over the grid to draw cubes and edges
    for z in range(n_rounds):
        for x in range(n_qubits_x):
            for y in range(n_qubits_y):
                # Add a cube at the given position
                add_cube(x, y, z, half_size, edgewidth=4, fig=fig)

                dash = 'solid'
                # Draw connecting edges between cubes along x, y, and z axes
                # dash = 'solid' if x<n_qubits_x-1 else 'dash'
                if x<n_qubits_x-1:
                    fig.add_trace(go.Scatter3d(
                        x=[x , x + 1],
                        y=[y, y],
                        z=[z, z],
                        mode="lines",
                        line=dict(color="black", width=linewidth, dash=dash),
                        showlegend=False
                    ))
                # dash = 'solid' if y<n_qubits_y-1 else 'dash'
                if y<n_qubits_y-1:
                    fig.add_trace(go.Scatter3d(
                        x=[x, x],
                        y=[y, y + 1],
                        z=[z, z],
                        mode="lines",
                        line=dict(color="black", width=linewidth, dash=dash),
                        showlegend=False
                    ))
                # dash = 'solid' if z<n_rounds-1 else 'dash'
                if z<n_rounds-1:
                    fig.add_trace(go.Scatter3d(
                        x=[x, x],
                        y=[y, y],
                        z=[z, z + 1],
                        mode="lines",
                        line=dict(color="black", width=linewidth, dash=dash),
                        showlegend=False
                    ))
                # diagonal edge x+1,y,z+1
                # dash = 'solid' if x<n_qubits_x-1 and z<n_rounds-1 else 'dash'
                if x<n_qubits_x-1 and z<n_rounds-1:
                    fig.add_trace(go.Scatter3d(
                        x=[x, x + 1],
                        y=[y, y],
                        z=[z, z + 1],
                        mode="lines",
                        line=dict(color="lightgrey", width=linewidth, dash=dash),
                        showlegend=False
                    ))
                # diagonal edge x,y+1,z-1
                if y<n_qubits_y-1 and z>0:
                    fig.add_trace(go.Scatter3d(
                        x=[x, x],
                        y=[y, y + 1],
                        z=[z, z - 1],
                        mode="lines",
                        line=dict(color="lightgrey", width=linewidth, dash=dash),
                        showlegend=False
                    ))
                # diagonal edge x+1,y+1,z+1
                # dash = 'solid' if x<n_qubits_x-1 and y<n_qubits_y-1 and z<n_rounds-1 else 'dash'
                if x<n_qubits_x-1 and y<n_qubits_y-1 and z<n_rounds-1:
                    fig.add_trace(go.Scatter3d(
                        x=[x, x + 1],
                        y=[y, y + 1],
                        z=[z, z + 1],
                        mode="lines",
                        line=dict(color="lightgrey", width=linewidth),
                        showlegend=False
                    ))

    # Set layout properties
    clean_fig(fig, eye = dict(x=2.1, y=-0.7, z=1.3))
    # save
    # fig.write_image("figures/surface_code_matching_graph.svg")
    pio.write_image(fig, "figures/surface_code_matching_graph.pdf", format="pdf", width=1920, height=1080, scale=2)

    fig.show()

def draw_surface_code_circuit(n_qubits_x=2, n_qubits_y=2):
    fig = go.Figure()
    linewidth = 8
    half_size = 0.075


    # at z=0 draw red squares between (x+0.5,y), (x,y+0.5), (x-0.5,y), (x,y-0.5) for x,y in [0,1,2]
    for x in range(n_qubits_x):
        for y in range(n_qubits_y):
            fig.add_trace(go.Mesh3d(
                x=[x - 0.5, x, x + 0.5, x, x],
                y=[y, y + 0.5, y, y - 0.5, y],
                z=[0]*5,
                i=[0, 1, 2, 3, 4],
                j=[1, 2, 3, 0, 4],
                k=[2, 3, 0, 1, 4],
                color='blue',
                opacity=0.15,
                flatshading=True,
                showlegend=False
            ))
    for x in np.arange(n_qubits_x-1)+0.5:
        for y in np.arange(n_qubits_y-1)+0.5:
            fig.add_trace(go.Mesh3d(
                x=[x - 0.5, x, x + 0.5, x, x],
                y=[y, y + 0.5, y, y - 0.5, y],
                z=[0] * 5,
                i=[0, 1, 2, 3, 4],
                j=[1, 2, 3, 0, 4],
                k=[2, 3, 0, 1, 4],
                color='red',
                opacity=0.15,
                flatshading=True,
                showlegend=False
            ))

    # Loop over the grid to draw cubes and edges
    # for z in range(2):
    #     for x in range(2):
    #         for y in range(2):
    #             # Add a cube at the given position
    #             add_cube(x, y, z, half_size, edgewidth=4, fig=fig)
    for x,y in [(0,0), (0,1), (1,0), (1,1), (0.5,0.5)]:
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[0, 1],
            mode="lines",
            line=dict(color="grey", width=linewidth, dash='dash'),
            showlegend=False
        ))
    for x,y in [(0,0.5), (0.5,0), (0.5,1), (1,0.5)]:
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[0, 1],
            mode="lines",
            line=dict(color="grey", width=linewidth),
            showlegend=False
        ))

    for x,y in [(0,0), (0,1), (1,0), (1,1)]:
        fig.add_trace(go.Scatter3d(
            x=[x, x + 0.5],
            y=[y, y],
            z=[1 / 5, 1 / 5],
            mode="lines",
            line=dict(color="black", width=linewidth),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y+0.5],
            z=[2 / 5, 2 / 5],
            mode="lines",
            line=dict(color="black", width=linewidth),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y-0.5],
            z=[3 / 5, 3 / 5],
            mode="lines",
            line=dict(color="black", width=linewidth),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x, x-0.5],
            y=[y, y],
            z=[4 / 5, 4 / 5],
            mode="lines",
            line=dict(color="black", width=linewidth),
            showlegend=False
        ))

    #cnots for x detector
    x,y = 0.5, 0.5
    fig.add_trace(go.Scatter3d(
        x=[x, x + 0.5],
        y=[y, y],
        z=[1 / 5, 1 / 5],
        mode="lines",
        line=dict(color="black", width=linewidth),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[x, x],
        y=[y, y - 0.5],
        z=[2 / 5, 2 / 5],
        mode="lines",
        line=dict(color="black", width=linewidth),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[x, x],
        y=[y, y + 0.5],
        z=[3 / 5, 3 / 5],
        mode="lines",
        line=dict(color="black", width=linewidth),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[x, x - 0.5],
        y=[y, y],
        z=[4 / 5, 4 / 5],
        mode="lines",
        line=dict(color="black", width=linewidth),
        showlegend=False
    ))

    #plot a sphere
    for x,y in [(0,0), (0,1), (1,0), (1,1)]:
        for z in [1/5, 2/5, 3/5, 4/5]:
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers",
                marker=dict(size=10, color="black"),
                showlegend=False
            ))

    fig.add_trace(go.Scatter3d(
        x=[1, 0.5, 0.5, 0],
        y=[0.5, 0, 1, 0.5],
        z=[1/5,2/5,3/5,4/5],
        mode="markers",
        marker=dict(size=10, color="black"),
        showlegend=False
    ))

    clean_fig(fig, eye=dict(x=-1.5 * 0.5, y=-2.1 * 0.5, z=0.9 * 0.5))
    fig.show()




# Example usage with scale parameter
draw_matching_graph_rep_code(n_qubits=5, n_rounds=3, n_vortices=0, filename='rep_code_matching_graph_0_vortex')
draw_matching_graph_rep_code(n_qubits=5, n_rounds=3, n_vortices=1, filename='rep_code_matching_graph_1_vortex')
draw_matching_graph_surface_code(n_qubits_x=3, n_qubits_y=3, n_rounds=2, scale=0.7)
draw_surface_code_circuit()
plt.show()