import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def int_to_color(i):
    return ['r', 'g', 'b'][i % 3]

@dataclass
class Plaquette:
    sites: list
    edges: list
    color: int
    wrapped: bool
    pos: tuple
    # pauli_label: str

    @property
    def color_str(self):
        return int_to_color(self.color)


class HexagonalLattice:
    def __init__(self, size):
        self.plaquettes = {}
        self.size = size
        self.G = nx.Graph()
        self.create_graph()
        self.create_plaquettes_and_colors()

    def shift_site(self, site, shift, return_was_wrapped=False):
        shifted_site = np.array(site) + np.array(shift)
        shifted_site_mod_size = tuple(shifted_site % np.array([self.size[0], self.size[1], 2]))
        was_wrapped = not np.allclose(shifted_site_mod_size, shifted_site)
        return shifted_site_mod_size, was_wrapped if return_was_wrapped else shifted_site_mod_size

    def coords_to_pos(self, coords):
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3) / 2, 1.5])
        sublat_offset = np.array([np.sqrt(3) / 2, 1 / 2])
        return tuple(coords[0] * a1 + coords[1] * a2 + coords[2] * sublat_offset)

    def create_graph(self):
        for row, col in np.ndindex(self.size):
            # Calculate skewed coordinate for the first site in the unit cell
            coord_1 = self.coords_to_pos((row, col, 0))
            coord_2 = self.coords_to_pos((row, col, 1))

            # Add the two sites of the unit cell to the graph
            self.G.add_node((row, col, 0), pos=tuple(coord_1), boundary=False)  # Sublattice 0
            self.G.add_node((row, col, 1), pos=tuple(coord_2), boundary=False)  # Sublattice 1

            # Add edges between neighboring sites using modulo arithmetic for periodicity
            site1 = (row, col, 1)
            for shift in [(0, 0, -1), (1, 0, -1), (0, 1, -1)]:
                site2, was_wrapped = self.shift_site(site1, shift, return_was_wrapped=True)
                self.G.add_edge(site1, site2, wrapped=was_wrapped, color=0, coords=(row, col),
                                pos=tuple(np.array(self.G.nodes[site1]["pos"]) + np.array(self.coords_to_pos(shift))/2))

    def create_plaquettes_and_colors(self):
        for row, col in np.ndindex(self.size):
            site1 = (row, col, 1)
            plaquette_shifts = [(0,0,0), (1,0,-1), (1,0,0), (1,1,-1), (0,1,0), (0,1,-1)]
            sites_and_wrapped = [self.shift_site(site1, shift, return_was_wrapped=True) for shift in plaquette_shifts]
            sites = list(map(lambda x: x[0], sites_and_wrapped))
            self.plaquettes[(row, col)] = Plaquette(sites=sites,
                                                    edges=[(sites[i], sites[(i + 1) % len(sites)]) for i in range(len(sites))],
                                                    color=(row - col) % 3,
                                                    wrapped=any(wrapped for site, wrapped in sites_and_wrapped),
                                                    pos=tuple(np.array(self.G.nodes[site1]["pos"]) +
                                                              np.array(self.coords_to_pos(np.mean(plaquette_shifts, axis=0)))))
            for edge in self.plaquettes[(row, col)].edges:
                self.G.edges[edge]["color"] = (self.G.edges[edge]["color"] - self.plaquettes[(row, col)].color) % 3

    def set_boundary(self, boundary_sites: list[tuple], boundary_type: str):
        for site in boundary_sites:
            self.G.nodes[site]["boundary"] = boundary_type
        edges_to_remove = []
        for site in boundary_sites:
            for neighbor in list(self.G.neighbors(site)):
                if neighbor in boundary_sites:
                    edges_to_remove.append((site, neighbor))
        self.G.remove_edges_from(edges_to_remove)

    def draw(self):
        pos = nx.get_node_attributes(self.G, 'pos')
        edge_colors = [int_to_color(self.G[u][v]['color']) for u, v in self.G.edges()]
        nx.draw(self.G, pos, with_labels=True, edge_color=edge_colors, width=3, node_size=300, font_size=8)

        # Draw plaquettes with different colors
        for plaquette in self.plaquettes.values():
            if plaquette.wrapped:
                continue
            hexagon = plt.Polygon([pos[v] for v in plaquette.sites], facecolor=plaquette.color_str, alpha=0.5,
                                  zorder=-1)
            plt.gca().add_patch(hexagon)

        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    # Add a hexagonal lattice with skewed coordinates, periodic boundary conditions, and plaquettes
    lattice = HexagonalLattice((6, 6))
    lattice.set_boundary([(ix, iy, s) for ix in range(lattice.size[0]) for iy in [0,lattice.size[1]-1] for s in range(2)], 'X')

    # Draw the graph to visualize the skewed coordinates and plaquette
    lattice.draw()
