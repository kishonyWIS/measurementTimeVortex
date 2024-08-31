import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable
from qiskit.quantum_info import Pauli


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


class Lattice:
    def __init__(self, size,
                 lattice_vectors:list[tuple], sublab_offsets:list[np.ndarray[2]],
                 edges_shifts: list[tuple[tuple[int,int,int],tuple[int,int,int]]],
                 plaquette_shifts: list[tuple[tuple[int,int,int], ...]]):
        self.lattice_vectors = list(map(np.array, lattice_vectors))
        self.sublab_offsets = sublab_offsets
        self.edges_shifts = edges_shifts
        self.plaquette_shifts = plaquette_shifts
        self.plaquettes = {}
        self.size = size
        self.G = nx.Graph()
        self.index_to_site = {}  # Mapping from index to site
        self.site_to_index = {}  # Mapping from site to index
        self.create_graph()
        self.create_plaquettes_and_colors()
        self.assign_indices()

    def shift_site(self, site, shift, return_was_wrapped=False):
        shifted_site = np.array(site) + np.array(shift)
        shifted_site_mod_size = tuple(shifted_site % np.array([self.size[0], self.size[1], len(self.sublab_offsets)]))
        was_wrapped = not np.allclose(shifted_site_mod_size, shifted_site)
        return shifted_site_mod_size, was_wrapped if return_was_wrapped else shifted_site_mod_size

    def coords_to_pos(self, coords):
        return tuple(np.sum([coord * vec for coord,vec in zip(coords[:-1],self.lattice_vectors)], axis=0) +
                     np.array(self.sublab_offsets[coords[2]]))

    def create_graph(self):
        for row, col in np.ndindex(self.size):
            for sublat in range(len(self.sublab_offsets)):
                site1 = (row, col, sublat)
                self.G.add_node(site1, pos=tuple(self.coords_to_pos(site1)), boundary=False)

        for row, col in np.ndindex(self.size):
            reference_site = (row, col, 0)
            for shifts in self.edges_shifts:
                site1, was_wrapped = self.shift_site(reference_site, shifts[0], return_was_wrapped=True)
                site2, was_wrapped = self.shift_site(reference_site, shifts[1], return_was_wrapped=True)
                mean_pos_of_shifts = np.mean([np.array(self.coords_to_pos(shift)) for shift in shifts], axis=0)
                self.G.add_edge(site1, site2, wrapped=was_wrapped, color=0,
                                coords=tuple(np.array(reference_site) + np.mean(shifts, axis=0)),
                                pos=tuple(np.array(self.G.nodes[reference_site]["pos"]) + mean_pos_of_shifts)
                                )

    def create_plaquettes_and_colors(self):
        for i_plaq, shifts in enumerate(self.plaquette_shifts):
            mean_pos_of_shifts = np.mean([np.array(self.coords_to_pos(shift)) for shift in shifts], axis=0)
            for row, col in np.ndindex(self.size):
                reference_site = (row, col, 0)
                sites_and_wrapped = [self.shift_site(reference_site, shift, return_was_wrapped=True) for shift in shifts]
                sites = list(map(lambda x: x[0], sites_and_wrapped))
                coords = (row, col, i_plaq)
                self.plaquettes[coords] = Plaquette(sites=sites,
                                                        edges=[(sites[i], sites[(i + 1) % len(sites)]) for i in range(len(sites))],
                                                        color=self.plaq_coords_to_color(coords),
                                                        wrapped=any(wrapped for site, wrapped in sites_and_wrapped),
                                                        pos=tuple(np.array(self.G.nodes[reference_site]["pos"]) +
                                                                  mean_pos_of_shifts))
                for edge in self.plaquettes[coords].edges:
                    self.G.edges[edge]["color"] = (self.G.edges[edge]["color"] - self.plaquettes[coords].color) % 3

    def plaq_coords_to_color(self, coords):
        return (coords[0] - coords[1]) % 3

    def assign_indices(self):
        # Assign unique indices to nodes not on boundary
        nodes = [site for site in self.G.nodes if not self.G.nodes[site]["boundary"]]
        self.index_to_site = {idx: site for idx, site in enumerate(nodes)}
        self.site_to_index = {site: idx for idx, site in self.index_to_site.items()}

    def set_boundary(self, boundary_sites: list[tuple], boundary_type: str):
        for site in boundary_sites:
            self.G.nodes[site]["boundary"] = boundary_type
        edges_to_remove = []
        for site in boundary_sites:
            for neighbor in list(self.G.neighbors(site)):
                if neighbor in boundary_sites:
                    edges_to_remove.append((site, neighbor))
        self.G.remove_edges_from(edges_to_remove)
        # remove edges_to_remove from all plaquettes
        for plaquette in self.plaquettes.values():
            plaquette.edges = [(site1, site2) for site1, site2 in plaquette.edges
                               if (site1, site2) not in edges_to_remove and (site2, site1) not in edges_to_remove]

        # Reindex the nodes after removing the boundary nodes
        self.assign_indices()

    def get_site_from_index(self, index):
        return self.index_to_site.get(index, None)

    def get_index_from_site(self, site):
        return self.site_to_index.get(site, None)

    def get_sites_on_logical_path(self, direction:str='x'):
        if direction == 'x':
            return [(ix, 2, s) for ix in range(self.size[0]) for s in range(len(self.sublab_offsets))]
        elif direction == 'y':
            return [(2, iy, s) for iy in range(self.size[1]) for s in range(len(self.sublab_offsets))]
        else:
            raise ValueError(f"Invalid direction {direction}")

    def draw(self):
        fig, ax = plt.subplots()

        pos = nx.get_node_attributes(self.G, 'pos')
        edge_colors = [int_to_color(self.G[u][v]['color']) for u, v in self.G.edges()]
        nx.draw(self.G, pos, ax=ax, with_labels=True, edge_color=edge_colors, width=3, node_size=300, font_size=8)

        # Draw plaquettes with different colors
        for plaquette in self.plaquettes.values():
            if plaquette.wrapped:
                continue
            hexagon = plt.Polygon([pos[v] for v in plaquette.sites], facecolor=plaquette.color_str, alpha=0.5,
                                  zorder=-1)
            ax.add_patch(hexagon)
        ax.set_aspect('equal')


class HexagonalLatticeSheared(Lattice):
    '''
    Hexagonal lattice on Torus
    '''
    def __init__(self, size: tuple):
        lattice_vectors = [(np.sqrt(3), 0), (np.sqrt(3) / 2, 1.5)]
        sublab_offsets = [(0, 0), (np.sqrt(3) / 2, 1 / 2 )]
        edges_shifts = [((0,0,1), (0,0,0)), ((0,0,1), (1,0,0)), ((0,0,1),(0,1,0))]
        plaquette_shifts = [((0,0,1), (1,0,0), (1,0,1), (1,1,0), (0,1,1), (0,1,0))]
        super().__init__(size, lattice_vectors, sublab_offsets, edges_shifts, plaquette_shifts)

class HexagonalLatticeShearedOnCylinder(HexagonalLatticeSheared):
    def __init__(self, size):
        super().__init__(size)
        self.set_boundary([(ix, iy, s) for ix in range(self.size[0]) for iy in [0, self.size[1]-1] for s in (0, 1)], 'X')


class HexagonalLatticeGidney(Lattice):
    def __init__(self, size):
        lattice_vectors = [(2,0), (0,2)]
        sublab_offsets = [(0,0), (0,1), (1,0), (1,1)]
        edges_shifts = [((0,0,0), (0,0,1)), ((0,0,2), (0,0,3)), ((0,0,1),(0,0,3)),
                        ((0,0,1), (0,1,0)), ((0,0,3), (0,1,2)), ((0,0,2), (1,0,0))]
        plaquette_shifts = [((0,0,1), (0,0,3), (0,1,2), (0,1,3), (0,1,1), (0,1,0)),
                            ((0,0,2), (0,0,3), (0,1,2), (1,1,0), (1,0,1), (1,0,0))]
        super().__init__(size, lattice_vectors, sublab_offsets, edges_shifts, plaquette_shifts)

    def get_sites_on_logical_path(self, direction:str='x'):
        if direction == 'x':
            return [(ix, 1, s) for ix in range(self.size[0]) for s in [0,1,3,2]]
        elif direction == 'y':
            return [(1, iy, s) for iy in range(self.size[1]) for s in [0,1]]
        else:
            raise ValueError(f"Invalid direction {direction}")

    def plaq_coords_to_color(self, coords):
        return (-coords[2] - coords[1]) % 3


class HexagonalLatticeGidneyOnCylinder(HexagonalLatticeGidney):
    def __init__(self, size):
        super().__init__(size)
        self.set_boundary([(ix, iy, s) for ix in range(self.size[0]) for iy in [0, self.size[1]-1] for s in range(len(self.sublab_offsets))], 'X')


if __name__ == '__main__':
    # Add a hexagonal lattice with skewed coordinates, periodic boundary conditions, and plaquettes
    lattice = Lattice((6, 6))
    lattice.set_boundary([(ix, iy, s) for ix in range(lattice.size[0]) for iy in [0,lattice.size[1]-1] for s in range(2)], 'X')

    # Draw the graph to visualize the skewed coordinates and plaquette
    lattice.draw()