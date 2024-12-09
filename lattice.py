import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from IPython.core.pylabtools import figsize
from pandas.core.array_algos.transforms import shift
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
                 lattice_vectors:list[tuple], sublat_offsets:list[np.ndarray[2]],
                 edges_shifts: list[tuple[tuple[int,int,int],tuple[int,int,int]]],
                 plaquette_shifts: list[tuple[tuple[int,int,int], ...]]):
        self.lattice_vectors = list(map(np.array, lattice_vectors))
        self.sublat_offsets = sublat_offsets
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

    def shift_site1_near_site2(self, site1, site2):
        # shift site1 by a multiple of the lattice vectors to be close to site2 in terms of position
        min_dist = np.inf
        for i in range(-1, 2):
            for j in range(-1, 2):
                shifted_site = np.array(site1) + i * np.array([self.size[0], 0, 0]) + j * np.array([0, self.size[1], 0])
                dist = np.linalg.norm(np.array(self.coords_to_pos(shifted_site)) - np.array(self.coords_to_pos(site2)))
                if dist < min_dist:
                    min_dist = dist
                    best_shift = (i, j)
        shifted_site = np.array(site1) + best_shift[0] * np.array([self.size[0], 0, 0]) + best_shift[1] * np.array([0, self.size[1], 0])
        return shifted_site

    def unwrap_periodic(self, sites, return_was_wrapped=False):
        was_wrapped = False
        # get the max and argmax
        ref_i = np.argmax([sum(site) for site in sites])
        ref_site = sites[ref_i]
        unwrapped_sites = []
        for site in sites[:ref_i] + sites[ref_i + 1:]:
            # Shift site to be close to ref_site by a multiple of the lattice vectors
            shifted_site = self.shift_site1_near_site2(site, ref_site)
            unwrapped_sites.append(shifted_site)
            was_wrapped = was_wrapped or not np.allclose(shifted_site, site)
        unwrapped_sites.insert(ref_i, ref_site)
        return (unwrapped_sites, was_wrapped) if return_was_wrapped else unwrapped_sites

    def shift_site(self, site, shift, return_was_wrapped=False):
        shifted_site = np.array(site) + np.array(shift)
        shifted_site_mod_size = tuple(shifted_site % np.array([self.size[0], self.size[1], len(self.sublat_offsets)]))
        was_wrapped = not np.allclose(shifted_site_mod_size, shifted_site)
        return shifted_site_mod_size, was_wrapped if return_was_wrapped else shifted_site_mod_size

    def coords_to_pos(self, coords):
        return tuple(np.sum([coord * vec for coord,vec in zip(coords[:-1],self.lattice_vectors)], axis=0) +
                     np.array(self.sublat_offsets[coords[2]]))

    def create_graph(self):
        for row, col in np.ndindex(self.size):
            for sublat in range(len(self.sublat_offsets)):
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
            return [(ix, 2, s) for ix in range(self.size[0]) for s in range(len(self.sublat_offsets))]
        elif direction == 'y':
            return [(2, iy, s) for iy in range(self.size[1]) for s in range(len(self.sublat_offsets))]
        else:
            raise ValueError(f"Invalid direction {direction}")

    def draw(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        pos = nx.get_node_attributes(self.G, 'pos')

        # plot edges
        for u, v in self.G.edges():
            unwrapped_sites, was_wrapped = self.unwrap_periodic([u, v], return_was_wrapped=True)
            pos1 = self.coords_to_pos(unwrapped_sites[0])
            pos2 = self.coords_to_pos(unwrapped_sites[1])
            x = [pos1[0], pos2[0]]
            y = [pos1[1], pos2[1]]
            color = int_to_color(self.G[u][v]['color'])
            ax.plot(x, y, color=color, linewidth=3, zorder=0)

        # plot plaquettes
        for plaquette in self.plaquettes.values():
            sites = plaquette.sites
            if plaquette.wrapped:
                # continue
                sites = self.unwrap_periodic(sites)
            plaquette_pos = [self.coords_to_pos(site) for site in sites]
            hexagon = plt.Polygon(plaquette_pos, facecolor=plaquette.color_str, alpha=0.5,
                                  zorder=-10)
            ax.add_patch(hexagon)

        # plot nodes
        for site in self.G.nodes():
            pos = self.G.nodes[site]['pos']
            color = 'grey' if self.G.nodes[site]['boundary'] else 'black'
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=10, zorder=10)

        ax.set_aspect('equal')

    def draw_3d(self, zplane=0, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        pos = nx.get_node_attributes(self.G, 'pos')
        for u, v in self.G.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [zplane, zplane]
            ax.plot(x, y, z, color='black')
        for plaquette in self.plaquettes.values():
            if plaquette.wrapped:
                continue
            points = [pos[v] for v in plaquette.sites]
            points3d = [(x, y, zplane) for x, y in points]  # Add constant z-plane
            hexagon = Poly3DCollection([points3d], facecolors=plaquette.color_str, alpha=0.5)
            ax.add_collection3d(hexagon)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


    # if ax is None:
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    # # Draw plaquettes as shaded polygons
    # for plaquette in code.plaquettes:
    #     if plaquette.pauli_label == 'Z':
    #         continue
    #     sites, was_shifted = code.geometry.sites_unwrap_periodic(plaquette.sites, return_was_shifted=True)
    #     if was_shifted:
    #         continue
    #     points = list(map(code.geometry.site_to_physical_location, sites))
    #     points_3d = [(x, y, z_plane) for x, y in points]  # Add constant z-plane
    #     color = code.geometry.get_plaquette_color(plaquette.coords)
    #
    #     # Create a 3D polygon
    #     polygon = Poly3DCollection([points_3d], color=color, alpha=0.2)
    #     ax.add_collection3d(polygon)
    #
    # # Draw bonds
    # for bond in code.bonds:
    #     sites = copy(bond.sites)
    #     sites = code.geometry.sites_unwrap_periodic(sites)
    #     points = [code.geometry.site_to_physical_location(site) for site in sites]
    #     xs, ys = zip(*points)
    #     zs = [z_plane] * len(xs)  # Set z to the constant plane
    #     ax.plot(xs, ys, zs, 'k')
    #
    #     # Add bond label
    #     x = np.mean(xs)
    #     y = np.mean(ys)
    #     z = z_plane
    #     fontsize = 10
    #     y = y + (bond.pauli_label == 'XX') * 0.2 - (bond.pauli_label == 'ZZ') * 0.2
    #     # ax.text(x, y, z, '{:.1f}'.format(bond.order * 6) + bond.pauli_label, fontsize=fontsize, ha='center', va='center')


class HexagonalLatticeSheared(Lattice):
    def __init__(self, size):
        lattice_vectors = [(np.sqrt(3), 0), (np.sqrt(3) / 2, 1.5)]
        sublat_offsets = [(0, 0), (np.sqrt(3) / 2, 1 / 2 )]
        edges_shifts = [((0,0,1), (0,0,0)), ((0,0,1), (1,0,0)), ((0,0,1),(0,1,0))]
        plaquette_shifts = [((0,0,1), (1,0,0), (1,0,1), (1,1,0), (0,1,1), (0,1,0))]
        super().__init__(size, lattice_vectors, sublat_offsets, edges_shifts, plaquette_shifts)

class HexagonalLatticeShearedNew(Lattice):
    def __init__(self, size):
        lattice_vectors = [(3, 0), (1.5, 3/2*np.sqrt(3))]
        sublat_offsets = [(0, 0), (1, 0), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3, np.sqrt(3)), (4, np.sqrt(3))]
        edges_shifts = [((0,0,0),(0,0,1)), ((0,0,2),(0,0,3)), ((0,0,4),(0,0,5)),
                        ((0,0,1),(0,0,2)), ((0,0,3),(0,0,4)), ((0,0,5),(1,1,0)),
                        ((1,0,0),(0,0,3)), ((1,0,2),(0,0,5)), ((0,0,4),(0,1,1))]
        plaquette_shifts = [((0,0,3), (0,0,4), (0,0,5), (1,0,2), (1,0,1), (1,0,0)),
                            ((0,0,4), (0,0,5), (1,1,0), (0,1,3), (0,1,2), (0,1,1)),
                            ((0,0,5), (1,1,0), (1,1,1), (1,0,4), (1,0,3), (1,0,2)),
                            ]
        super().__init__(size, lattice_vectors, sublat_offsets, edges_shifts, plaquette_shifts)

    def get_sites_on_logical_path(self, direction:str='x'):
        if direction == 'x':
            return [(ix, 1, s) for ix in range(self.size[0]) for s in [0,1,2,3]]
        elif direction == 'y':
            return [(1, iy, s) for iy in range(self.size[1]) for s in [1,2,3,4]]
        else:
            raise ValueError(f"Invalid direction {direction}")

    def plaq_coords_to_color(self, coords):
        return (coords[2]) % 3

class HexagonalLatticeShearedOnCylinder(HexagonalLatticeSheared):
    def __init__(self, size):
        super().__init__(size)
        self.set_boundary([(ix, iy, s) for ix in range(self.size[0]) for iy in [0, self.size[1]-1] for s in (0, 1)], 'X')

class HexagonalLatticeGidney(Lattice):
    def __init__(self, size):
        lattice_vectors = [(2,0), (0,2)]
        sublat_offsets = [(0,0), (0,1), (1,0), (1,1)]
        edges_shifts = [((0,0,0), (0,0,1)), ((0,0,2), (0,0,3)), ((0,0,1),(0,0,3)),
                        ((0,0,1), (0,1,0)), ((0,0,3), (0,1,2)), ((0,0,2), (1,0,0))]
        plaquette_shifts = [((0,0,1), (0,0,3), (0,1,2), (0,1,3), (0,1,1), (0,1,0)),
                            ((0,0,2), (0,0,3), (0,1,2), (1,1,0), (1,0,1), (1,0,0))]
        super().__init__(size, lattice_vectors, sublat_offsets, edges_shifts, plaquette_shifts)

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
        self.set_boundary([(ix, iy, s) for ix in range(self.size[0]) for iy in [0, self.size[1]-1] for s in range(len(self.sublat_offsets))], 'X')


class HexagonalLatticeGidneyOnPlaneWithHole(HexagonalLatticeGidney):
    def __init__(self, size, hole_size=(2,2)):
        super().__init__(size)
        hole_x, hole_y = self.size[0]//2, self.size[1]//2
        self.hole_xs = range(hole_x-hole_size[0]//2, hole_x+hole_size[0]//2)
        self.hole_ys = range(hole_y-hole_size[1]//2, hole_y+hole_size[1]//2)
        self.set_boundary([(ix, iy, s) for ix in range(self.size[0]) for iy in [0, self.size[1]-1] for s in range(len(self.sublat_offsets))], 'X')
        self.set_boundary([(ix, iy, s) for ix in [0, self.size[0]-1] for iy in range(self.size[1]) for s in range(len(self.sublat_offsets))], 'X')
        self.set_boundary([(ix, iy, s) for ix in self.hole_xs for iy in self.hole_ys for s in range(len(self.sublat_offsets))], 'X')

    def get_sites_on_logical_path(self, direction:str='around_hole'):
        if direction == 'around_hole':
            return [(ix, iy, s) for ix in range(self.hole_xs[0]-1, self.hole_xs[-1]+2) for iy in [self.hole_ys[0]-1, self.hole_ys[-1]+1] for s in range(4)] + \
                     [(ix, iy, s) for ix in [self.hole_xs[0]-1, self.hole_xs[-1]+1] for iy in range(self.hole_ys[0]-1, self.hole_ys[-1]+2) for s in [0,1]]
        elif direction == 'edge_to_hole':
            return [(self.hole_xs[-1], iy, s) for iy in range(0, self.hole_ys[0]+1) for s in [0,1]]
        else:
            raise ValueError(f"Invalid direction {direction}")

if __name__ == '__main__':
    # Add a hexagonal lattice with skewed coordinates, periodic boundary conditions, and plaquettes
    lattice = Lattice((6, 6))
    lattice.set_boundary([(ix, iy, s) for ix in range(lattice.size[0]) for iy in [0,lattice.size[1]-1] for s in range(2)], 'X')

    # Draw the graph to visualize the skewed coordinates and plaquette
    lattice.draw()