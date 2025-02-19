import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from IPython.core.pylabtools import figsize
from pandas.core.array_algos.transforms import shift
from qiskit.quantum_info import Pauli
from scipy.constants import point


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
    def __init__(self, lattice_vectors:list[tuple], coords_to_pos:callable, sublat_offsets:list[np.ndarray[2]],
                 edges_shifts: list[tuple[tuple[int,int,int],tuple[int,int,int]]],
                 plaquette_shifts: list[tuple[tuple[int,int,int], ...]]):
        self.lattice_vectors = list(map(np.array, lattice_vectors))
        self.sublat_offsets = sublat_offsets
        self.coords_to_pos = coords_to_pos
        self.edges_shifts = edges_shifts
        self.plaquette_shifts = plaquette_shifts
        self.all_coords_i_j = self.get_all_coords_i_j()
        self.plaquettes = {}
        self.G = nx.Graph()
        self.index_to_site = {}  # Mapping from index to site
        self.site_to_index = {}  # Mapping from site to index
        self.create_graph()
        self.create_plaquettes_and_colors()
        self.assign_indices()

    def shift_site1_near_site2(self, site1, site2):
        # shift site1 by a multiple of the lattice vectors to be close to site2 in terms of position
        min_dist = np.inf
        lattice_vectors_with_sublat = [np.array((vec[0], vec[1], 0)) for vec in self.lattice_vectors]
        for i in range(-1, 2):
            for j in range(-1, 2):
                shifted_site = np.array(site1) + i * lattice_vectors_with_sublat[0] + j * lattice_vectors_with_sublat[1]
                dist = np.linalg.norm(np.array(self.coords_to_pos(shifted_site)) - np.array(self.coords_to_pos(site2)))
                if dist < min_dist:
                    min_dist = dist
                    best_shifted_site = shifted_site
        return best_shifted_site

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

    def wrap_periodic(self, site):
        # shift sites by integer multiples of the lattice vectors to be within the unit cell
        lattice_matrix = np.column_stack(self.lattice_vectors).astype(int)
        coords = np.linalg.solve(lattice_matrix, site[:2])
        coords_mod = np.mod(coords.round(decimals=10), 1)
        site_i_j = lattice_matrix @ coords_mod
        return (int(np.round(site_i_j[0])), int(np.round(site_i_j[1])), site[2])

    def shift_site(self, site, shift, return_was_wrapped=False):
        shifted_site = np.array(site) + np.array(shift)
        shifted_site_mod_size = self.wrap_periodic(shifted_site)
        was_wrapped = not np.allclose(shifted_site_mod_size, shifted_site)
        return shifted_site_mod_size, was_wrapped if return_was_wrapped else shifted_site_mod_size

    def get_all_coords_i_j(self):
        # generate all possible coordinates using lattice vectors
        v1, v2 = self.lattice_vectors
        basis = np.array([v1, v2]).T  # Basis as columns of a matrix

        # Determine bounding box limits for search
        max_x = max(abs(v1[0]), abs(v2[0]), abs(v1[0] + v2[0]))
        max_y = max(abs(v1[1]), abs(v2[1]), abs(v1[1] + v2[1]))

        points = []

        for x in range(-max_x, max_x + 1):
            for y in range(-max_y, max_y + 1):
                # Solve for m, n in r = m*v1 + n*v2
                try:
                    m, n = np.linalg.solve(basis, [x, y])

                    # Check if the point is in the unit cell
                    if 0 <= m.round(decimals=10) < 1 and 0 <= n.round(decimals=10) < 1:
                        points.append((x, y))
                except np.linalg.LinAlgError:
                    # Skip if the basis is singular (not invertible)
                    pass
        # wrap all points to the unit cell
        points = [self.wrap_periodic((point[0], point[1], 0))[:-1] for point in points]
        points = sorted(points)

        # draw points and lattice vectors
        # fig, ax = plt.subplots()
        # for point in points:
        #     ax.plot(point[0], point[1], 'o')
        # ax.plot([0, v1[0]], [0, v1[1]], 'r')
        # ax.plot([0, v2[0]], [0, v2[1]], 'b')
        # ax.set_aspect('equal')
        # plt.show()
        return points

    def create_graph(self):
        for row, col in self.all_coords_i_j:
            for sublat in range(len(self.sublat_offsets)):
                site1 = (row, col, sublat)
                self.G.add_node(site1, pos=tuple(self.coords_to_pos(site1)), boundary=False)

        for row, col in self.all_coords_i_j:
            reference_site = (row, col, 0)
            for shifts in self.edges_shifts:
                site1, was_wrapped = self.shift_site(reference_site, shifts[0], return_was_wrapped=True)
                site2, was_wrapped = self.shift_site(reference_site, shifts[1], return_was_wrapped=True)
                if site2 == (1,-8,0):
                    self.shift_site(reference_site, shifts[1], return_was_wrapped=True)
                mean_pos_of_shifts = np.mean([np.array(self.coords_to_pos(shift)) for shift in shifts], axis=0)
                self.G.add_edge(site1, site2, wrapped=was_wrapped, color=0,
                                coords=tuple(np.array(reference_site) + np.mean(shifts, axis=0)),
                                pos=tuple(np.array(self.G.nodes[reference_site]["pos"]) + mean_pos_of_shifts)
                                )
        # nx.draw(self.G, pos=nx.get_node_attributes(self.G, 'pos'), node_color='black', edge_color='black')

    def create_plaquettes_and_colors(self):
        for i_plaq, shifts in enumerate(self.plaquette_shifts):
            mean_pos_of_shifts = np.mean([np.array(self.coords_to_pos(shift)) for shift in shifts], axis=0)
            for row, col in self.all_coords_i_j:
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
        return (-coords[0] + coords[1]) % 3

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
        def flexible_range(*args, step=1):
            if len(args) == 1:
                start, end = 0, args[0]  # Default start is 0
            elif len(args) == 2:
                start, end = args
            else:
                raise ValueError("flexible_range() accepts 1 or 2 positional arguments (end, or start and end)")

            if start < end:
                return range(start, end, step)
            else:
                return range(start, end, -step)
        if direction == 'x':
            v = self.lattice_vectors[0]
        elif direction == 'y':
            v = self.lattice_vectors[1]
        else:
            raise ValueError(f"Invalid direction {direction}")
        sites = []
        for index, i in enumerate(flexible_range(v[0])):
            for sublat in range(len(self.sublat_offsets)):
                if index == 0 and sublat == 0 and np.sign(v[0]*v[1]) == -1:
                    continue
                site = (i, 0, sublat)
                sites.append(site)
        for index, i in enumerate(flexible_range(v[1])):
            for sublat in range(len(self.sublat_offsets)):
                if index == 0 and sublat == 1 and np.sign(v[0]*v[1]) == -1:
                    continue
                site = (v[0], i, sublat)
                sites.append(site)

        sites = [self.wrap_periodic(site) for site in sites]
        return sites

    def draw(self, ax=None, linewidth=3, qubitsize=10):
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))

        # plot edges
        for u, v in self.G.edges():
            unwrapped_sites, was_wrapped = self.unwrap_periodic([u, v], return_was_wrapped=True)
            pos1 = self.coords_to_pos(unwrapped_sites[0])
            pos2 = self.coords_to_pos(unwrapped_sites[1])
            x = [pos1[0], pos2[0]]
            y = [pos1[1], pos2[1]]
            color = int_to_color(self.G[u][v]['color'])
            ax.plot(x, y, color=color, linewidth=linewidth, zorder=0)

        # plot plaquettes
        for plaquette in self.plaquettes.values():
            sites = plaquette.sites
            if plaquette.wrapped:
                # continue
                sites = self.unwrap_periodic(sites)
            plaquette_pos = [self.coords_to_pos(site) for site in sites]
            hexagon = plt.Polygon(plaquette_pos, facecolor=plaquette.color_str, alpha=0.3,
                                  zorder=-10)
            ax.add_patch(hexagon)

        # plot nodes
        for site in self.G.nodes():
            pos = self.G.nodes[site]['pos']
            color = 'grey' if self.G.nodes[site]['boundary'] else 'black'
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=qubitsize, zorder=10)

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


class HexagonalLattice(Lattice):
    def __init__(self, L1, L2):
        lattice_vectors = [L1, L2]
        sublat_offsets = [(0, 0), (1, 0)]
        edges_shifts = [((0,0,1), (0,0,0)), ((0,0,1), (1,0,0)), ((0,0,1),(0,1,0))]
        plaquette_shifts = [((0,0,1), (1,0,0), (1,0,1), (1,1,0), (0,1,1), (0,1,0))]
        i_to_pos = (1.5, -np.sqrt(3)/2)
        j_to_pos = (1.5, np.sqrt(3)/2)
        coords_to_pos = lambda site: (site[0] * i_to_pos[0] + site[1] * j_to_pos[0] + sublat_offsets[site[2]][0],
                                      site[0] * i_to_pos[1] + site[1] * j_to_pos[1] + sublat_offsets[site[2]][1])
        super().__init__(lattice_vectors, coords_to_pos, sublat_offsets, edges_shifts, plaquette_shifts)