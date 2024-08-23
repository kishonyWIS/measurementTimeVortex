from copy import copy
from itertools import product, chain
from typing import Optional

import numpy as np


class Bond:
    def __init__(self, sites: np.ndarray, pauli_label: str, order: int):
        assert sites.shape[0] == len(pauli_label)
        self.sites = sites
        self.pauli_label = pauli_label
        self.order = order
        self.measurement_indexes = []

    def __repr__(self):
        return f'Bond({self.sites}, {self.pauli_label}, {self.order})'

    def overlaps(self, other):
        return np.any([np.all(self_site == other_site) for self_site, other_site in product(self.sites, other.sites)])


class Plaquette:
    def __init__(self, sites: list[np.ndarray], coords: list, center_x_y, pauli_label: Optional[str] = None):
        self.sites = sites
        self.bonds = []
        self.coords = coords
        self.center_x_y = center_x_y
        self.pauli_label = pauli_label

    def get_bonds(self, all_bonds):
        bonds = []
        for bond in all_bonds:
            if len(bond.pauli_label) == 1 and len(self.sites) == 6:
                continue
            if np.all([np.all(s == self.sites, axis=1).any() for s in bond.sites]):
                if self.pauli_label is None or np.all([p == self.pauli_label for p in bond.pauli_label]):
                    bonds.append(bond)
        self.bonds = bonds

    def measurement_indexes_and_sizes(self):
        all_indexes = [(index, len(bond.pauli_label)) for bond in self.bonds for index in bond.measurement_indexes]
        return sorted(all_indexes, key=lambda x: x[0])

class Geometry:
    def __init__(self, num_sites_x: int, num_sites_y: int, boundary_conditions: tuple):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.boundary_conditions = boundary_conditions
        self.test_boundary_conditions()

    def test_boundary_conditions(self):
        pass

    def site_to_physical_location(self, site):
        """
        Convert a site index to a physical location on the lattice.
        """
        pass

    def site_neighbor_directions_and_labels_and_colors(self, site):
        """
        Return the directions of the neighbors of a site.
        """
        pass

    def plaquette_offsets(self, ix, iy):
        """
        Return the offsets of the sites in a plaquette.
        """
        pass

    def get_sites_on_logical_path(self, direction: str):
        """
        Return the sites on a logical path.
        """
        pass

    def get_plaquette_color(self, coords):
        """
        Return the color of a plaquette.
        """
        pass

    def sites_unwrap_periodic(self, sites, return_was_shifted=False):
        # the site with maximal x and y coordinates is the reference_site
        reference_site = sorted(sites, key=lambda x: tuple(x))[len(sites)//2]
        shifted_sites = []
        for site in sites:
            new_site = copy(site)
            if np.linalg.norm(site[0] - reference_site[0]) > 1 and self.boundary_conditions[0] == 'periodic':
                new_site = new_site + np.array(
                    [self.num_sites_x * round((reference_site[0] - site[0]) / self.num_sites_x), 0, 0])
            if np.linalg.norm(site[1] - reference_site[1]) > 1 and self.boundary_conditions[1] == 'periodic':
                new_site = new_site + np.array(
                    [0, self.num_sites_y * round((reference_site[1] - site[1]) / self.num_sites_y), 0])
            shifted_sites.append(new_site)
        was_shifted = not all([np.allclose(site, shifted_site) for site, shifted_site in zip(sites, shifted_sites)])
        if return_was_shifted:
            return shifted_sites, was_shifted
        else:
            return shifted_sites

    def shift_site(self, site, shift):
        shifted_site = site + shift
        if not self.boundary_conditions[0] == 'periodic':
            if shifted_site[0] < 0 or shifted_site[0] >= self.num_sites_x:
                return None
        if not self.boundary_conditions[1] == 'periodic':
            if shifted_site[1] < 0 or shifted_site[1] >= self.num_sites_y:
                return None
        return shifted_site % np.array([self.num_sites_x, self.num_sites_y, 2])

    def get_plaquettes(self):
        plaquettes = []
        # add hexagonal plaquettes
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in
                         self.plaquette_offsets(ix, iy)]
                center_x_y = self.site_to_physical_location(reference_site + np.mean(np.array(self.plaquette_offsets(ix, iy)), axis=0))
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                for pauli_label in ['X', 'Z']:
                    plaquettes.append(Plaquette(sites, [ix, iy], center_x_y, pauli_label))
        return plaquettes

class SymmetricCylinder(Geometry):
    #periodic in x, open in y
    def test_boundary_conditions(self):
        assert self.boundary_conditions == ('periodic', 'open')

    def site_to_physical_location(self, site):
        x = np.sqrt(3) * site[0] + np.sqrt(3) / 2 * site[1] + np.sqrt(3) / 2 * site[2]
        y = 1.5 * site[1] + 0.5 * site[2]
        return x, y

    def site_neighbor_directions_and_labels_and_colors(self, site):
        if site[1] == 0:
            return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 1, 1]]), 'XX', (site[0] - site[1] - 0 + 1) % 3),
                    (np.array([[0, 0, 0],[0, 1, 1]]), 'ZZ', (site[0] - site[1] - 0 + 1) % 3),
                    (np.array([[0, 0, 1]]), 'X', (site[0] - site[1] - 1 + 0) % 3)]
        elif site[1] == self.num_sites_y-1:
            return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0]]), 'X', (site[0] - site[1] - 0 + 1) % 3)]
        else:
            return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (site[0] - site[1] - 0 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (site[0] - site[1] - 1 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 1, 1]]), 'XX', (site[0] - site[1] - 0 + 1) % 3),
                    (np.array([[0, 0, 0],[0, 1, 1]]), 'ZZ', (site[0] - site[1] - 0 + 1) % 3)]

    def plaquette_offsets(self, ix, iy):
        if iy == -1:
            return np.array([[0, 1, 0],
                             [0, 1, 1],
                             [1, 1, 0]])
        elif iy == self.num_sites_y-1:
            return np.array([[0, 0, 1],
                             [1, 0, 1],
                             [1, 0, 0]])
        else:
            return np.array([[0, 0, 1],
                             [0, 1, 0],
                             [0, 1, 1],
                             [1, 1, 0],
                             [1, 0, 1],
                             [1, 0, 0]])

    def get_plaquettes(self):
        plaquettes = []
        # add hexagonal plaquettes
        for ix in range(self.num_sites_x):
            for iy in range(-1, self.num_sites_y):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in
                         self.plaquette_offsets(ix, iy)]
                center_x_y = self.site_to_physical_location(
                    reference_site + np.mean(np.array(self.plaquette_offsets(ix, iy)), axis=0))
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                for pauli_label in ['X', 'Z']:
                    if iy in [-1, self.num_sites_y-1] and pauli_label == 'Z':
                        continue
                    plaquettes.append(Plaquette(sites, [ix, iy], center_x_y, pauli_label))
        return plaquettes

    def get_sites_on_logical_path(self, direction: str):
        if direction == 'x':
            return [[ix, iy, s]
                    for ix in range(self.num_sites_x)
                    for iy in [1]
                    for s in [0, 1]]
        elif direction == 'y':
            return [[ix, iy, s]
                    for ix in [0]
                    for iy in range(self.num_sites_y)
                    for s in [0, 1]]

    def get_plaquette_color(self, coords):
        color = (coords[0] - coords[1]) % 3
        return ['r', 'g', 'b'][color]


class SymmetricTorus(Geometry):
    def test_boundary_conditions(self):
        assert self.boundary_conditions == ('periodic', 'periodic')

    def site_to_physical_location(self, site):
        x = np.sqrt(3) * site[0] + np.sqrt(3) / 2 * site[1] + np.sqrt(3) / 2 * site[2]
        y = 1.5 * site[1] + 0.5 * site[2]
        return x, y

    def site_neighbor_directions_and_labels_and_colors(self, site):
        return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (site[0] - site[1] - 0 + 0) % 3),
                (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (site[0] - site[1] - 0 + 0) % 3),
                (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (site[0] - site[1] - 1 + 0) % 3),
                (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (site[0] - site[1] - 1 + 0) % 3),
                (np.array([[0, 0, 0],[0, 1, 1]]), 'XX', (site[0] - site[1] - 0 + 1) % 3),
                (np.array([[0, 0, 0],[0, 1, 1]]), 'ZZ', (site[0] - site[1] - 0 + 1) % 3)]

    def plaquette_offsets(self, ix, iy):
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 1, 0],
                         [1, 0, 1],
                         [1, 0, 0]])

    def get_sites_on_logical_path(self, direction: str):
        if direction == 'x':
            return [[ix, iy, s]
                    for ix in range(self.num_sites_x)
                    for iy in [0]
                    for s in [0, 1]]
        elif direction == 'y':
            return [[ix, iy, s]
                    for ix in [0]
                    for iy in range(self.num_sites_y)
                    for s in [0, 1]]

    def get_plaquette_color(self, coords):
        color = (coords[0] - coords[1]) % 3
        return ['r', 'g', 'b'][color]


class AsymmetricTorus(Geometry):
    def test_boundary_conditions(self):
        assert self.boundary_conditions == ('periodic', 'periodic')

    def site_to_physical_location(self, site):
        x = site[0] + site[2]
        y = 2 * site[1] + site[0] % 2
        return x, y

    def site_neighbor_directions_and_labels_and_colors(self, site):
        if site[0] % 2 == 0:
            return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (- site[1] + 0 + 0) % 3),
                    (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (- site[1] + 0 + 0) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (- site[1] + 0 - 1) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (- site[1] + 0 - 1) % 3),
                    (np.array([[0, 0, 0],[1, -1, 1]]), 'XX', (- site[1] + -1 - 1) % 3),
                    (np.array([[0, 0, 0],[1, -1, 1]]), 'ZZ', (- site[1] + -1 - 1) % 3)]
        else:
            return [(np.array([[0, 0, 0],[0, 0, 1]]), 'XX', (- site[1] + 0 + 0 + 1) % 3),
                    (np.array([[0, 0, 0],[0, 0, 1]]), 'ZZ', (- site[1] + 0 + 0 + 1) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'XX', (- site[1] + 1 + 0 + 1) % 3),
                    (np.array([[0, 0, 0],[1, 0, 1]]), 'ZZ', (- site[1] + 1 + 0 + 1) % 3),
                    (np.array([[0, 0, 0],[1, 1, 1]]), 'XX', (- site[1] + 1 + 1 + 1) % 3),
                    (np.array([[0, 0, 0],[1, 1, 1]]), 'ZZ', (- site[1] + 1 + 1 + 1) % 3)]


    def plaquette_offsets(self, ix, iy):
        hex_offsets_even = np.array([[0, 0, 1],
                                     [1, 0, 0],
                                     [1, 0, 1],
                                     [2, 0, 0],
                                     [1, -1, 1],
                                     [1, -1, 0]])
        hex_offsets_odd = np.array([[0, 0, 1],
                                    [1, 1, 0],
                                    [1, 1, 1],
                                    [2, 0, 0],
                                    [1, 0, 1],
                                    [1, 0, 0]])
        return hex_offsets_even if ix % 2 == 0 else hex_offsets_odd

    def get_sites_on_logical_path(self, direction: str):
        if direction == 'x':
            return [[ix, iy, s]
                    for ix in range(self.num_sites_x)
                    for iy in [0]
                    for s in [0, 1]]
        elif direction == 'y':
            return ([[ix, iy, s]
                     for ix in [0]
                     for iy in range(self.num_sites_y)
                     for s in [1]] +
                    [[ix, iy, s]
                     for ix in [1]
                     for iy in range(self.num_sites_y)
                     for s in [0]])

    def get_plaquette_color(self, coords):
        color = (coords[0] % 2 - coords[1]) % 3
        return ['r', 'g', 'b'][color]
