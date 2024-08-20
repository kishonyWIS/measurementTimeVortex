from copy import copy

import numpy as np


class Geometry:
    def __init__(self, num_sites_x: int, num_sites_y: int):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y

    def site_to_physical_location(self, site):
        """
        Convert a site index to a physical location on the lattice.
        """
        pass

    def site_neighbor_directions(self, site):
        """
        Return the directions of the neighbors of a site.
        """
        pass

    def plaquette_offsets(self, ix, iy):
        """
        Return the offsets of the sites in a plaquette.
        """
        pass

    def get_bond_color(self, direction, site1):
        """
        Return the color of a bond.
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
            if np.linalg.norm(site[0] - reference_site[0]) > 1:
                new_site = new_site + np.array(
                    [self.num_sites_x * round((reference_site[0] - site[0]) / self.num_sites_x), 0, 0])
            if np.linalg.norm(site[1] - reference_site[1]) > 1:
                new_site = new_site + np.array(
                    [0, self.num_sites_y * round((reference_site[1] - site[1]) / self.num_sites_y), 0])
            shifted_sites.append(new_site)
        was_shifted = not all([np.allclose(site, shifted_site) for site, shifted_site in zip(sites, shifted_sites)])
        if return_was_shifted:
            return shifted_sites, was_shifted
        else:
            return shifted_sites

# class SymmetricCylinder(Geometry):
#     #periodic in x, open in y
#     def site_to_physical_location(self, site):
#         x = np.sqrt(3) * site[0] + np.sqrt(3) / 2 * site[1] + np.sqrt(3) / 2 * site[2]
#         y = 1.5 * site[1] + 0.5 * site[2]
#         return x, y
#
#     def site_neighbor_directions(self, site):
#         if site[1] == self.num_sites_y-1:
#             return [np.array([0, 0, 1]), np.array([1, 0, 1])]
#         else:
#             return [np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([0, 1, 1])]
#
#     def plaquette_offsets(self, ix, iy):
#         return np.array([[0, 0, 1],
#                          [0, 1, 0],
#                          [0, 1, 1],
#                          [1, 1, 0],
#                          [1, 0, 1],
#                          [1, 0, 0]])
#
#     def get_bond_color(self, direction, site1):
#         return (site1[0] - site1[1] - direction[0] + direction[1]) % 3
#
#     def get_sites_on_logical_path(self, direction: str):
#         if direction == 'x':
#             return [[ix, iy, s]
#                     for ix in range(self.num_sites_x)
#                     for iy in [0]
#                     for s in [0, 1]]
#         elif direction == 'y':
#             return [[ix, iy, s]
#                     for ix in [0]
#                     for iy in range(self.num_sites_y)
#                     for s in [0, 1]]
#
#     def get_plaquette_color(self, coords):
#         color = (coords[0] - coords[1]) % 3
#         return ['r', 'g', 'b'][color]


class SymmetricTorus(Geometry):
    def site_to_physical_location(self, site):
        x = np.sqrt(3) * site[0] + np.sqrt(3) / 2 * site[1] + np.sqrt(3) / 2 * site[2]
        y = 1.5 * site[1] + 0.5 * site[2]
        return x, y

    def site_neighbor_directions(self, site):
        return [np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([0, 1, 1])]

    def plaquette_offsets(self, ix, iy):
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 1, 0],
                         [1, 0, 1],
                         [1, 0, 0]])

    def get_bond_color(self, direction, site1):
        return (site1[0] - site1[1] - direction[0] + direction[1]) % 3

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
    def site_to_physical_location(self, site):
        x = site[0] + site[2]
        y = 2 * site[1] + site[0] % 2
        return x, y

    def site_neighbor_directions(self, site):
        return [np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([1, -1, 1])] if site[0] % 2 == 0 else [
            np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([1, 1, 1])]

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

    def get_bond_color(self, direction, site1):
        if site1[0] % 2 == 0:
            return (-site1[1] + direction[1] - direction[0]) % 3
        else:
            return (-site1[1] - 2 - 2 * direction[0] - 2 * direction[1]) % 3

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
