from typing import Optional

import stim
import numpy as np
from matplotlib import pyplot as plt, patches
import pymatching
from itertools import chain
from qiskit.quantum_info import Pauli


def site_to_physical_location(site):
    x = np.sqrt(3) * site[0] + np.sqrt(3) / 2 * site[1] + np.sqrt(3) / 2 * site[2]
    y = 1.5 * site[1] + 0.5 * site[2]
    return x, y

def cyclic_permute(s, n=1):
    """Cyclically permute the string `s` by `n` positions."""
    n = n % len(s)  # Ensure n is within the bounds of the string length
    return s[n:] + s[:n]

class FloquetCode:
    def __init__(self, num_sites_x, num_sites_y, vortex_location=None, boundary_conditions=('periodic', 'periodic'),
                 vortex_sign=-1):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.vortex_location = vortex_location
        self.vortex_sign = vortex_sign
        self.boundary_conditions = boundary_conditions
        self.bonds = self.get_bonds()
        self.plaquettes = self.get_plaquettes()
        for plaquette in self.plaquettes:
            plaquette.get_bonds(self.bonds)

    def get_bonds(self):
        bonds = []
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                site1 = np.array([ix, iy, 1])
                for direction in [np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([0, 1, 1])]:
                    # drop the edge bond if it is not periodic
                    site2 = self.shift_site(site1, direction)
                    if site2 is None:
                        continue
                    for pauli_label in ['XX', 'ZZ']:
                        order = self.get_bond_order(site1, direction, pauli_label)
                        bond = Bond(site1, site2, pauli_label, order)
                        bonds.append(bond)
        bonds = sorted(bonds, key=lambda b: b.order)
        return bonds

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
        hex_offsets = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [0, 1, 1],
                                [1, 1, 0],
                                [1, 0, 1],
                                [1, 0, 0]])
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in hex_offsets]
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                for pauli_label in ['X', 'Z']:
                    plaquettes.append(Plaquette(sites, [ix, iy], pauli_label))
        return plaquettes

    def get_bond_order(self, site1, direction, pauli_label):
        site_midpoint = site1 + direction / 2
        color = self.get_bond_color(direction, site1)
        order_without_vortex = (-color/3 + (pauli_label == 'ZZ')/2) % 1

        if self.vortex_location == 'x':
            location_dependent_delay = lambda site: site[0] / self.num_sites_x
        elif self.vortex_location == 'y':
            location_dependent_delay = lambda site: site[1] / self.num_sites_y
        else:
            location_dependent_delay = lambda site: 0

        order = order_without_vortex + location_dependent_delay(site_midpoint) * self.vortex_sign
        order = order % 1
        return order

    def get_bond_color(self, direction, site1):
        return (site1[0] - site1[1] - direction[0] + direction[1]) % 3

    def get_circuit(self, reps=12, reps_without_noise=4, noise_rate=0.01, noise_type=None,
                    logical_operator_color='blue', logical_operator_pauli_type='X', logical_operator_direction='x',
                    detector_indexes=None, detector_args=None):
        assert reps % 2 == 0
        circ = stim.Circuit()

        if logical_operator_direction == 'x':
            sites_on_logical_path = [[ix, iy, s]
                                     for ix in range(self.num_sites_x)
                                     for iy in [0]
                                     for s in [0, 1]]
            permutation_amount = {'red': 0, 'green': 1, 'blue': 2}
        elif logical_operator_direction == 'y':
            sites_on_logical_path = [[ix, iy, s]
                                     for ix in [0]
                                     for iy in range(self.num_sites_y)
                                     for s in [0, 1]]
            permutation_amount = {'red': 0, 'green': 2, 'blue': 1}
        logical_operator_string = (2 * logical_operator_pauli_type + 'I') * (len(sites_on_logical_path) // 3)
        logical_operator_string = cyclic_permute(logical_operator_string, permutation_amount[logical_operator_color])

        logical_operator_string = []
        for i_along_path, site in enumerate(sites_on_logical_path):
            bonds_connected_to_site = [bond for bond in self.bonds if np.all(bond.site1 == site) or np.all(bond.site2 == site)]
            append_logical = 0 # 0: I, 1: logical_operator_pauli_type
            for bond in bonds_connected_to_site:
                if bond.pauli_label[0] != logical_operator_pauli_type:
                    break
                if np.all(bond.site1 == sites_on_logical_path, axis=1).any() and np.all(bond.site2 == sites_on_logical_path, axis=1).any():
                    append_logical += 1
            # check if the bond is fully contained in the logical path
            if np.all(bond.site1 == sites_on_logical_path, axis=1).any() and np.all(bond.site2 == sites_on_logical_path, axis=1).any():
                append_logical += 1
            logical_operator_string.append(logical_operator_pauli_type if append_logical%2 else 'I')
        logical_operator_string = ''.join(logical_operator_string)



        # if self.num_sites_x == 9:
        #     logical_operator_string = 'IXXIXXIXIXXIXIXXIX'
        # elif self.num_sites_x == 6:
        #     logical_operator_string = 'IXXIXIXXIXIX'
        # Initialize data qubits along logical observable column into correct basis for observable to be deterministic.
        all_sites = [[ix, iy, s]
                     for ix in range(self.num_sites_x)
                     for iy in range(self.num_sites_y)
                     for s in [0, 1]]

        full_logical_operator_string = []
        for site in all_sites:
            if site in sites_on_logical_path:
                full_logical_operator_string.append(logical_operator_string[sites_on_logical_path.index(site)])
            else:
                full_logical_operator_string.append('I')
        full_logical_operator_string = ''.join(full_logical_operator_string)
        logical_pauli = Pauli(full_logical_operator_string)
        self.draw_pauli(logical_pauli)

        # indexes where the logical operator is X, Y, Z
        x_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'X']
        y_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'Y']
        z_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'Z']

        circ.append_operation("H", x_initialized)
        circ.append_operation("H_YZ", y_initialized)

        i_meas = 0
        measurements_to_include_in_logical = set()
        for rep in range(reps):
            for ibond, bond in enumerate(self.bonds):


                qubit_pair = [self.site_to_index(bond.site1), self.site_to_index(bond.site2)]
                if noise_rate is not None and rep >= reps_without_noise and rep < reps - reps_without_noise:
                    circ.append_operation(noise_type, qubit_pair, noise_rate)
                circ.append('M' + bond.pauli_label, qubit_pair)
                bond.measurement_indexes.append(i_meas)

                # if the measured bond is in sites_on_logical_path, and of the same pauli type as the logical operator, include it in the logical operator
                if (np.all(bond.site1 == sites_on_logical_path, axis=1).any() and
                        np.all(bond.site2 == sites_on_logical_path, axis=1).any() and
                        bond.pauli_label[0] == logical_operator_pauli_type):
                    # self.draw_pauli(logical_pauli)
                    measurements_to_include_in_logical.add(i_meas)
                    logical_pauli = logical_pauli.compose(self.bond_to_full_pauli(bond))
                i_meas += 1

        # add detectors
        if detector_indexes is not None and detector_args is not None:
            for i, indexes in enumerate(detector_indexes):
                circ.append_operation("DETECTOR", list(map(stim.target_rec, indexes)), detector_args[i])
        else:
            detector_indexes = []
            detector_args = []
            for plaq in self.plaquettes:
                n_bonds = len(plaq.bonds)
                plaq_measurement_idx = plaq.measurement_indexes()
                plaq_coords = plaq.coords
                rep = 0
                for i in range(len(plaq_measurement_idx) - n_bonds + 1):
                    record_targets = [stim.target_rec(plaq_measurement_idx[j] - i_meas) for j in range(i, i + n_bonds)]
                    new_circ = circ.copy()
                    new_circ.append_operation("DETECTOR", record_targets, [plaq_coords[0], plaq_coords[1], rep, plaq.pauli_label=='X'])
                    try:
                        new_circ.detector_error_model(decompose_errors=True)
                        circ = new_circ
                        detector_indexes.append([plaq_measurement_idx[j] - i_meas for j in range(i, i + n_bonds)])
                        detector_args.append([plaq_coords[0], plaq_coords[1], rep, plaq.pauli_label=='X'])
                        rep += 1
                    except:
                        pass

        # include measurements in the dynamics of the observable
        for i_to_include in measurements_to_include_in_logical:
            circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(i_to_include - i_meas), 0)

        # Finish circuit with data measurements according to logical operator
        x_finalized = [i for i, p in enumerate(logical_pauli[::-1]) if p.to_label() == 'X']
        y_finalized = [i for i, p in enumerate(logical_pauli[::-1]) if p.to_label() == 'Y']
        z_finalized = [i for i, p in enumerate(logical_pauli[::-1]) if p.to_label() == 'Z']
        circ.append_operation("MX", x_finalized)
        circ.append_operation("MY", y_finalized)
        circ.append_operation("MZ", z_finalized)
        observable_length = len(x_finalized) + len(y_finalized) + len(z_finalized)
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - observable_length)
                               for i in range(observable_length)],
                              0)
        return circ, detector_indexes, detector_args

    def site_to_index(self, site):
        return np.ravel_multi_index(site, (self.num_sites_x, self.num_sites_y, 2))

    def index_to_site(self, index):
        return np.unravel_index(index, (self.num_sites_x, self.num_sites_y, 2))

    def bond_to_full_pauli(self, bond):
        full_pauli = ['I'] * self.num_sites_x * self.num_sites_y * 2
        site1_index = self.site_to_index(bond.site1)
        site2_index = self.site_to_index(bond.site2)
        full_pauli[site1_index] = bond.pauli_label[0]
        full_pauli[site2_index] = bond.pauli_label[1]
        return Pauli(''.join(full_pauli))

    def draw_pauli(self, pauli: Pauli):
        fig, ax = plt.subplots(figsize=(15, 10))

        for plaquette in self.plaquettes:
            if plaquette.pauli_label == 'X':
                continue
            if self.boundary_conditions[0] == 'periodic' and plaquette.coords[0] == self.num_sites_x - 1:
                continue
            if self.boundary_conditions[1] == 'periodic' and plaquette.coords[1] == self.num_sites_y - 1:
                continue
            color = (plaquette.coords[0] - plaquette.coords[1]) % 3
            color = ['r', 'g', 'b'][color]
            # draw a shaded polygon for the plaquette
            points = list(map(site_to_physical_location, plaquette.sites))
            polygon = patches.Polygon(points, closed=True, edgecolor=None, facecolor=color, alpha=0.5)
            # Add the polygon to the plot
            ax.add_patch(polygon)
        for bond in self.bonds:
            # if the two sites are far apart, the bond is an edge bond should be plotted as if site2 is the shifted site
            site1, site2 = bond.site1.copy(), bond.site2.copy()
            x1, y1 = site_to_physical_location(bond.site1)
            if np.linalg.norm(site1[0] - site2[0]) > 1:
                site2 = site2 + np.array([self.num_sites_x,0,0])
            if np.linalg.norm(site1[1] - site2[1]) > 1:
                site2 = site2 + np.array([0,self.num_sites_y,0])
            x2, y2 = site_to_physical_location(site2)
            ax.plot([x1, x2], [y1, y2], 'k')
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            fontsize = 10
            y = y + (bond.pauli_label == 'XX')*0.2 - (bond.pauli_label == 'ZZ')*0.2
            ax.text(x, y, '{:.1f}'.format(bond.order*6) + bond.pauli_label, fontsize=fontsize, ha='center', va='center')
        for i, pp in enumerate(pauli[::-1]):
            p = pp.to_label()
            site = self.index_to_site(i)
            x, y = site_to_physical_location(site)
            if p == 'I':
                ax.plot(x, y, 'ko')
            else:
                ax.plot(x, y, 'co', markersize=20)
                ax.text(x, y, p, fontsize=20, ha='center', va='center')
        ax.set_aspect('equal')
        plt.show()


class Bond:
    def __init__(self, site1: np.ndarray, site2: np.ndarray, pauli_label: str, order: int):
        self.site1 = site1
        self.site2 = site2
        self.pauli_label = pauli_label
        self.order = order
        self.measurement_indexes = []

    def __repr__(self):
        return f'Bond({self.site1}, {self.site2}, {self.pauli_label}, {self.order})'

    def overlaps(self, other):
        return ((self.site1 == other.site1).all() or
                (self.site1 == other.site2).all() or
                (self.site2 == other.site1).all() or
                (self.site2 == other.site2).all())


class Plaquette:
    def __init__(self, sites: list[np.ndarray], coords: list, pauli_label: Optional[str] = None):
        self.sites = sites
        self.bonds = []
        self.coords = coords
        self.pauli_label = pauli_label

    def get_bonds(self, all_bonds):
        bonds = []
        for bond in all_bonds:
            if np.all(bond.site1 == self.sites, axis=1).any() and np.all(bond.site2 == self.sites, axis=1).any():
                if self.pauli_label is None or np.all([p == self.pauli_label for p in bond.pauli_label]):
                    bonds.append(bond)
        self.bonds = bonds

    def measurement_indexes(self):
        all_indexes = [bond.measurement_indexes for bond in self.bonds]
        all_indexes = list(chain(*all_indexes))
        return sorted(all_indexes)


d_list = [6]
phys_err_rate_list = [0.003, 0.01, 0.03]  #np.linspace(0.,0.15, 15)#[0.01]# #0.03 #
shots = 100000
log_err_rate = np.zeros((len(d_list), len(phys_err_rate_list)))
reps = 12
reps_without_noise = 4
noise_type = 'DEPOLARIZE1'

for id, d in enumerate(d_list):
    detector_indexes = None
    detector_args = None
    for ierr_rate, phys_err_rate in enumerate(phys_err_rate_list):
        code = FloquetCode(d, 3, boundary_conditions=('periodic','periodic'), vortex_location='x', vortex_sign=1)
        circ, detector_indexes, detector_args = code.get_circuit(
            reps=reps, reps_without_noise=reps_without_noise,
            noise_rate=phys_err_rate, noise_type=noise_type,
            logical_operator_color='blue', logical_operator_pauli_type='X', logical_operator_direction='y',
            detector_indexes=detector_indexes, detector_args=detector_args)
        # blue, X, x
        # blue, X, y
        # red, Z, x
        # red, Z, y
        model = circ.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        sampler = circ.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

        predicted_observables = matching.decode_batch(syndrome)
        num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

        print("logical error_rate", num_errors / shots)
        log_err_rate[id, ierr_rate] = num_errors / shots
        # print(circ)

plt.figure()
for id in range(len(d_list)):
    plt.errorbar(phys_err_rate_list, log_err_rate[id, :],
                 np.sqrt(log_err_rate[id, :] * (1 - log_err_rate[id, :]) / shots))
plt.xlabel('physical error rate')
plt.ylabel('logical error rate')
plt.yscale('log')
plt.xscale('log')
plt.legend(d_list, title='code size')
plt.tight_layout()
plt.show()
