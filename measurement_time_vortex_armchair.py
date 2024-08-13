import stim
import numpy as np
from matplotlib import pyplot as plt
import pymatching
from itertools import chain
from qiskit.quantum_info import Pauli
from simple_stabilizer import StabilizerGroup, PauliMeasurement
from stabilizer import pauli_weight

PAULI_LABELS_TO_INT = {'MXX': 2, 'MYY': 0, 'MZZ': 1, 'MYZ':0, 'MZY':1}


def site_to_physical_location(site):
    x = 1.5 * (2*site[0] + site[1]) + site[2]
    y = site[1] * np.sqrt(3) / 2
    return x, y


class FloquetCode:
    def __init__(self, num_sites_x, num_sites_y, vortex_location=None, periodic_bc=False, vortex_sign=-1):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.vortex_location = vortex_location
        self.vortex_sign = vortex_sign
        self.periodic_bc = periodic_bc if isinstance(periodic_bc,tuple) else (periodic_bc, periodic_bc)
        self.bonds = self.get_bonds()
        self.redefine_bond_order()
        self.plaquettes = self.get_plaquettes()
        for plaquette in self.plaquettes:
            plaquette.get_bonds(self.bonds)

    def get_bonds(self):
        bonds = []
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                site1 = np.array([ix, iy, 1])
                for direction, pauli_label in zip([np.array([0,0,1]), np.array([1,-1,1]), np.array([0,1,1])], ['MXX', 'MYY', 'MZZ']):
                    # drop the edge bond if it is not periodic
                    site2 = self.shift_site(site1, direction)
                    if site2 is None:
                        continue
                    order = self.get_bond_order(site1, site1+direction/2, pauli_label)
                    bond = Bond(site1, site2, pauli_label, order)
                    bonds.append(bond)

        if not self.periodic_bc[1]:
            direction = np.array([1, 0, 1])
            for ix in range(self.num_sites_x):
                site1 = np.array([ix, 0, 1])
                site2 = self.shift_site(site1, direction)
                if site2 is None:
                    continue
                pauli_label = 'MYZ'
                order = self.get_bond_order(site1, site1+direction/2, pauli_label)
                bond = Bond(site1, site2, pauli_label, order)
                bonds.append(bond)

            direction = np.array([1, 0, 1])
            for ix in range(self.num_sites_x):
                site1 = np.array([ix, self.num_sites_y-1, 1])
                site2 = self.shift_site(site1, direction)
                if site2 is None:
                    continue
                pauli_label = 'MZY'
                order = self.get_bond_order(site1, site1+direction/2, pauli_label)
                bond = Bond(site1, site2, pauli_label, order)
                bonds.append(bond)
        return bonds

    def shift_site(self, site, shift):
        shifted_site = site + shift
        if not self.periodic_bc[0]:
            if shifted_site[0] < 0 or shifted_site[0] >= self.num_sites_x:
                return None
        if not self.periodic_bc[1]:
            if shifted_site[1] < 0 or shifted_site[1] >= self.num_sites_y:
                return None
        return shifted_site % np.array([self.num_sites_x, self.num_sites_y, 2])

    def redefine_bond_order(self):
        self.bonds = sorted(self.bonds, key=lambda b: b.order)
        for bond in self.bonds:
            bond.order = np.round(bond.order*3*2)/2
        return

        for i, bond in enumerate(self.bonds):
            bond.order = 0
            for previous_bond in self.bonds[:i]:
                if previous_bond.overlaps(bond):
                    bond.order = max(previous_bond.order + 1, bond.order)

    def get_plaquettes(self):
        plaquettes = []
        # add hexagonal plaquettes
        hex_offsets = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [0, 1, 1],
                                [1, -1, 0],
                                [1, -1, 1],
                                [1, 0, 0]])
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in hex_offsets]
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                plaquettes.append(Plaquette(sites, [ix, iy]))
        
        if not self.periodic_bc[1]:
            bottom_square_offsets = np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [0, 1, 1],
                                              [1, 0, 0]])
            iy = 0
            for ix in range(self.num_sites_x):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in bottom_square_offsets]
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                plaquettes.append(Plaquette(sites, [ix, iy]))

            top_square_offsets = np.array([[0, 0, 1],
                                           [1, -1, 0],
                                           [1, -1, 1],
                                           [1, 0, 0]])
            iy = self.num_sites_y - 1
            for ix in range(self.num_sites_x):
                reference_site = np.array([ix, iy, 0])
                sites = [self.shift_site(reference_site, offset) for offset in top_square_offsets]
                # drop the plaquette if it is not periodic
                if np.any([s is None for s in sites]):
                    continue
                plaquettes.append(Plaquette(sites, [ix, iy]))
        return plaquettes

    def get_bond_order(self, site1, site_midpoint, pauli_label):
        pauli_label_int = PAULI_LABELS_TO_INT[pauli_label]
        order_without_vortex = (site1[1] + pauli_label_int)/3

        if self.vortex_location == 'x':
            location_dependent_delay = lambda site: site[0]/self.num_sites_x
        elif self.vortex_location == 'y':
            location_dependent_delay = lambda site: site[1]/self.num_sites_y
        else:
            location_dependent_delay = lambda site: 0

        order = order_without_vortex + location_dependent_delay(site_midpoint) * self.vortex_sign
        order = order%1
        return order

    def get_circuit(self, reps=12, reps_without_noise=4, before_parity_measure_2q_depolarization=None, logical_operator='inner_x'):
        assert reps%2==0
        circ = stim.Circuit()

        if logical_operator == 'inner_x':
            logical_operator_string = 'YZ' * self.num_sites_x
            sites_on_logical_path = [[ix, iy, s]
                                  for ix in range(self.num_sites_x)
                                  for iy in [0]
                                  for s in [0,1]]
            # Initialize data qubits along logical observable column into correct basis for observable to be deterministic.
        if logical_operator == 'outer_y':
            if self.periodic_bc[1]:
                logical_operator_string = 'XIYZ' + 'IYXIYZ' * (self.num_sites_y // 3 - 1) + 'IY'
            else:
                logical_operator_string = 'IYXI' + 'YIZYXI'*(self.num_sites_y//3-1) + 'YI'
            sites_on_logical_path = [[ix, iy, s]
                                  for ix in [0]
                                  for iy in range(self.num_sites_y)
                                  for s in [0,1]]
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
        # indexes where the logical operator is X, Y, Z
        x_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'X']
        y_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'Y']
        z_initialized = [i for i, p in enumerate(logical_pauli.to_label()) if p == 'Z']

        circ.append_operation("H", x_initialized)
        circ.append_operation("H_YZ", y_initialized)

        stabilizer = StabilizerGroup()

        i_meas = 0
        measurements_to_include_in_logical = set()
        for rep in range(reps):
            for ibond, bond in enumerate(self.bonds):
                # check if logical operator anticommutes with the bond, if it does, multiply it by another bond in the stabilizer that anticommutes with the bond
                if logical_pauli.anticommutes(self.bond_to_full_pauli(bond)):
                    self.draw_pauli(logical_pauli)
                    anticommuting_generators = [p for p in stabilizer.paulis if p.pauli.anticommutes(self.bond_to_full_pauli(bond))]
                    if len(anticommuting_generators) == 0:
                        raise ValueError('Logical operator anticommutes with a bond that is not in the stabilizer')
                    # choose the minimal weight generator
                    p = min(anticommuting_generators, key=lambda p: len(p.index))#pauli_weight(logical_pauli.compose(p.pauli)))
                    logical_pauli = logical_pauli.compose(p.pauli)
                    # update the observable with the last index at which b was measured
                    # the measurements_to_include is the union minus the intersection with the new measurements
                    measurements_to_include_in_logical = measurements_to_include_in_logical.union(p.index) - measurements_to_include_in_logical.intersection(p.index)

                #update bonds in stabilizer by adding the current bond and removing overlapping bonds
                stabilizer.measure_pauli(PauliMeasurement(self.bond_to_full_pauli(bond), [i_meas]))
                print(len(stabilizer.paulis))

                qubit_pair = [self.site_to_index(bond.site1), self.site_to_index(bond.site2)]
                if before_parity_measure_2q_depolarization is not None and rep>=reps_without_noise and rep<reps-reps_without_noise:
                    circ.append_operation("DEPOLARIZE2", qubit_pair, before_parity_measure_2q_depolarization)
                if bond.pauli_label == 'MYZ':
                    circ.append_operation("H_YZ", qubit_pair[0])
                    circ.append('MZZ', qubit_pair)
                    circ.append_operation("H_YZ", qubit_pair[0])
                elif bond.pauli_label == 'MZY':
                    circ.append_operation("H_YZ", qubit_pair[1])
                    circ.append('MZZ', qubit_pair)
                    circ.append_operation("H_YZ", qubit_pair[1])
                else:
                    circ.append(bond.pauli_label, qubit_pair)
                bond.measurement_indexes.append(i_meas)
                i_meas += 1

        # add detectors
        for plaq in self.plaquettes:
            n_bonds = len(plaq.bonds)
            plaq_measurement_idx = plaq.measurement_indexes()
            plaq_coords = plaq.coords
            rep = 0
            for i in range(len(plaq_measurement_idx) - n_bonds * 2 + 1):
                record_targets = [stim.target_rec(plaq_measurement_idx[j] - i_meas) for j in range(i, i+n_bonds * 2)]
                new_circ = circ.copy()
                new_circ.append_operation("DETECTOR", record_targets, [plaq_coords[0], plaq_coords[1], rep])
                try:
                    new_circ.detector_error_model(decompose_errors=True)
                    circ = new_circ
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
        return circ

    def site_to_index(self, site):
        return np.ravel_multi_index(site, (self.num_sites_x, self.num_sites_y, 2))

    def index_to_site(self, index):
        return np.unravel_index(index, (self.num_sites_x, self.num_sites_y, 2))

    def bond_to_full_pauli(self, bond):
        full_pauli = ['I'] * self.num_sites_x * self.num_sites_y * 2
        site1_index = self.site_to_index(bond.site1)
        site2_index = self.site_to_index(bond.site2)
        full_pauli[site1_index] = bond.pauli_label[1]
        full_pauli[site2_index] = bond.pauli_label[2]
        return Pauli(''.join(full_pauli))

    def draw(self):
        fig, ax = plt.subplots()
        for bond in self.bonds:
            x1, y1 = site_to_physical_location(bond.site1)
            x2, y2 = site_to_physical_location(bond.site2)
            ax.plot([x1, x2], [y1, y2], 'k')
            x = (x1+x2)/2
            y = (y1+y2)/2
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 5:
                fontsize = 10
            else:
                fontsize = 15
                x = x + 0.1*np.random.randn()
                y = y + 0.1*np.random.randn()
            ax.text(x, y, str(bond.order)+bond.pauli_label, fontsize=fontsize, ha='center', va='center')
        ax.set_aspect('equal')

    def draw_pauli(self, pauli:Pauli):
        fig, ax = plt.subplots(figsize=(15, 8))
        for i, pp in enumerate(pauli[::-1]):
            p = pp.to_label()
            site = self.index_to_site(i)
            x, y = site_to_physical_location(site)
            if p == 'I':
                ax.plot(x, y, 'ko')
            if p == 'X':
                ax.plot(x, y, 'ro', markersize=10)
            if p == 'Y':
                ax.plot(x, y, 'go', markersize=10)
            if p == 'Z':
                ax.plot(x, y, 'bo', markersize=10)
        for bond in self.bonds:
            x1, y1 = site_to_physical_location(bond.site1)
            x2, y2 = site_to_physical_location(bond.site2)
            ax.plot([x1, x2], [y1, y2], 'k')
            x = (x1+x2)/2
            y = (y1+y2)/2
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 5:
                fontsize = 10
            else:
                fontsize = 15
                x = x + 0.1*np.random.randn()
                y = y + 0.1*np.random.randn()
            ax.text(x, y, str(bond.order)+bond.pauli_label, fontsize=fontsize, ha='center', va='center')
        ax.set_aspect('equal')
        plt.show()


class Bond:
    def __init__(self, site1:tuple, site2:tuple, pauli_label:str, order:int):
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
    def __init__(self, sites:list[np.ndarray], coords:list):
        self.sites = sites
        self.bonds = []
        self.coords = coords

    def get_bonds(self, all_bonds):
        bonds = []
        for bond in all_bonds:
            if np.all(bond.site1 == self.sites, axis=1).any() and np.all(bond.site2 == self.sites, axis=1).any():
                bonds.append(bond)
        self.bonds = bonds

    def measurement_indexes(self):
        all_indexes = [bond.measurement_indexes for bond in self.bonds]
        all_indexes = list(chain(*all_indexes))
        return sorted(all_indexes)

d_list = [3,6,9]
phys_err_rate_list = [0.001,0.003,0.01,0.03,0.1,0.3]#np.linspace(0.,0.15, 15)#[0.01]# #0.03 #
shots = 100000
log_err_rate = np.zeros((len(d_list), len(phys_err_rate_list)))
reps = 12
reps_without_noise = 4

for id,d in enumerate(d_list):
    for ierr_rate,phys_err_rate in enumerate(phys_err_rate_list):
        code = FloquetCode(d, d, periodic_bc=(True, False), vortex_location='x', vortex_sign=1)
        circ = code.get_circuit(reps=reps, reps_without_noise=reps_without_noise, before_parity_measure_2q_depolarization=phys_err_rate,
                                logical_operator='outer_y')#'
        model = circ.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        sampler = circ.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

        predicted_observables = matching.decode_batch(syndrome)
        num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

        print("logical error_rate", num_errors/shots)
        log_err_rate[id, ierr_rate] = num_errors / shots
        # print(circ)



plt.figure()
for id in range(len(d_list)):
    plt.errorbar(phys_err_rate_list, log_err_rate[id, :], np.sqrt(log_err_rate[id, :] * (1-log_err_rate[id, :]) / shots))
plt.xlabel('physical error rate')
plt.ylabel('logical error rate')
plt.yscale('log')
plt.xscale('log')
plt.legend(d_list, title='code size')
plt.tight_layout()
plt.show()

