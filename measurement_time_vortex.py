import stim
import numpy as np
from matplotlib import pyplot as plt
import pymatching
from itertools import chain

PAULI_LABELS_TO_INT = {'MXX': 0, 'MYY': 1, 'MZZ': 2}

class FloquetCode:
    def __init__(self, num_sites_x, num_sites_y, vortex_location=None, periodic_bc=False):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.vortex_location = vortex_location
        self.periodic_bc = periodic_bc
        self.bonds = self.get_bonds()
        self.bonds = sorted(self.bonds, key=lambda bond: bond.order)
        self.plaquettes = self.get_plaquettes()
        for plaquette in self.plaquettes:
            plaquette.get_bonds(self.bonds)

    def get_bonds(self):
        bonds = []
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                site1 = np.array([ix, iy, 1])
                for direction, pauli_label in zip([[0,0], [1,0], [0,1]], ['MXX', 'MYY', 'MZZ']):
                    # drop the edge bond if it is not periodic
                    if self.periodic_bc is False:
                        if (ix == self.num_sites_x-1 and iy == self.num_sites_y-1 and pauli_label == 'MXX') or \
                                (ix == 0 and iy == 0 and pauli_label == 'MXX'):
                            continue
                    site2 = site1 + np.array([direction[0], direction[1], 0])
                    site2[2] = 0
                    if self.periodic_bc is True:
                        site2[0] = site2[0] % self.num_sites_x
                        site2[1] = site2[1] % self.num_sites_y
                    elif self.periodic_bc == [True, False]:
                        site2[0] = site2[0] % self.num_sites_x
                    elif self.periodic_bc == [False, True]:
                        site2[1] = site2[1] % self.num_sites_y
                    else:
                        pass
                    # check if site2 is outside the lattice
                    if site2[0] < 0 or site2[0] >= self.num_sites_x or site2[1] < 0 or site2[1] >= self.num_sites_y:
                        continue
                    order = self.get_bond_order(site1, pauli_label)
                    bond = Bond(site1, site2, pauli_label, order)
                    bonds.append(bond)
        return bonds

    def get_plaquettes(self):
        plaquettes = []
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                sites = np.array([[ix, iy, 1],
                                  [ix+1, iy, 0],
                                  [ix, iy+1, 0],
                                  [ix+1, iy, 1],
                                  [ix, iy+1, 1],
                                  [ix+1, iy+1, 0]])
                sites[:,:2] %= [self.num_sites_x, self.num_sites_y]
                # drop the plaquette if it is not periodic
                if self.periodic_bc is False and (ix == self.num_sites_x-1 or iy == self.num_sites_y-1):
                    continue
                if self.periodic_bc == [True, False] and ix == self.num_sites_x-1:
                    continue
                if self.periodic_bc == [False, True] and iy == self.num_sites_y-1:
                    continue
                plaquettes.append(Plaquette(sites, [ix, iy]))
        return plaquettes

    def get_bond_order(self, site1, pauli_label):
        pauli_label_int = PAULI_LABELS_TO_INT[pauli_label]
        return (site1[0] - site1[1] - pauli_label_int)%3

    def get_circuit(self, reps=1, before_parity_measure_2q_depolarization=None):
        circ = stim.Circuit()

        # Initialize data qubits along logical observable column into correct basis for observable to be deterministic.
        # circ.append_operation("H", x_initialized)
        # circ.append_operation("H_YZ", y_initialized)

        i_meas = 0
        for rep in range(reps):
            for bond in self.bonds:
                qubit_pair = [self.site_to_index(bond.site1), self.site_to_index(bond.site2)]
                if before_parity_measure_2q_depolarization is not None and rep>=2 and rep<reps-2:
                    circ.append_operation("DEPOLARIZE2", qubit_pair, before_parity_measure_2q_depolarization)
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

        # Finish circuit with data measurements.
        qubits_for_observable = [self.site_to_index([ix, 0, s])
                                 for ix in range(self.num_sites_x)
                                 for s in range(2)]
        circ.append_operation("M", qubits_for_observable)
        circ.append_operation("OBSERVABLE_INCLUDE",
                                      [stim.target_rec(i - len(qubits_for_observable)) for i in range(len(qubits_for_observable))],
                                      0)
        return circ

    def site_to_index(self, site):
        return np.ravel_multi_index(site, (self.num_sites_x, self.num_sites_y, 2))

    def site_to_physical_location(self, site):
        x = 1.5 * (site[0] + site[1]) + site[2]
        y = (site[1] - site[0]) * np.sqrt(3) / 2
        return x, y

    def draw(self):
        fig, ax = plt.subplots()
        for bond in self.bonds:
            x1, y1 = self.site_to_physical_location(bond.site1)
            x2, y2 = self.site_to_physical_location(bond.site2)
            ax.plot([x1, x2], [y1, y2], 'k')
            ax.text((x1+x2)/2, (y1+y2)/2, bond.pauli_label+str(bond.order), fontsize=8, ha='center', va='center')
        ax.set_aspect('equal')


class Bond:
    def __init__(self, site1:tuple, site2:tuple, pauli_label:str, order:int):
        self.site1 = site1
        self.site2 = site2
        self.pauli_label = pauli_label
        self.order = order
        self.measurement_indexes = []

    def __repr__(self):
        return f'Bond({self.site1}, {self.site2}, {self.pauli_label}, {self.order})'


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

d_list = [3, 6, 9]
phys_err_rate_list = np.linspace(0.,0.15, 16)
shots = 100000
log_err_rate = np.zeros((len(d_list), len(phys_err_rate_list)))
for id,d in enumerate(d_list):
    for ierr_rate,phys_err_rate in enumerate(phys_err_rate_list):
        code = FloquetCode(d, d, periodic_bc=True)

        reps = 7
        circ = code.get_circuit(reps=reps, before_parity_measure_2q_depolarization=phys_err_rate)
        model = circ.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        # print(circ)
        sampler = circ.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

        predicted_observables = matching.decode_batch(syndrome)
        num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

        print("logical error_rate", num_errors/shots)
        log_err_rate[id, ierr_rate] = num_errors / shots
        # code.draw()
        # plt.show()

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

