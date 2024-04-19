import stim
import numpy as np
from matplotlib import pyplot as plt
import pymatching

PAULI_LABELS_TO_INT = {'MXX': 0, 'MYY': 1, 'MZZ': 2}

class FloquetCode:
    def __init__(self, num_sites_x, num_sites_y, vortex_location=None, periodic_bc=False):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.vortex_location = vortex_location
        self.periodic_bc = periodic_bc
        self.bonds, self.plaquettes = self.get_bonds_plaquettes()
        self.bonds = sorted(self.bonds, key=lambda bond: bond.order)

    def get_bonds_plaquettes(self):
        bonds = []
        for ix in range(self.num_sites_x):
            for iy in range(self.num_sites_y):
                site1 = np.array([ix, iy, 1])
                for direction, pauli_label in zip([[0,0], [1,0], [0,1]], ['MXX', 'MYY', 'MZZ']):
                    # drop the edge bond if it is not periodic
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
        return bonds, plaquettes

    def get_bond_order(self, site1, pauli_label):
        pauli_label_int = PAULI_LABELS_TO_INT[pauli_label]
        return (site1[0] - site1[1] - pauli_label_int)%3

    def get_circuit(self, reps=1, before_parity_measure_2q_depolarization=None):
        circ = stim.Circuit()
        for _ in range(reps):
            for bond in self.bonds:
                qubit_pair = [self.site_to_index(bond.site1), self.site_to_index(bond.site2)]
                if before_parity_measure_2q_depolarization is not None:
                    circ.append_operation("DEPOLARIZE2", qubit_pair, before_parity_measure_2q_depolarization)
                circ.append(bond.pauli_label, qubit_pair)
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
    def __init__(self, bonds:list[Bond]):
        self.bonds = bonds

    def last_measure_index(self):
        return [bond.measurement_indexes[-1] for bond in self.bonds]


code = FloquetCode(2,2)


reps = 5
circ = code.get_circuit(reps=reps, before_parity_measure_2q_depolarization=0.1)
for rep in range(reps-1):
    record_targets = [stim.target_rec(rep*6 + i - 6*reps) for i in range(12)]
    circ.append_operation("DETECTOR", record_targets, [0,0,rep])

model = circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

sampler = circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)

predicted_observables = matching.decode_batch(syndrome)
num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(num_errors)

code.draw()
plt.show()

