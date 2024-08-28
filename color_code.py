import os
from copy import copy

import numpy as np

from entanglement import get_num_logical_qubits
import stim
import pymatching
from itertools import product
import pandas as pd
from lattice import *
from noise import get_noise_model


def cyclic_permute(s, n=1):
    """Cyclically permute the string `s` by `n` positions."""
    n = n % len(s)  # Ensure n is within the bounds of the string length
    return s[n:] + s[:n]

class Bond:
    def __init__(self, sites: list, pauli_label: str, order: int):
        assert len(sites) == len(pauli_label)
        self.sites = sites
        self.pauli_label = pauli_label
        self.order = order
        self.measurement_indexes = []

    def __repr__(self):
        return f'Bond({self.sites}, {self.pauli_label}, {self.order})'

    def overlaps(self, other):
        return np.any([np.all(self_site == other_site) for self_site, other_site in product(self.sites, other.sites)])

    def pauli_length(self):
        return len(self.pauli_label.replace('I', ''))


class PlaquetteStabilizer:
    def __init__(self, sites: list[tuple], bonds:list[Bond], coords: tuple, pos: tuple, pauli_label: str = None):
        self.sites = sites
        self.bonds = bonds
        self.coords = coords
        self.pos = pos
        self.pauli_label = pauli_label

    def measurement_indexes_and_sizes(self):
        all_indexes = [(index, bond.pauli_length()) for bond in self.bonds for index in bond.measurement_indexes]
        return sorted(all_indexes, key=lambda x: x[0])


class FloquetCode:
    def __init__(self, lat: Lattice, num_vortexes=(0, 0), detectors=('X', 'Z')):
        self.num_vortexes = num_vortexes
        self.detectors = detectors
        self.lat = lat
        self.bonds = self.get_bonds()
        self.plaquettes = self.get_plaquettes()

    def get_bonds(self):
        bonds = []
        for edge, edge_data in self.lat.G.edges.items():
            edge_data['bonds'] = []
            # sort the sites so that the first is not a boundary site
            sites = [edge[0], edge[1]]
            sites = sorted(sites, key=lambda s: self.lat.G.nodes[s]['boundary'] is not False)
            boundary_type = self.lat.G.nodes[sites[1]]['boundary']
            if not boundary_type:
                pauli_labels = ['XX', 'ZZ']
            elif boundary_type == 'X':
                pauli_labels = ['XI']
            elif boundary_type == 'Z':
                pauli_labels = ['ZI']
            else:
                raise ValueError(f'Unknown boundary type {boundary_type}')
            for pauli_label in pauli_labels:
                order = self.get_bond_order(edge_data, pauli_label)
                bond = Bond(sites, pauli_label, order)
                bonds.append(bond)
                edge_data['bonds'].append(bond)
        bonds = sorted(bonds, key=lambda b: b.order)
        return bonds

    def get_plaquettes(self):
        plaquettes = []
        for coords, data in self.lat.plaquettes.items():
            sites = data.sites
            edges = data.edges
            pos = data.pos
            for pauli_label, opposite_pauli_label in zip(['X', 'Z'], ['Z', 'X']):
                # if there is a boundary site of the opposite pauli label, continue
                if any([self.lat.G.nodes[s]['boundary'] == opposite_pauli_label for s in sites]):
                    continue
                bonds = []
                for edge in edges:
                    for bond in self.lat.G.edges[edge]['bonds']:
                        if pauli_label in bond.pauli_label:
                            bonds.append(bond)
                plaquettes.append(PlaquetteStabilizer(sites=sites, bonds=bonds, coords=coords, pos=pos, pauli_label= pauli_label))
        return plaquettes

    @property
    def num_data_qubits(self):
        return len(self.lat.site_to_index)

    @property
    def measurement_ancilla(self):
        return self.num_data_qubits

    @property
    def boundary_ancilla(self):
        return self.num_data_qubits + 1

    def location_dependent_delay(self, edge_data):
        if type(self.lat) in [HexagonalLatticeGidneyOnCylinder, HexagonalLatticeGidney]:
            pos = edge_data['pos']
            return (pos[0]/np.linalg.norm(self.lat.lattice_vectors[0]) / self.lat.size[0] * self.num_vortexes[0] +
                    pos[1]/np.linalg.norm(self.lat.lattice_vectors[1]) / self.lat.size[1] * self.num_vortexes[1])
        elif 'Sheared' in type(self.lat).__name__:
            coords = edge_data['coords']
            return (coords[0] / self.lat.size[0] * self.num_vortexes[0] +
                    coords[1] / self.lat.size[1] * self.num_vortexes[1])
        elif type(self.lat) is HexagonalLatticeGidneyOnPlaneWithHole:
            pos = edge_data['pos']
            hole_coords = (self.lat.size[0]//2, self.lat.size[1]//2, 0)
            hole_pos = self.lat.plaquettes[hole_coords].pos
            # measure angle and distance of pos with respect to hole_pos
            angle = np.arctan2(pos[1] - hole_pos[1], pos[0] - hole_pos[0])
            distance = np.linalg.norm(np.array(pos) - np.array(hole_pos))
            return (angle / (2 * np.pi) * self.num_vortexes[0] +
                    distance / np.linalg.norm(self.lat.lattice_vectors[0]) / (self.lat.size[0] / 2) * self.num_vortexes[1])
        else:
            raise ValueError(f'Cant add vortex to lattice type {type(self.lat).__name__}')

    def get_bond_order(self, edge_data, pauli_label):
        color = edge_data['color']
        order_without_vortex = (-color / 3 + ('Z' in pauli_label) / 2) % 1
        order = order_without_vortex + self.location_dependent_delay(edge_data)
        order = order % 1
        return order

    def get_circuit(self, reps=12, reps_without_noise=4, noise_model=None,
                    logical_operator_pauli_type='X', logical_op_directions=('x', 'y'),
                    detector_indexes=None, detector_args=None, draw=True, return_num_logical_qubits=False):
        circ = stim.Circuit()
        for site, data in self.lat.G.nodes.items():
            circ.append_operation("QUBIT_COORDS", self.lat.site_to_index.get(site, []), data['pos'])
        for bond in self.bonds:
            bond.measurement_indexes = []

        logical_operators = dict()
        for i_logical, logical_operator_direction in enumerate(logical_op_directions):
            logical_pauli, sites_on_logical_path = self.get_logical_operator(logical_operator_direction,
                                                                             logical_operator_pauli_type, draw=draw)
            logical_operators[logical_operator_direction] = {'logical_pauli': logical_pauli,
                                                             'sites_on_logical_path': sites_on_logical_path,
                                                             'measurements_to_include': set(),
                                                             'index': i_logical}

        if logical_operator_pauli_type == 'X':
            circ.append_operation("H", list(self.lat.site_to_index.values()))

        i_meas = 0

        for rep in range(reps):
            layer, i_meas = self.get_measurements_layer(i_meas, logical_operator_pauli_type, logical_operators)
            if noise_model is not None and reps_without_noise <= rep < reps - reps_without_noise:
                layer = noise_model.noisy_circuit(layer)
            circ += layer

        # add detectors
        if detector_indexes is not None and detector_args is not None:
            for i, indexes in enumerate(detector_indexes):
                circ.append_operation("DETECTOR", list(map(stim.target_rec, indexes)), detector_args[i])
        else:
            detector_indexes = []
            detector_args = []
            for plaq in self.plaquettes:
                if plaq.pauli_label not in self.detectors:
                    continue
                plaq_measurement_idx_and_sizes = plaq.measurement_indexes_and_sizes()
                plaq_pos = plaq.pos
                rep = 0
                for i in range(len(plaq_measurement_idx_and_sizes)):
                    cur_meas_indexes = []
                    num_sites_in_measurements = 0
                    for j, (meas_idx, meas_size) in enumerate(plaq_measurement_idx_and_sizes[i:]):
                        cur_meas_indexes.append(meas_idx - i_meas)
                        num_sites_in_measurements += meas_size
                        if num_sites_in_measurements >= sum([bond.pauli_length() for bond in plaq.bonds]):
                            break
                    if num_sites_in_measurements > sum([bond.pauli_length() for bond in plaq.bonds]):
                        continue
                    new_circ = circ.copy()
                    current_detector_args = [plaq_pos[0], plaq_pos[1], np.mean(cur_meas_indexes), plaq.pauli_label == 'X']
                    new_circ.append_operation("DETECTOR", list(map(stim.target_rec, cur_meas_indexes)),
                                              current_detector_args)
                    try:
                        new_circ.detector_error_model(decompose_errors=True)
                        circ = new_circ
                        detector_indexes.append(cur_meas_indexes)
                        detector_args.append(current_detector_args)
                        rep += 1
                    except:
                        pass

        # include measurements in the dynamics of the observable
        for direction, logical in logical_operators.items():
            for i_to_include in logical['measurements_to_include']:
                circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(i_to_include - i_meas),
                                      logical['index'])

        # check how many logical qubits are in the code
        num_logical_qubits = get_num_logical_qubits(circ, list(range(self.num_data_qubits)))
        print('num logical qubits: ', num_logical_qubits)

        # Finish circuit with data measurements according to logical operator
        for direction, logical in logical_operators.items():
            logical_pauli = logical['logical_pauli']
            for basis in ['X', 'Y', 'Z']:
                qubits_in_basis = [i for i, p in enumerate(logical_pauli[::-1]) if p.to_label() == basis]
                circ.append_operation("M"+basis, qubits_in_basis)
                circ.append_operation("OBSERVABLE_INCLUDE",
                                      [stim.target_rec(i - len(qubits_in_basis))
                                       for i in range(len(qubits_in_basis))],
                                      logical['index'])
        if return_num_logical_qubits:
            return circ, detector_indexes, detector_args, num_logical_qubits
        else:
            return circ, detector_indexes, detector_args

    def get_measurements_layer(self, i_meas, logical_operator_pauli_type, logical_operators):
        circ = stim.Circuit()
        for bond in self.bonds:
            qubits = [self.lat.site_to_index.get(site, self.boundary_ancilla) for site in bond.sites]
            tp = [[stim.target_x, stim.target_y, stim.target_z, stim.target_z]["XYZI".index(p)] for p in bond.pauli_label]
            if len(bond.pauli_label.replace('I','')) == 2:
                circ.append_operation("MPP", [tp[0](qubits[0]), stim.target_combiner(), tp[1](qubits[1])])
            else:
                # circ.append_operation("M"+bond.pauli_label[0], [qubits[0]])
                # reset an ancilla at index self.num_data_qubits()+1 then do a parity measurement
                # circ.append_operation("R", [self.boundary_ancilla])
                circ.append_operation("M"+bond.pauli_label[0], [qubits[0]])
            bond.measurement_indexes.append(i_meas)

            # if the measured bond is in sites_on_logical_path, and of the same pauli type as the logical operator, include it in the logical operator
            for direction, logical in logical_operators.items():
                if (self.bond_in_path(bond, logical['sites_on_logical_path']) and
                        bond.pauli_label[0] == logical_operator_pauli_type):
                    # self.draw_pauli(logical['logical_pauli'])
                    logical['measurements_to_include'].add(i_meas)
                    logical['logical_pauli'] = (
                        logical['logical_pauli'].compose(self.bond_to_full_pauli(bond)))
            i_meas += 1
        return circ, i_meas

    def bond_in_path(self, bond, sites_on_path):
        return all([site in sites_on_path for site in bond.sites])

    def get_logical_operator(self, logical_operator_direction, logical_operator_pauli_type, draw=False):
        sites_on_logical_path = self.lat.get_sites_on_logical_path(logical_operator_direction)
        logical_operator_string = []
        for site in sites_on_logical_path:
            bonds_connected_to_site = sorted(self.get_bonds_with_site(site), key=lambda b: b.order)
            append_logical = 0  # 0: I, 1: logical_operator_pauli_type
            for bond in bonds_connected_to_site:
                if self.bond_in_path(bond, sites_on_logical_path):
                    append_logical += 1
                if bond.pauli_label[0] != logical_operator_pauli_type:
                    break
            logical_operator_string.append(logical_operator_pauli_type if append_logical % 2 else 'I')
        logical_operator_string = ''.join(logical_operator_string)
        logical_pauli = self.pauli_string_on_sites_to_Pauli(logical_operator_string, sites_on_logical_path)
        if draw:
            self.draw_pauli(logical_pauli)
        return logical_pauli, sites_on_logical_path

    def get_bonds_with_site(self, site):
        bonds = []
        for edge in self.lat.G.edges(site):
            edge_data = self.lat.G.edges[edge]
            bonds.extend(edge_data['bonds'])
        return bonds

    def pauli_string_on_sites_to_Pauli(self, pauli_string, sites):
        full_logical_operator_string = ['I'] * self.num_data_qubits
        for site, index in self.lat.site_to_index.items():
            if site in sites:
                full_logical_operator_string[index] = pauli_string[sites.index(site)]
        full_logical_operator_string = ''.join(full_logical_operator_string)
        logical_pauli = Pauli(full_logical_operator_string)
        return logical_pauli

    def bond_to_full_pauli(self, bond):
        return self.pauli_string_on_sites_to_Pauli(bond.pauli_label, bond.sites)

    def draw_pauli(self, pauli: Pauli):
        self.lat.draw()
        ax = plt.gca()
        for bond in self.bonds:
            # if the two sites are far apart, the bond is an edge bond should be plotted as if site2 is the shifted site
            sites = copy(bond.sites)
            # sites = self.geometry.sites_unwrap_periodic(sites)
            xs, ys = zip(*[self.lat.G.nodes[site]['pos'] for site in sites])
            ax.plot(xs, ys, 'k')
            x = np.mean(xs)
            y = np.mean(ys)
            fontsize = 10
            y = y + (bond.pauli_label == 'XX') * 0.2 - (bond.pauli_label == 'ZZ') * 0.2
            ax.text(x, y, '{:.1f}'.format(bond.order * 6) + bond.pauli_label, fontsize=fontsize, ha='center',
                    va='center')
        if pauli is not None:
            for i, pp in enumerate(pauli[::-1]):
                p = pp.to_label()
                site = self.lat.index_to_site[i]
                x, y = self.lat.G.nodes[site]['pos']
                if p == 'I':
                    ax.plot(x, y, 'ko')
                else:
                    ax.plot(x, y, 'co', markersize=20)
                    ax.text(x, y, p, fontsize=20, ha='center', va='center')
        ax.set_aspect('equal')
        plt.show()


def simulate_vs_noise_rate(phys_err_rate_list, shots, reps_without_noise, noise_type, logical_operator_pauli_type,
                           logical_op_directions, num_vortexes, lat: Lattice, get_reps_by_graph_dist=False,
                           detectors=('X','Z'), draw=False, **kwargs):
    rows = []
    detector_indexes = None
    detector_args = None
    code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=detectors)

    if get_reps_by_graph_dist:
        circ, _, _ = code.get_circuit(reps=1+2*reps_without_noise, reps_without_noise=reps_without_noise,
            noise_model = get_noise_model(noise_type, 0.1),
            logical_operator_pauli_type=logical_operator_pauli_type,
            logical_op_directions=logical_op_directions,
            detector_indexes=detector_indexes, detector_args=detector_args, draw=draw, **kwargs)
        graph_dist = len(circ.shortest_graphlike_error())
        reps = 3 * graph_dist + 2 * reps_without_noise  # 3 cycles - init, idle, meas
        print('graph_dist: ', graph_dist)
    else:
        reps = 3 * min(lat.size) + 2 * reps_without_noise  # 3 cycles - init, idle, meas

    print(f'Simulating: dx={lat.size[0]}, dy={lat.size[1]}, reps={reps}, reps_without_noise={reps_without_noise}, \n'
          f'noise_type={noise_type}, logical_operator_pauli_type={logical_operator_pauli_type}, \n'
          f'num_vortexes={num_vortexes}, shots={shots}')
    for ierr_rate, phys_err_rate in enumerate(phys_err_rate_list):
        noise_model = get_noise_model(noise_type, phys_err_rate)
        circ, detector_indexes, detector_args = code.get_circuit(
            reps=reps, reps_without_noise=reps_without_noise,
            noise_model=noise_model,
            logical_operator_pauli_type=logical_operator_pauli_type,
            logical_op_directions=logical_op_directions,
            detector_indexes=detector_indexes, detector_args=detector_args, draw=False, **kwargs)

        log_err_rate = circ_to_logical_error_rate(circ, shots)
        # print(circ)

        for i_direction, direction in enumerate(logical_op_directions):
            rows.append({
                'dx': lat.size[0],
                'dy': lat.size[1],
                'reps_with_noise': reps - 2 * reps_without_noise,
                'reps_without_noise': reps_without_noise,
                'phys_err_rate': phys_err_rate,
                'log_err_rate': log_err_rate[i_direction],
                'logical_operator_pauli_type': logical_operator_pauli_type,
                'logical_operator_direction': direction,
                'num_vortexes': num_vortexes,
                'noise_type': noise_type,
                'shots': shots,
                'geometry': type(lat).__name__,
                'detectors': detectors,
            })
    df = pd.DataFrame(rows)
    # append to the csv file if exists, otherwise create a new file
    df.to_csv('data/threshold.csv', mode='a', header=not os.path.exists('data/threshold.csv'))


def circ_to_logical_error_rate(circ, shots):
    model = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circ.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    predicted_observables = matching.decode_batch(syndrome)
    num_errors = np.sum(predicted_observables != actual_observables, axis=0)
    log_err_rate = num_errors / shots
    print("logical error_rate", log_err_rate)
    return log_err_rate


