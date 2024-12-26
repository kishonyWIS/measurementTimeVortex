from color_code import FloquetCode
from lattice import HexagonalLattice
from matplotlib import pyplot as plt
import ast
import pandas as pd
import numpy as np
import qiskit

def get_depths(bonds, include_end=False):
    # Initialize variables
    last_act = {}  # To store the current depth of each qubit
    idle_times = {}  # To store the total idle time for each qubit
    all_gate_depths = []  # To store the depth of each gate

    # Iterate through the bonds (gates)
    for gate in bonds:
        # Determine the depth of the current gate
        current_gate_depth = max(last_act.get(qubit, 0) for qubit in gate) + 1
        all_gate_depths.append(current_gate_depth)

        # Update idle times and depths for each qubit in the gate
        for qubit in gate:
            if qubit in last_act:
                idle_time = current_gate_depth - last_act[qubit] - 1
                idle_times[qubit] = idle_times.get(qubit, 0) + max(idle_time, 0)
            else:
                # First activity for this qubit
                idle_times[qubit] = idle_times.get(qubit, 0)

            # Update the depth of the qubit
            last_act[qubit] = current_gate_depth

    # Handle the final idle times for all qubits
    if include_end:
        final_depth = max(last_act.values(), default=0)
        for qubit, last_depth in last_act.items():
            idle_time = final_depth - last_depth
            idle_times[qubit] += max(idle_time, 0)

    # Print the idle times for each qubit
    # print("Idle times for each qubit:")
    # for qubit, idle_time in idle_times.items():
        # print(f"Qubit {qubit}: {idle_time}")

    return all_gate_depths

def draw_gates_layer_by_layer(bonds, gate_depths, code: FloquetCode):
    depth = 0
    fig, ax = plt.subplots()
    prev_bonds = []
    current_bonds = []
    bonds, gate_depths = zip(*sorted(zip(bonds, gate_depths), key=lambda x: x[1]))  # Sort by gate depth
    for i, bond in enumerate(bonds):
        if gate_depths[i] > depth:
            depth = gate_depths[i]
            code.draw_pauli(None, ax=ax, show=False)
            for b in prev_bonds:
                sites_unwrapped = code.lat.unwrap_periodic(b)
                xs, ys = zip(*[code.lat.coords_to_pos(site) for site in sites_unwrapped])
                plt.plot(xs, ys, color='cyan', linewidth=10)
            for b in current_bonds:
                sites_unwrapped = code.lat.unwrap_periodic(b)
                xs, ys = zip(*[code.lat.coords_to_pos(site) for site in sites_unwrapped])
                plt.plot(xs, ys, color='orange', linewidth=10)
            plt.show()
            prev_bonds = current_bonds
            current_bonds = []
        # positions = [code.lat.coords_to_pos(site) for site in bond]
        # plt.plot([p[0] for p in positions], [p[1] for p in positions], [gate_depths[i], gate_depths[i]], color='cyan', linewidth=10)
        current_bonds.append(bond)


def brick_wall_circuit_on_circle(num_qubits, n_vortices):
    # generate a list of bonds
    # sites are on 1d circle, enumerate them from 0 to num_qubits-1
    # bonds are between nearest neighbors
    bonds_to_delay = {(i, (i + 1) % num_qubits): (i%2*0.5 + i/num_qubits*n_vortices) % 1 for i in range(num_qubits)}
    # sort bonds by delay
    bonds = sorted(bonds_to_delay.keys(), key=lambda x: bonds_to_delay[x])
    return bonds

def draw_bonds_as_qiskit(bonds):
    qc = qiskit.QuantumCircuit(max(max(bond) for bond in bonds)+1)
    depths = get_depths(bonds, include_end=False)
    cur_depth = 0
    for bond, depth in zip(bonds, depths):
        if depth > cur_depth:
            qc.barrier()
            cur_depth = depth
        qc.cz(*bond)
    qc.draw(output='mpl')
    plt.title('depth: ' + str(max(depths)))
    plt.show()

if __name__ == '__main__':

    num_qubits = 12
    n_vortices = 2
    bonds = brick_wall_circuit_on_circle(num_qubits, n_vortices)
    draw_bonds_as_qiskit(bonds*3)

    # def tuple_int_converter(value):
    #     if not value or pd.isna(value):  # Check for empty or NaN values
    #         return ()  # Return an empty tuple or another default value
    #     try:
    #         return tuple(map(int, ast.literal_eval(value)))
    #     except (ValueError, SyntaxError):
    #         # Handle cases where the value is not properly formatted
    #         raise ValueError(f"Invalid tuple format: {value}")
    #
    #
    # df = pd.read_csv('data/best_torus_for_distance.csv',
    #                  converters={'L1': tuple_int_converter, 'L2': tuple_int_converter,
    #                              'L1_no_vortex': tuple_int_converter, 'L2_no_vortex': tuple_int_converter})
    #
    # unique_L1_L2_dist = set()
    # for irow, row in df.iterrows():
    #     for L1, L2, dist in [(row['L1'], row['L2'], row['distance']),
    #                          (row['L1_no_vortex'], row['L2_no_vortex'], row['distance'])]:
    #         if len(L1) == 0 or len(L2) == 0:
    #             continue
    #         unique_L1_L2_dist.add((L1, L2, dist))
    #
    # # sort by abs(L1 x L2)
    # unique_L1_L2 = sorted(unique_L1_L2_dist, key=lambda x: abs(x[0][0] * x[1][1] - x[0][1] * x[1][0]))
    # for L1, L2, dist in unique_L1_L2:
    #     num_vortexes = (int(np.round(-L1[-1] / 6)), int(np.round(-L2[-1] / 6)))
    #     print(f'L1:{L1}, L2:{L2}, num_vortices:{num_vortexes}, distance:{dist}')
    #     lat = HexagonalLattice(L1[:-1], L2[:-1])
    #     reps_without_noise = 1
    #     reps_with_noise = 0
    #     code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=('X',))
    #
    #     bonds = [b.sites for b in code.bonds]*10
    #     depths = get_depths(bonds, include_end=False)
    #     print(max(depths))
    #     print()
    # (2, 5, -12)	(8, -4, -18)
    lat = HexagonalLattice((3,0), (1,-5))
    reps_without_noise = 1
    reps_with_noise = 0
    code = FloquetCode(lat, num_vortexes=(1,0), detectors=('X',))

    bonds = [b.sites for b in code.bonds]*10
    #filter only bonds with first coordinate of both qubits equal to 0
    depths = get_depths(bonds, include_end=False)
    # count how many times each depth appears
    depth_counts = {depth: depths.count(depth) for depth in set(depths)}
    print("Depth counts:", depth_counts)
    print('Max depth:', max(depths))
    draw_gates_layer_by_layer(bonds, depths, code)