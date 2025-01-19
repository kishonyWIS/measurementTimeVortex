import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value
from lll_reduce import lll_reduce_2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import olll
import pandas as pd
from plot_utils import edit_graph
from collections import deque


def minimize_L1_norm(M, v):
    n, m = M.shape
    # Create problem
    prob = LpProblem("Minimize_L1_Norm", LpMinimize)

    # Define variables
    w = [LpVariable(f"w_{i}", cat="Integer") for i in range(m)]
    u = [LpVariable(f"u_{i}", lowBound=0, cat="Continuous") for i in range(m)]

    # Objective function
    prob += lpSum(u)

    # Constraints for absolute value bounds
    for i in range(m):
        prob += w[i] <= u[i]
        prob += w[i] >= -u[i]

    # Constraints for the equation wM = v
    for i in range(n):
        prob += lpSum(M[i, j] * w[j] for j in range(m)) == v[i]

    # Solve the problem
    prob.solve()

    # Retrieve the solution
    w_solution = np.array([value(w[i]) for i in range(m)])
    return w_solution


# # Example usage
# M = np.array([
#     [1, 0, 0, 1, 0, 1], #x
#     [0, 1, 0, 0, 1, 1], #y
#     [0, 0, 1, 1, -1, -1] #t
# ])
# v = np.array([-7, -1, -3])
#
# w = minimize_L1_norm(M, v)
# print("Solution w:", w)
# print("L1 norm of w:", np.linalg.norm(w, ord=1))



def is_valid_torus(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    return True

def get_area(L1: tuple[int,int,int], L2: tuple[int,int,int], signed=False):
    a1, b1, t1 = L1
    a2, b2, t2 = L2
    area = 2 * (a1*b2 - a2*b1)
    return area if signed else abs(area)

def get_vector_length(v: tuple[int,int,int]):
    M = np.array([
        [1, 0, 0, 1, 0, 1],  # x
        [0, 1, 0, 0, 1, 1],  # y
        [0, 0, 1, 1, -1, -1]  # t
    ])
    w = minimize_L1_norm(M, v)
    return np.linalg.norm(w, ord=1)

# def all_symmetric_lattices(L1: tuple[int,int,int], L2: tuple[int,int,int]):
#     def reflection(L):
#         return (-L[0], L[0]+L[1], L[2])
#     def rotation60degrees_and_time_reversal(L):
#         return (-L[1], L[0]+L[1], -L[2])
#     symmetric_lattices = []
#     for num_reflections in range(2):
#         for num_rotations in range(6):
#             L1_symmetric = L1
#             L2_symmetric = L2
#             for _ in range(num_reflections):
#                 L1_symmetric = reflection(L1_symmetric)
#                 L2_symmetric = reflection(L2_symmetric)
#             for _ in range(num_rotations):
#                 L1_symmetric = rotation60degrees_and_time_reversal(L1_symmetric)
#                 L2_symmetric = rotation60degrees_and_time_reversal(L2_symmetric)
#             symmetric_lattices.append((L1_symmetric, L2_symmetric))
#     return symmetric_lattices
#
# def lattices_are_equiv_or_symmetric(L1first: tuple[int,int,int], L2first: tuple[int,int,int], L1second: tuple[int,int,int], L2second: tuple[int,int,int]):
#     for L1_symmetric, L2_symmetric in all_symmetric_lattices(L1second, L2second):
#         if lattices_are_equiv(L1first, L2first, L1_symmetric, L2_symmetric):
#             return True
#     return False

def lattices_are_equiv(L1first: tuple[int,int,int], L2first: tuple[int,int,int], L1second: tuple[int,int,int], L2second: tuple[int,int,int]):
    # check if L1first, L2first are integer linear combinations of L1second, L2second and vice versa
    # solve the linear system
    A = np.array([L1second, L2second]).T
    B = np.array([L1first, L2first]).T
    X = np.linalg.solve(A[:2,:], B[:2,:])
    # check if the solution is an integer matrix
    is_integer = np.allclose(X, np.round(X))
    # check if the solution is a valid transformation
    is_valid = np.allclose(A @ X, B)
    return is_integer and is_valid

def reduce_equivalent_lattices(lattices):
    reduced_lattices = []
    for L1, L2 in lattices:
        is_new = True
        for L1_, L2_ in reduced_lattices:
            if lattices_are_equiv(L1, L2, L1_, L2_):
                is_new = False
                break
        if is_new:
            reduced_lattices.append((L1, L2))
    return reduced_lattices

def lll_reduce_new(L1: tuple, L2: tuple):
    B = [L1, L2]
    B = np.array(olll.reduction(B,0.75), dtype=int)
    # fix signs
    for i in range(len(B)):
        if B[i][0] < 0 or (B[i][0] == 0 and B[i][1] < 0):
            B[i] *= -1
    return tuple(B)

def lll_reduce(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    B = np.array([L1, L2])
    reduced_B = lll_reduce_2d(B)
    return tuple(reduced_B)

def get_distance_torus(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    d = np.inf
    max_m = 2
    for m1 in range(-max_m,max_m+1):
        for m2 in range(-max_m,max_m+1):
            if m1 == 0 and m2 == 0:
                continue
            v = m1*np.array(L1) + m2*np.array(L2)
            d = min(d, get_vector_length(v))
    return d

def generate_all_spatial_lattices(max_num_qubits):
    unique_spatial_lattices = set()
    A = max_num_qubits/2
    d1 = 0
    for c1 in tqdm(range(1,int(A)+1)):
        for d2 in range(1,int(A/c1)+1):
            for c2 in range(-c1, c1+1):
                l1 = (d1, c1)
                l2 = (d2, c2)
                l1, l2 = lll_reduce(l1, l2)
                l1,l2 = tuple(map(int,l1)), tuple(map(int,l2))
                assert lattices_are_equiv(l1, l2, (d1, c1), (d2, c2))
                unique_spatial_lattices.add((l1,l2))
    return list(unique_spatial_lattices)

def find_integer_points_in_convex_region(is_inside):
    """
    Find all integer points within a convex region using BFS.

    Args:
        is_inside (function): A function that returns True if a point (x, y) is inside the region.

    Returns:
        set: A set of tuples representing all integer points inside the region.
    """
    visited = set()  # To store points we've already checked
    inside_points = set()  # To store points inside the region
    queue = deque([(0, 0)])  # Start from the origin

    while queue:
        x, y = queue.popleft()

        # If we've already visited this point, skip it
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Check if the point is inside the region
        if is_inside(x, y):
            inside_points.add((x, y))
            # Add neighbors to the queue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in visited:
                    queue.append(neighbor)

    return inside_points

def find_optimal_lattice_for_each_distance(max_num_qubits = 1000):
    best_area_for_dist_no_vortex = dict()
    best_torus_for_dist_no_vortex = dict()
    for L1, L2 in tqdm(generate_all_spatial_lattices(max_num_qubits)):
        # filter even number of vortices in each direction
        dist = get_distance_torus(L1, L2)
        area = get_area(L1, L2)
        if L1[-1] == 0 and L2[-1] == 0:
            if dist not in best_area_for_dist_no_vortex or area < best_area_for_dist_no_vortex[dist]:
                best_area_for_dist_no_vortex[dist] = int(area)
                best_torus_for_dist_no_vortex[dist] = (L1, L2)
    # save the data to a csv file using pandas
    rows = []
    for dist in sorted(best_area_for_dist.keys()):
        L1_no_vortex, L2_no_vortex = best_torus_for_dist_no_vortex.get(dist, (None, None))
        num_qubits_no_vortex = best_area_for_dist_no_vortex.get(dist, None)
        rows.append({'distance': dist,
                     'num_qubits_no_vortex': num_qubits_no_vortex,
                     'L1_no_vortex': L1_no_vortex,
                     'L2_no_vortex': L2_no_vortex})
    df = pd.DataFrame(rows)
    # make dtype of num_qubits_no_vortex an integer
    df['num_qubits_no_vortex'] = df['num_qubits_no_vortex'].astype('Int64')
    # save to csv without index, overwrite the file if it exists, create the file if it does not exist
    df.to_csv('../data/best_torus_for_distance_toric_code.csv', index=False)
    # print the results
    for dist in sorted(best_area_for_dist_no_vortex.keys()):
        print(
            f'distance: {dist}, area: {best_area_for_dist_no_vortex[dist]}, A/d^2: {best_area_for_dist_no_vortex[dist] / dist ** 2}, torus: {best_torus_for_dist_no_vortex[dist]}')
    # plot the area/distance squared as a function of distance
    plt.figure()
    x = sorted(best_area_for_dist_no_vortex.keys())
    y = [best_area_for_dist_no_vortex[dist] / dist ** 2 for dist in x]
    plt.plot(x, y)
    edit_graph('$D$', '$N/D^2$', scale=1.5)
    plt.savefig('../figures/figure_of_merit_vs_distance_toric_code.pdf')
    plt.show()

def find_all_optimal_lattices(max_num_qubits = 1000):
    # for each distance, find all tori with the best area
    best_area_for_dist_no_vortex = dict()
    best_torus_for_dist_no_vortex = dict()
    for l1, l2 in tqdm(generate_all_spatial_lattices(max_num_qubits)):
        # filter even number of vortices in each direction
        L1 = (l1[0], l1[1], 0)
        L2 = (l2[0], l2[1], 0)
        dist = get_distance_torus(L1, L2)
        area = get_area(L1, L2)
        if L1[-1] == 0 and L2[-1] == 0:
            if dist not in best_area_for_dist_no_vortex or area < best_area_for_dist_no_vortex[dist]:
                best_area_for_dist_no_vortex[dist] = int(area)
                best_torus_for_dist_no_vortex[dist] = [(L1, L2)]
            elif area == best_area_for_dist_no_vortex[dist]:
                best_torus_for_dist_no_vortex[dist].append((L1, L2))
    # reduce equivalent lattices
    for dist in best_torus_for_dist_no_vortex:
        best_torus_for_dist_no_vortex[dist] = reduce_equivalent_lattices(best_torus_for_dist_no_vortex[dist])

    for with_vortexes in [False]:
        best_torus = best_torus_for_dist if with_vortexes else best_torus_for_dist_no_vortex
        best_area = best_area_for_dist if with_vortexes else best_area_for_dist_no_vortex
        data = []
        for distance in sorted(best_torus.keys()):
            configurations = best_torus[distance]
            num_qubits = best_area.get(distance, None)
            for L1, L2 in configurations:
                data.append({
                    "distance": distance,
                    "# qubits": num_qubits,
                    "L1": L1,
                    "L2": L2
                })

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(f"../data/torus_configurations_{'with' if with_vortexes else 'without'}_vortices_toric_code.csv", index=False)

    return best_torus_for_dist_no_vortex, best_area_for_dist_no_vortex


if __name__ == '__main__':
    # draw_distance_vs_num_vortices(6,0,0,6)

    # lattices = find_all_lattices_with_num_qubits_and_distance(72,5)
    # print(lattices)
    # print(len(lattices))

    # find the torus with the best area for each distance
    # find_optimal_lattice_for_each_distance(1000)

    # find all optimal lattices
    d1 = get_distance_torus((2, -3, 0), (3, 3, 0))
    d2 = get_distance_torus((2, -3, 0), (3, 3, 1))
    d3 = get_distance_torus((2, -3, 0), (3, 3, -1))
    d4 = get_distance_torus((2, -3, 1), (3, 3, 0))
    d5 = get_distance_torus((2, -3, -1), (3, 3, 0))

    best_torus_for_dist_no_vortex, best_area_for_dist_no_vortex = find_all_optimal_lattices(30)