import numpy as np
from generate_all_lattices import generate_unique_lattices

def get_num_qubits_honeycomb(m: tuple[int, int], k: tuple[int, int]):
    return 6 * abs(m[0] * k[1] - m[1] * k[0])

def get_distance_vector_triangular_lattice(m: tuple[int, int]):
    return max(abs(m[0]), abs(m[1]), abs(m[0] + m[1]))

def check_valid_torus(m: tuple[int, int], k: tuple[int, int]):
    # make sure vectors are not parallel
    return m[0] * k[1] != m[1] * k[0]

def get_distance_torus(m: tuple[int, int], k: tuple[int, int]):
    # minimum distance for ways to wrap around the torus
    return min(get_distance_vector_triangular_lattice(m), get_distance_vector_triangular_lattice(k),
               get_distance_vector_triangular_lattice((m[0] + k[0], m[1] + k[1])),
               get_distance_vector_triangular_lattice((m[0] - k[0], m[1] - k[1])))

if __name__ == '__main__':
    # exhaustive search for the minimum number of qubits for each distance
    best_num_qubits_for_dist = dict()
    best_vectors_for_dist = dict()
    for M in generate_unique_lattices(30):
        m = M[0]
        k = M[1]
        if not check_valid_torus(m, k):
            continue
        dist = get_distance_torus(m, k)
        num_qubits = get_num_qubits_honeycomb(m, k)
        if dist not in best_num_qubits_for_dist or num_qubits < best_num_qubits_for_dist[dist]:
            best_num_qubits_for_dist[dist] = num_qubits
            best_vectors_for_dist[dist] = (m, k)

    # print the results
    for dist in sorted(best_num_qubits_for_dist.keys()):
        print(f'distance: {dist}, num_qubits: 6*{best_num_qubits_for_dist[dist]/6}, C: {best_num_qubits_for_dist[dist]/dist**2}, vectors: {best_vectors_for_dist[dist]}')