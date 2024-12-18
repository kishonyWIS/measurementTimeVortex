import numpy as np
from lll_reduce import lll_reduce_2d

def generate_unique_lattices(max_det):
    """Enumerate all unique 2D integer lattices up to a given determinant."""
    unique_lattices = set()

    for a1 in range(-max_det, max_det + 1):
        for b1 in range(-max_det, max_det + 1):
            for a2 in range(-max_det, max_det + 1):
                for b2 in range(-max_det, max_det + 1):
                    # Construct basis matrix
                    B = np.array([[a1, a2], [b1, b2]]).astype(int)

                    # Compute determinant and check if it's non-zero
                    det = int(np.linalg.det(B))
                    if abs(det) > 0 and abs(det) <= max_det:
                        # Reduce the basis to canonical form
                        reduced_B = tuple(map(tuple, lll_reduce_2d(B)))

                        # Add the reduced basis to the set
                        unique_lattices.add(reduced_B)

    return list(unique_lattices)

if __name__ == '__main__':
    # Example usage
    unique_lattices = generate_unique_lattices(max_det=10)
    print(f"Number of unique lattices: {len(unique_lattices)}")
    for lattice in unique_lattices:
        print(np.array(lattice))
