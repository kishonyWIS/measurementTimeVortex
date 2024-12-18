import numpy as np


def lll_reduce_2d(B):
    """
    Perform LLL reduction on a 2D basis matrix B.
    Input:
        B: numpy array of shape (2, 2), representing the lattice basis.
    Output:
        Reduced basis matrix of the same shape.
    """
    # Ensure numpy array
    B = np.array(B, dtype=int)
    if B.shape[1] > 2:
        B[:,2:] = 1/6*B[:,2:]

    while True:
        v1, v2 = B[0], B[1]

        # Compute Gram-Schmidt coefficient
        mu = round(np.dot(v1, v2) / np.dot(v1, v1))

        # Update v2
        v2 = v2 - mu * v1

        # Swap if needed
        if np.dot(v2, v2) < np.dot(v1, v1):
            B = np.array([v2, v1])  # Swap rows
        else:
            B[1] = v2
            break

    # Fix signs of vectors for consistency
    for i in range(B.shape[0]):
        if B[i][0] < 0 or (B[i][0] == 0 and B[i][1] < 0):
            B[i] *= -1

    if B.shape[1] > 2:
        B[:,2:] = 6*B[:,2:]
    return B


import numpy as np


def gram_schmidt(B):
    """
    Perform Gram-Schmidt orthogonalization on the input basis B.
    Input:
        B: numpy array of shape (n, n), where n is the dimension.
    Output:
        Q: Orthonormalized vectors (columns).
        R: Upper triangular matrix of coefficients.
    """
    n = B.shape[0]
    Q = np.zeros_like(B, dtype=float)
    R = np.zeros((n, n), dtype=float)

    for i in range(n):
        Q[:, i] = B[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], B[:, i]) / np.dot(Q[:, j], Q[:, j])
            Q[:, i] -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R


def lll_reduce(B, delta=0.75):
    """
    Perform the LLL reduction on a lattice basis B.
    Input:
        B: numpy array of shape (n, n), representing the basis vectors as columns.
        delta: Lovász condition parameter (0.5 < delta < 1).
    Output:
        Reduced basis matrix with fixed sign convention.
    """
    B = np.array(B, dtype=float).T  # Columns as basis vectors
    n = B.shape[1]

    def lovasz_condition(k):
        return delta * np.dot(B[:, k - 1], B[:, k - 1]) <= np.dot(B[:, k], B[:, k])

    # Perform Gram-Schmidt and initialize
    Q, R = gram_schmidt(B)
    k = 1

    while k < n:
        # Reduce step: make coefficients of Gram-Schmidt orthogonalization small
        for j in range(k - 1, -1, -1):
            mu = np.dot(Q[:, j], B[:, k]) / np.dot(Q[:, j], Q[:, j])
            if abs(mu) > 0.5:
                B[:, k] -= round(mu) * B[:, j]

        # Update Gram-Schmidt basis
        Q, R = gram_schmidt(B)

        # Lovász condition
        if lovasz_condition(k):
            k += 1
        else:
            # Swap vectors
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            Q, R = gram_schmidt(B)
            k = max(k - 1, 1)

    # Fix signs for consistency
    for i in range(B.shape[1]):
        if B[:, i][0] < 0 or (B[:, i][0] == 0 and np.sum(B[:, i] < 0) % 2 != 0):
            B[:, i] *= -1  # Ensure deterministic sign convention

    return B.T  # Return basis as rows

# Example usage
if __name__ == '__main__':
    B = np.array([[4, 1], [1, 3]])
    reduced_B = lll_reduce_2d(B)
    print("Original Basis:")
    print(B)
    print("LLL Reduced Basis:")
    print(reduced_B)
    print(lll_reduce_2d(-B))
    print(lll_reduce_2d(B[::-1]))
    print(lll_reduce_2d(np.stack((B[0,:], -B[1,:]), axis=0)))


    # use the LLL reduction function
    B = np.array([[4, 1], [1, 3]])
    reduced_B = lll_reduce(B)
    print("Original Basis:")
    print(B)
    print("LLL Reduced Basis:")
    print(reduced_B)
    print(lll_reduce(-B))
    print(lll_reduce(B[::-1]))
    print(lll_reduce(np.stack((B[0,:], -B[1,:]), axis=0)))
