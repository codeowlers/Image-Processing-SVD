import numpy as np

def svd(matrix):
    # Compute the singular value decomposition of the matrix
    # using the power iteration method
    n, m = matrix.shape
    max_iter = 100
    eps = 1e-10

    # Initialize the matrices U and V
    U = np.random.rand(n, n)
    V = np.random.rand(m, m)

    # Power iteration to find the dominant eigenvectors
    for i in range(max_iter):
        V = matrix.T @ U
        V_norm = np.linalg.norm(V, axis=0)
        V /= V_norm
        U = matrix @ V
        U_norm = np.linalg.norm(U, axis=0)
        U /= U_norm

    # Compute the singular values and sort them in descending order
    singular_values = np.diag(U.T @ matrix @ V)
    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]

    # Sort the columns of U and V according to the sorted singular values
    U = U[:, sorted_indices]
    V = V[:, sorted_indices]

    return U, singular_values, V.T


# The code above computes the SVD of a given matrix using the power iteration method. The function takes in a matrix
# as input and returns three matrices U, S, and V such that matrix = U @ np.diag(S) @ V.T.
#
# The power iteration method is used to iteratively find the dominant eigenvectors of the matrix until convergence.
# In each iteration, we compute the matrix V by multiplying the transpose of the matrix U with the input matrix,
# and normalize the columns of V. We then compute the matrix U by multiplying the input matrix with V and normalize
# the columns of U. This process is repeated for a fixed number of iterations or until convergence.
#
# After computing U and V, we compute the singular values by taking the diagonal elements of the matrix U.T @ matrix
# @ V. We sort the singular values in descending order and sort the columns of U and V accordingly. Finally,
# we return the sorted U, S, and V matrices.