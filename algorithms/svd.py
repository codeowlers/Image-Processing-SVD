import numpy as np


# In linear algebra, the singular value decomposition (SVD) is a matrix factorization that decomposes a matrix into
# three matrices. If A is an m x n matrix, then its SVD is given by:
#
# A = UΣV^T
#
# where U is an m x m unitary matrix, Σ is an m x n diagonal matrix with non-negative real numbers on the diagonal (
# the singular values), and V^T is the transpose of an n x n unitary matrix V.
#
# In this factorization, the columns of U are called the left singular vectors, the columns of V are called the right
# singular vectors, and the diagonal entries of Σ are called the singular values.
#
# The matrices U and V have special properties that make them useful for a variety of applications, including
# dimensionality reduction, data compression, and matrix approximation. For example, the first k columns of U and V
# can be used to approximate A with a rank-k matrix, where k is a positive integer less than or equal to the rank of
# A. The singular values themselves provide information about the importance of each singular vector in the
# approximation.
#
# In summary, the matrices U, Σ, and V^T returned from the SVD of a matrix A provide a way to decompose A into its
# constituent parts, and they have many useful applications in linear algebra and beyond.

def svd(matrix):
    """
    Computes the Singular Value Decomposition of a matrix using the power iteration method.

    Parameters:
        matrix (ndarray): The input matrix.

    Returns:
        U (ndarray): The left singular vectors.
        s (ndarray): The singular values.
        Vt (ndarray): The right singular vectors (transposed).
    """
    # Compute the matrix product of A and A_transpose to find the eigenvalues
    A = matrix @ matrix.T
    n = matrix.shape[0]

    # Use the eig function to find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Sort the eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the singular values
    s = np.sqrt(eigenvalues[sorted_indices])

    # Compute the right singular vectors
    Vt = (eigenvectors.T @ matrix) / s

    # Compute the left singular vectors
    U = np.zeros((n, n))
    for i in range(n):
        U[:, i] = matrix @ Vt[:, i]

    return U, s, Vt


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
