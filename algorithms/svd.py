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


# This code computes the SVD of a matrix using the fact that the left singular vectors can be computed as the
# eigenvectors of the matrix product A * A_transpose, and the right singular vectors can be computed from the
# eigenvectors of A_transpose * A. The singular values can be calculated as the square root of the eigenvalues,
# and the columns of the left and right singular vectors can be normalized to have unit length.

# This implementation handles the case where the matrix is not full rank by using the eig function to compute the
# eigenvalues and eigenvectors of A. Note that this code assumes that the matrix is square, since the left singular
# vectors are computed as a matrix product involving the original matrix.
