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
    u_matrix = np.dot(matrix, np.transpose(matrix))
    v_matrix = np.dot(np.transpose(matrix), matrix)
    if np.size(u_matrix) > np.size(v_matrix):
        s_matrix = np.dot(np.transpose(matrix), matrix)
    else:
        s_matrix = np.dot(matrix, np.transpose(matrix))

    u_eigenvalues, u_eigenvectors = np.linalg.eig(u_matrix)
    v_eigenvalues, v_eigenvectors = np.linalg.eig(v_matrix)
    s_eigenvalues, s_eigenvectors = np.linalg.eig(s_matrix)
    s_eigenvalues = np.sqrt(s_eigenvalues)
    s = s_eigenvalues[:: -1]
    v_ncols = np.argsort(v_eigenvalues)[::-1]
    u_ncols = np.argsort(u_eigenvalues)[::-1]

    v = v_eigenvectors[:, v_ncols].T
    u = u_eigenvectors[:, u_ncols]

    return u, s, v

# transpose = np.transpose(matrix)
# square_matrix = np.dot(matrix, transpose)
# eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
# indices = eigenvalues.argsort()[::-1]
# eigenvalues = eigenvalues[indices]
# eigenvectors = eigenvectors[:, indices]
# diagonal_matrix = np.zeros_like(square_matrix)
# np.fill_diagonal(diagonal_matrix, np.sqrt(eigenvalues))
# U = np.dot(eigenvectors.T, matrix).T
# V = np.dot(np.dot(transpose, eigenvectors), np.linalg.inv(diagonal_matrix))
#
# return U, diagonal_matrix, V

# This code computes the SVD of a matrix using the fact that the left singular vectors can be computed as the
# eigenvectors of the matrix product A * A_transpose, and the right singular vectors can be computed from the
# eigenvectors of A_transpose * A. The singular values can be calculated as the square root of the eigenvalues,
# and the columns of the left and right singular vectors can be normalized to have unit length.

# This implementation handles the case where the matrix is not full rank by using the eig function to compute the
# eigenvalues and eigenvectors of A. Note that this code assumes that the matrix is square, since the left singular
# vectors are computed as a matrix product involving the original matrix.


# Find the transpose of the matrix. Multiply the matrix and its transpose to get a square matrix. Find the
# eigenvalues and eigenvectors of the square matrix. Sort the eigenvalues in decreasing order and arrange the
# corresponding eigenvectors in the same order. Form the diagonal matrix by placing the square root of the
# eigenvalues along the diagonal. Multiply the original matrix with the eigenvectors to get the U matrix. Multiply
# the transpose of the matrix with the eigenvectors and multiply it with the inverse of the diagonal matrix to get
# the V matrix. Return the U, S, and V matrices as the SVD of the original matrix.
