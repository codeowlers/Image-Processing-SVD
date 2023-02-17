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

import numpy as np

def svd(A):
    #Compute the Singular Value Decomposition (SVD) of a matrix A.
    #Returns the matrices U, S, and V such that A = U*S*V.T.
    #Uses Eigenvalue Decomposition (EVD) method to compute SVD.
    
    # Compute eigenvalues and eigenvectors of A*A.T
    eigenvals, U = np.linalg.eigh(np.dot(A, A.T))
    
    # Sort eigenvalues in decreasing order and corresponding eigenvectors
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]


    # the matrix U is the matrix of left singular vectors of A.
    # It is an orthogonal matrix of size M x M, where M is the number of rows of A. 
    # The columns of U are the eigenvectors of A*A.T, and they are arranged in decreasing order 
    # of corresponding singular values in the diagonal matrix S.
    U = U[:, idx]
    
    # Compute singular values and matrix of right singular vectors
    
    # The matrix of singular values, S, is computed from the eigenvalues of the matrix AA.T or A.TA. 
    # Specifically, the singular values are the square root of
    # the non-zero eigenvalues of AA.T or A.TA, sorted in descending order.
    S = np.sqrt(eigenvals)


    # the matrix V is the matrix of right singular vectors of A.
    # It is an orthogonal matrix of size N x N, where N is the number of columns of A. 
    # The columns of V are the eigenvectors of A.T*A, and they are arranged in decreasing order
    # of corresponding singular values in the diagonal matrix S.
    V = np.dot(np.dot(A.T, U), np.diag(1.0 / S))
    
    return U, S, V.T


