
"""
algebra.py contains the  algebraic functions the program needs to its correct operation
"""

import numpy as np

def GaussJordan(M):
    """
    Transform the matrix M in an upper triangular matrix using the Gauss-Jordan algorithm.

    Parameters
    ----------
        M:
            Matrix to which the algorithm is applied.
    """
    
    nrows, ncolumns = M.shape
    assert nrows <= ncolumns, "Kirchhoff matrix dimensions are incorrect."
    M = M.copy()
    order = np.arange(ncolumns)
    for i in range(nrows):
        k = np.argmax(np.abs(M[i, i:]))
        if k != 0:
            Maux = M.copy()
            M[:, i], M[:, i + k] = Maux[:, i + k], Maux[:, i]
            order[i], order[i + k] = order[i + k], order[i]
        for j in range(i + 1, nrows):
            M[j, :] -= M[i, :] * M[j, i] / M[i, i]

    return M, order


def reverseGaussJordan(M):
    """
    Transform an upper triangular matrix, M, into a diagonal matrix using the Gauss-Jordan algorithm.

    Parameters
    ----------
        M:
            Upper triangular matrix to which the algorithm is applied.
    """

    if False:
        factor = 1 / np.diag(M)
        M = factor[:, np.newaxis] * M
    else:
        M = np.diag(1.0 / np.diag(M)) @ M

    for i, row in reversed(list(enumerate(M))):
        for j in range(i):
            M[j, :] -= M[j, i] * row

    return M


def remove_zero_rows(M, tol=1e-16):
    """
    Removes all-zero rows from a matrix M.

    Parameters
    ----------
        M:
            Matrix to which the algorithm is applied.
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-16.
    """

    row_norm_1 = np.sum(np.abs(M), -1)
    M = M[(row_norm_1 > tol), :]
    return M


def first_nonzero_index(v, tol=1e-14):
    """
    It returns the index of the firs non-zero element in the vector v
    Retorna el índice del primer elemento no nulo en el vector v, con una tolerancia.

    Parameters
    ----------
        v:
            Vector from which we want the first non-zero index.
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-16.
    """
    for i, val in enumerate(v):
        if abs(val) > tol:  
            return i
    return None  # If there are no non-zero elements, it returns None


Matrix = np.ndarray

def symplectic_form(A: Matrix, tol: float = 1e-16) -> tuple[Matrix, Matrix]:
    """
    Produce the symplectic form for an antisymmetric matrix.
    
    Parameters
    ----------
        A:
            Antisymmetric matrix to which the algorithm is applied.
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-16.
    """

    assert np.allclose(A, - A.T) == True, "The input matrix must be antisymmetric"

    s, U = np.linalg.eig(A)

    """For the antisymmetric matrix, the eigenvalues are either zero or
    purely imaginary.

    For each complex eigenvalue ib with vector x + iy we have another
    one -ib, with vector x - iy. Since the columns of 'U' are the
    eigenvectors, this allows us to easily identify all pairs and sort
    them.

    If we work with 'x' and 'y', then we will have

       A * x = A * 0.5 * (x - iy + x + iy)
             = 0.5 * [(-ib) (x - iy) + ib (x + iy)]
             = b * (-y)
       A * y = A * 0.5 * (-i) (x + iy - (x - iy))
             = -0.5i * [(ib) (x - iy) + ib (x + iy)]
             = b * x
    """
    K = np.abs(U.T @ U)

    found = 0 * U[:, 0]
    pairs = []
    ordered_pairs = []
    zeros = []
    for i, overlaps in enumerate(K):
        if found[i] == 0:
            j = np.argmax(overlaps)
            found[i] = 1
            if i == j:
                zeros.append(U[:, j].real)
            elif s[i].imag < tol:
                zeros.append(U[:, i].real)
            else:
                found[j] = 1
                b = s[i].imag
                if b > 0:
                    x, y = U[:, i].real, U[:, i].imag   
                else:
                    x, y = U[:, j].real, U[:, j].imag
                    b = -b
                pairs.append((b, x / np.sqrt(b * 0.5), y / np.sqrt(b * 0.5)))

                if first_nonzero_index(x) < first_nonzero_index(y):
                    ordered_pairs.append((b, x / np.sqrt(b * 0.5), y / np.sqrt(b * 0.5)))
                elif first_nonzero_index(y) < first_nonzero_index(x):
                    ordered_pairs.append((b, y / np.sqrt(b * 0.5), x / np.sqrt(b * 0.5)))

    assert len(pairs) == len(ordered_pairs), "There is an error in the construction of the 'ordered_pairs' array"

    # Construct the base change matrix, V
    V = np.array([upper_v for b, upper_v, _ in ordered_pairs] + [lower_v for b, _, lower_v in ordered_pairs] + zeros).T

    # Construct the symplectic matrix, J, for the matrix A.
    dimension = A.shape[0]
    number_of_pairs = len(pairs)
    J_abs = np.zeros((dimension, dimension))
    I = np.eye(number_of_pairs)
    
    J_abs[:number_of_pairs, number_of_pairs:number_of_pairs*2] = I
    J_abs[number_of_pairs:number_of_pairs*2, :number_of_pairs] = I

    J = J_abs * np.sign(V.T @ A @ V)

    assert np.allclose(J, (V.T @ A @ V)), "There is an error in the construction of the symplectic matrix"

    return J, V, number_of_pairs
    

