
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

    Returns
    ----------
        M:  
            Upper triangular form of the input matrix once the algorithm has been applied.
        order:  
            Variable order of the new upper diagonal matrix.
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

    Returns
    ----------
        M:  
            Diagonal form of the input matrix once the algorithm has been applied.
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
            Tolerance below which a element is considered zero. By default, it is 1e-16.

    Returns
    ----------
        M:  
            Input matrix with no zero rows.
    """

    row_norm_1 = np.sum(np.abs(M), -1)
    M = M[(row_norm_1 > tol), :]
    return M

def pseudo_inv(M, tol=1e-15):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and considering a total
    tolerance below which each singular value is considered zero.

    Parameters
    ----------
        M:
            Input matrix 
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-15.

    Returns
    ----------
        pseudo_inv:  
            Moore-Penrose pseudo-inverse matrix of input matrix.
    """
    # SVD decomposition
    U, S, Vt = np.linalg.svd(M)
    
    # Invert the singular values taking into account the tolerance
    S_inv = np.zeros((Vt.shape[0], U.shape[1]))  # Preallocate S_inv matrix with correct dimensions
    for i in range(len(S)):
        if S[i] > tol:  # Invert only if the singular value is bigger than the tolerance
            S_inv[i, i] = 1 / S[i]
    
    # Get the pseudo-inverse
    pseudo_inv = Vt.T @ S_inv @ U.T
    return pseudo_inv

def nonzero_indexes(M, tol=1e-14):
    """
    It returns the row indexes of the non-zero elements in the matrix M, with a tolerance.
    Parameters
    ----------
        M:
            Matrix from which we want the non-zero row indexes.
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    Returns
    ----------
        indexes:  
            List of row indexes of the non-zero elements in the input matrix M.
    """
    indexes = []
    for j in range(M.shape[1]):
        for i in range(M.shape[0]):
            if abs(M[i,j]) > tol:
                indexes.append(i)

    indexes = sorted(set(indexes))
    
    return indexes


Matrix = np.ndarray
def symplectic_transformation(M: Matrix, no_flux_variables: int, Omega: bool = False, tol: float = 1e-16) -> tuple[Matrix, Matrix]:
    """
    Transform a square matrix M = JH (with H a positive semidefinite matrix and J the Symplectic matrix) to eigval*J = [[0,eigval*1],[-eigval*1,0]]
    such that eigval*J = inv(T) @ M @ T. If Omega = True, instead transform M antisymmetric matrix to J such that J = T.T @ M @ T.

    Parameters
    ----------
        M:
            Square matrix to which the algorithm is applied.
        no_flux_variables:
            Parameter to indicate how many flux variables we have.
        Omega:
            Parameter to indicate whether we transform M to eigval*J (False) or J (True). By default it is False
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-16.
    
    Returns
    ----------
        M_out:  
            Output matrix. If Omega = False: M_out = eigval*J. If Omega = True: M_out = J
        T or V:
            Basis change matrix that transforms the input matrix M into eigval*J (T, Omega = False) or J (V, Omega = True).
    """

    # Verify that the input matrix is square, with an even dimenstion if Omega == False and antisymmetric if Omega = True and 
    assert M.shape[0] == M.shape[1], "The input matrix must be square"
    if Omega == False:
        assert M.shape[0]%2 == 0, "For the case Omega == False, the input matrix must be even"
    else:
        assert np.allclose(M.T, -M, rtol = tol), "If Omega = True, input matrix must be antisymmetric"

    # Obtain the eigenvalues and eigenvectors of the input matrix and sort them 
    M_eigval, M_eigvec = np.linalg.eig(M)

    index = np.argsort(M_eigval.imag)
    M_eigval = M_eigval[index]
    M_eigvec = M_eigvec[:, index]

    # Verify the input matrix does not have degenerate eigenvalues with geometric multiplicity < algebraic multiplicity
    assert np.linalg.matrix_rank(M_eigvec, tol) == M.shape[0], "There are degenerate eigenvalues with geometric \
        multiplicity < algebraic multiplicity -> I fail my assumption and the program is not ready."

    # Organize the eigenvalues with their eigenvectors in two groups: zero and pure imaginary eigenvalues
    zero_eigval, zero_eigvec = np.empty(0), np.empty((M.shape[1], 0))
    imag_eigval, imag_eigvec = np.empty(0), np.empty((M.shape[1], 0))

    for i, eigval in enumerate(M_eigval):

        if np.allclose(eigval.real, 0) and np.allclose(eigval.imag, 0):
            zero_eigval = np.hstack((zero_eigval, 0)) 
            zero_eigvec = np.hstack((zero_eigvec, M_eigvec[:,i].reshape(-1,1)))

        elif np.allclose(eigval.real, 0) and eigval.imag > 0:
            imag_eigval = np.hstack((imag_eigval, 1j * eigval.imag)) # Positive purely imaginary eigenvalue
            imag_eigvec = np.hstack((imag_eigvec, M_eigvec[:,i].reshape(-1,1)))

    # Verify the input matrix has the correct eigenvalues
    assert 2 * len(imag_eigval) + len(zero_eigval) == len(M_eigval), \
        "The input matrix must have only zero or pure imaginary eigenvalues by conjugate pairs"
    
    # Define the symplectic matrix J with the correct dimensions
    J = np.zeros((2 * len(imag_eigval) + len(zero_eigval), 2 * len(imag_eigval) + len(zero_eigval)))
    I = np.eye(len(imag_eigval))
    
    J[:len(imag_eigval), len(imag_eigval):2 * len(imag_eigval)] = I
    J[len(imag_eigval):2 * len(imag_eigval), :len(imag_eigval)] = -I 

    
    # If Omega == False:
    # - Normalize the pure imaginary eigenvectors under the standard symplectic inner product x.T @ J @ y 
    # - Construct the basis change matrix T such that eigval*J = inv(T) @ M @ T
    if Omega == False:
    
        # Eigenvectors normalization under the symplectic inner product
        normal_imag_eigvec = np.empty((M.shape[0], 0))

        for i, eigval in enumerate(imag_eigval):

            # Repeated eigenvalues
            if i > 0 and np.allclose(imag_eigval[i-1], imag_eigval[i]):
                j += 1 
                summary = 0
                for m in range(1,j+1):
                    Phi_star = np.conj(normal_imag_eigvec[:,i-m].T @ J @ np.conj(imag_eigvec[:,i]))
                    summary += Phi_star * normal_imag_eigvec[:,i-m].reshape(-1,1) 

                eigvec = (imag_eigvec[:,i].reshape(-1,1) - sigma * summary)
                norm = np.abs(np.sqrt(eigvec.T @ J @ np.conj(eigvec)))
                normal_imag_eigvec = np.hstack((normal_imag_eigvec, eigvec/norm)) 
                continue
            j = 0

            # First eigenvalues
            alpha = imag_eigvec[:,i].T @ J @ np.conj(imag_eigvec[:,i])
            sigma = 1j * np.sign(alpha/1j)
            Phi = np.sqrt(sigma * alpha)
            normal_imag_eigvec = np.hstack((normal_imag_eigvec, (imag_eigvec[:,i].reshape(-1,1))/Phi)) 

            # Verify the orthonormalization of the term i
            assert np.allclose(normal_imag_eigvec[:,i].T @ J @ np.conj(normal_imag_eigvec[:,i]), 1j, rtol = tol) \
                or np.allclose(normal_imag_eigvec[:,i].T @ J @ np.conj(normal_imag_eigvec[:,i]), -1j, rtol = tol), \
                "There is an error in the orthonormalization of an eigenvector from a purely imaginary eigenvalue"
            
        # Add an aditional phase to the imaginary eigenvectors, if it is necessary, to to obtain a block diagonal V matrix if it is possible
        for i in range(len(imag_eigval)):
            if np.allclose(sum(normal_imag_eigvec[:no_flux_variables,i]).real, 0):
                normal_imag_eigvec[:,i] = 1j * normal_imag_eigvec[:,i]

        # Construct the basis change matrix T that brings M to eigval*J
        T_plus = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))
        T_minus = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))
        T = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))

        for i in range(len(imag_eigval)):
            sigma = 1j * np.sign((imag_eigvec[:,i].T @ J @ np.conj(imag_eigvec[:,i]))/1j)

            T_plus = np.hstack((T_plus, np.sqrt(2) * (normal_imag_eigvec[:,i].real).reshape(-1,1)))
            #T_minus = np.hstack((T_minus, np.sqrt(2)*((np.sign(sigma) * np.conjugate(normal_imag_eigvec[:,i])).imag).reshape(-1,1)))
            T_minus = np.hstack((T_minus, np.sqrt(2)*((sigma * (-1) * np.conjugate(normal_imag_eigvec[:,i])).real).reshape(-1,1)))

        T = np.hstack((T, T_plus))
        T = np.hstack((T, T_minus))
        T = np.hstack((T, zero_eigvec))

        # Verify that the matrix T satisies the conditions it must satisfy
        assert T.shape[0] == M.shape[0], "There is an error in the construction of the normal form transfromation matrix T. \
            It must have the same dimension as the input matrix"
        assert np.allclose(J, T.T @ J @ T, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. \
            It must be symplectic, T.T @ J @ T = J"
        assert np.allclose(T.imag, 0, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. It must be real"
        #assert np.allclose(T[:len(imag_eigval), len(imag_eigval):], np.zeros((len(imag_eigval),len(imag_eigval)))) and \
            #np.allclose(T[len(imag_eigval):, :len(imag_eigval)], np.zeros((len(imag_eigval),len(imag_eigval)))), \
            #"There is an error in the construction of the normal form transfromation matrix T. It must be a block diagonal matrix"

        # Obtain and return the output matrix 
        M_out = np.linalg.pinv(T) @ M @ T
        return M_out, T
    
    # If Omega == True:
    # - Construct a matrix V = (eimag_eigvec.real, imag_eigvec.imag, zero_eigvec)
    # - Normalize the matrix V such that J = V.T @ M @ V
    elif Omega == True: 

        # Eigenvectors orthonormalization under the inner product x.T @ M @ y  
        normal_imag_eigvec = np.empty((M.shape[0], 0))

        for i, eigval in enumerate(imag_eigval):

            # Repeated eigenvalues
            if i > 0 and np.allclose(imag_eigval[i-1], imag_eigval[i]):
                j += 1 
                summary = 0
                for m in range(1,j+1):
                    Phi_star = np.conj(normal_imag_eigvec[:,i-m].T @ M @ np.conj(imag_eigvec[:,i]))
                    summary += Phi_star * normal_imag_eigvec[:,i-m].reshape(-1,1) 

                eigvec = (imag_eigvec[:,i].reshape(-1,1) - sigma * summary)
                norm = np.abs(np.sqrt(eigvec.T @ M @ np.conj(eigvec)))
                normal_imag_eigvec = np.hstack((normal_imag_eigvec, eigvec/norm)) 
                continue
            j = 0

            # First eigenvalues
            alpha = imag_eigvec[:,i].T @ M @ np.conj(imag_eigvec[:,i])
            sigma = 1j * np.sign(alpha/1j)
            Phi = np.sqrt(sigma * alpha)
            normal_imag_eigvec = np.hstack((normal_imag_eigvec, (imag_eigvec[:,i].reshape(-1,1))/Phi)) 

            # Verify the orthonormalization of the term i
            assert np.allclose(normal_imag_eigvec[:,i].T @ M @ np.conj(normal_imag_eigvec[:,i]), 1j, rtol = tol) \
                or np.allclose(normal_imag_eigvec[:,i].T @ M @ np.conj(normal_imag_eigvec[:,i]), -1j, rtol = tol), \
                "There is an error in the orthonormalization of an eigenvector from a purely imaginary eigenvalue"

        # Add an aditional phase to the imaginary eigenvectors, if it is necessary, to to obtain a block diagonal V matrix if it is possible
        for i in range(len(imag_eigval)):
            if np.allclose(sum(normal_imag_eigvec[:no_flux_variables,i]).real, 0):
                normal_imag_eigvec[:,i] = 1j * normal_imag_eigvec[:,i]
        
        # Construct the basis change matrix V
        V = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))

        V = np.hstack((V, normal_imag_eigvec.real))
        V = np.hstack((V, normal_imag_eigvec.imag))
        V = np.hstack((V, zero_eigvec))

        # Multiply each eigenvector by sqrt(2) such that J = V.T @ M @ V
        for i, eigval in enumerate(imag_eigval):
            V[:,i] = np.sqrt(2) * V[:,i]
            V[:,i+len(imag_eigval)] = np.sqrt(2) * V[:,i+len(imag_eigval)]

        # Verify that the matrix T satisies the conditions it must satisfy
        assert V.shape[0] == M.shape[0], "There is an error in the construction of the normal form transfromation matrix V. \
            It must have the same dimension as the input matrix"
        assert np.allclose(J, V.T @ M @ V, rtol = tol), \
            "There is an error in the construction of the basis matrix change V. It must satisfy V.T @ M @ V = J"

        # Return the output matrix 
        return J, V
