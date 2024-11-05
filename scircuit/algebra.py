
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


def first_nonzero_index(v, tol=1e-14):
    """
    It returns the index of the firs non-zero element in the vector v, with a tolerance.

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
                    ordered_pairs.append((b, -y / np.sqrt(b * 0.5), x / np.sqrt(b * 0.5)))

    assert len(pairs) == len(ordered_pairs), "There is an error in the construction of the 'ordered_pairs' array"

    # Construct the base change matrix, V
    V = np.array([upper_v for b, upper_v, _ in ordered_pairs] + [lower_v for b, _, lower_v in ordered_pairs] + zeros).T

    # Construct the symplectic matrix, J, for the matrix A.
    dimension = A.shape[0]
    number_of_pairs = len(ordered_pairs)
    J = np.zeros((dimension, dimension))
    I = np.eye(number_of_pairs)
    
    J[:number_of_pairs, number_of_pairs:number_of_pairs*2] = I
    J[number_of_pairs:number_of_pairs*2, :number_of_pairs] = -I

    assert np.allclose(J, (V.T @ A @ V)), "There is an error in the construction of the symplectic matrix"

    return J, V, number_of_pairs


def canonical_transformation_quadratic_hamiltonian(H, tol=1e-15):
    """
    Produce the canonical form of a quadratic Hamiltonian
    
    Parameters
    ----------
        H:
            Quadratic Hamiltonian matrix to which the algorithm is applied.
        tol:
            Tolerance below which the element is considered zero. By default, it is 1e-15.
    """

    # Assert that the input Hamiltonian is a square matrix
    assert H.shape[0] == H.shape[1], "The input Hamiltonian matrix must be square"
    assert H.shape[0] % 2 == 0, "The input Hamiltonian matrix must have an even dimension"

    # Define the symplectic matrix J with the correct dimensions
    dimension = H.shape[0]
    half_dimension = dimension//2
    J = np.zeros((dimension, dimension))
    I = np.eye(half_dimension)
    
    J[:half_dimension, half_dimension:dimension] = I
    J[half_dimension:dimension, :half_dimension] = -I

    # Define the equation-of-motion matrix K
    K = J @ H
    assert np.allclose(np.zeros((dimension,dimension)), J @ K + K.T @ J), "There is an error in the construction of the equation-of-motion matrix"

    # Obtain the eigenvalues and eigenvectors of K
    K_eigenval, K_eigenvec = np.linalg.eig(K)
    assert np.linalg.matrix_rank(K_eigenvec, tol=1e-15) == dimension, "There are degenerate eigenvalues -> I fail my assumption and the program is not ready"

    zero_eigenval, zero_eigenvec = np.empty(0), np.empty((dimension, 0))
    imag_eigenval, imag_eigenvec = np.empty(0), np.empty((dimension, 0))
    
    # Organize the eigenvalues with their eigenvectors in two groups: zero and pure imaginary eigenvalues
    for i, eigenval in enumerate(K_eigenval):

        if np.allclose(eigenval.real, 0) and np.allclose(eigenval.imag, 0): 
            zero_eigenval = np.hstack((zero_eigenval, eigenval))
            zero_eigenvec = np.hstack((zero_eigenvec, K_eigenvec[:,i].reshape(-1,1)))

        elif np.allclose(eigenval.real, 0) and eigenval.imag > 0:
            imag_eigenval = np.hstack((imag_eigenval, eigenval)) # Positive eigenvalue
            imag_eigenvec = np.hstack((imag_eigenvec, K_eigenvec[:,i].reshape(-1,1))) # Eigenvector of the positive eigenvalue

    # Organize the pure imaginary positives eigenvalues from smallest to largest
    imag_index = np.argsort(imag_eigenval.imag)
    imag_eigenval = imag_eigenval[imag_index]
    imag_eigenvec = imag_eigenvec[:, imag_index]

    # Verify there are no real or non pure complex eigenvalues   
    assert 2 * len(imag_eigenval) + len(zero_eigenval) == len(K_eigenval), "Matrix JH has positive or not pure complex eigenvalues"
    
    # Verify that there are no zero eigenvalues
    assert len(zero_eigenval) == 0, "There are zero eigenvalues and the code is not ready"

    # Normalize the pure imaginary eigenvectors under the standard symplectic inner product x.T @ J @ y 
    normalized_imag_eigenvec = np.empty((dimension, 0))

    for i, eigenval in enumerate(imag_eigenval):

        # Repeated eigenvalues
        if i > 0 and np.allclose(imag_eigenval[i-1], imag_eigenval[i]):
            j += 1 
            summary = 0
            for m in range(1,j+1):
                Phi_star = np.conj(normalized_imag_eigenvec[:,i-m].T @ J @ np.conj(imag_eigenvec[:,i]))
                summary += Phi_star * normalized_imag_eigenvec[:,i-m].reshape(-1,1) 

            eigenvec = imag_eigenvec[:,i].reshape(-1,1) - sigma * summary 
            norm = np.sqrt(eigenvec.T @ J @ np.conj(eigenvec))
            normalized_imag_eigenvec = np.hstack((normalized_imag_eigenvec, eigenvec/norm)) 
            continue
        j = 0

        # First eigenvalues
        alpha = imag_eigenvec[:,i].T @ J @ np.conj(imag_eigenvec[:,i])
        sigma = 1j * np.sign(alpha/1j)
        Phi = np.sqrt(sigma * alpha)
        normalized_imag_eigenvec = np.hstack((normalized_imag_eigenvec, (imag_eigenvec[:,i].reshape(-1,1))/Phi)) 

        # Verify the orthonormalization of the term i
        assert np.allclose(normalized_imag_eigenvec[:,i].T @ J @ np.conj(normalized_imag_eigenvec[:,i]), 1j) \
            or np.allclose(normalized_imag_eigenvec[:,i].T @ J @ np.conj(normalized_imag_eigenvec[:,i]), -1j), \
            "There is an error in the orthonormalization of an eigenvector from a purely imaginary eigenvalue"
        
    # Construct the normal form transfromation matrix T such that H_normal = T.T @ H @ T
    T_plus = np.empty((dimension, 0))
    T_minus = np.empty((dimension, 0))
    T = np.empty((dimension, 0))

    # Purely imaginary pairs, c = 6
    for i, _ in enumerate(imag_eigenval):
        sigma = 1j * np.sign((imag_eigenvec[:,i].T @ J @ np.conj(imag_eigenvec[:,i]))/1j)

        T_plus = np.hstack((T_plus, np.sqrt(2)*(normalized_imag_eigenvec[:,i].real).reshape(-1,1)))
        T_minus = np.hstack((T_minus, np.sqrt(2)*((np.conjugate(normalized_imag_eigenvec[:,i]*(sigma))).real).reshape(-1,1)))

    # Create the total matrix T
    T  = np.hstack((T, T_plus))
    T  = np.hstack((T, T_minus))
    
    assert T.shape[0] == dimension, "There is an error in the construction of the normal form transfromation matrix T. \
        It must have the same dimension as the Hamiltonian"
    assert np.allclose(J, T.T @ J @ T), "There is an error in the construction of the normal form transfromation matrix T. \
        It must satisfy T.T @ J @ T = J"
    assert np.allclose(T.imag, 0), "There is an error in the construction of the normal form transfromation matrix T. It must be real"

    # Calculate the Hamiltonian matrix in its normal form, H_n
    H_n = T.T @ H @ T

    return H_n, T

   
   
    


    

    

        








    
   

   
        
        
    





    



    




    



    

