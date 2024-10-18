
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
            Tolerance below which the element is considered zero. By default, it is 1e-16.
    """

    # Assert that the input Hamiltonian is a square matrix
    assert H.shape[0] == H.shape[1], "The input Hamiltonian matrix must be square"
    assert H.shape[0] % 2 == 0, "The input Hamiltonian matrix must have an even dimension"

    # Define the symplectic matrix J with the correct dimensions
    dimension = H.shape[0]
    half_dimension = int(dimension/2)
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

    zero_eigenval, zero_eigenvec = np.empty(0), np.empty((dimension, 0)) # There should not appear zero eigenvalues
    real_eigenval, real_eigenvec = np.empty(0), np.empty((dimension, 0))
    purely_imag_eigenval, purely_imag_eigenvec = np.empty(0), np.empty((dimension, 0))
    complex_eigenval, complex_eigenvec = np.empty(0), np.empty((dimension, 0))
    
    # Organize the eigenvalues with their eigenvectors in four categories
    for i, eigenval in enumerate(K_eigenval):

        if np.allclose(eigenval.real, 0) and np.allclose(eigenval.imag, 0): # There should not appear zero eigenvalues
            zero_eigenval = np.hstack((zero_eigenval, eigenval))
            zero_eigenvec = np.hstack((zero_eigenvec, K_eigenvec[:,i].reshape(-1,1)))

        elif np.allclose(eigenval.imag, 0):
            real_eigenval = np.hstack((real_eigenval, eigenval))
            real_eigenvec = np.hstack((real_eigenvec, K_eigenvec[:,i].reshape(-1,1)))

        elif np.allclose(eigenval.real, 0):
            purely_imag_eigenval = np.hstack((purely_imag_eigenval, eigenval))
            purely_imag_eigenvec = np.hstack((purely_imag_eigenvec, K_eigenvec[:,i].reshape(-1,1)))
        
        else:
            complex_eigenval = np.hstack((complex_eigenval, eigenval))
            complex_eigenvec = np.hstack((complex_eigenvec, K_eigenvec[:,i].reshape(-1,1)))
    
    # Verify that there are no zero eigenvalues
    assert len(zero_eigenval) == 0, "There are zero eigenvalues and it is not possible"

    # Rearrange the elements in each category by grouping them in pairs of opposite sign

    for i in range (0, len(real_eigenval), 2):
        for j in range (i+1, len(real_eigenval)):
            if np.allclose(np.abs(real_eigenvec[:,i]), np.abs(real_eigenvec[:,j])):
                real_eigenval[i+1], real_eigenval[j] = real_eigenval[j], real_eigenval[i+1]
                real_eigenvec[:, [i+1, j]] = real_eigenvec[:, [j, i+1]]

    for i in range (0,len(purely_imag_eigenval), 2):
        for j in range (i+1, len(purely_imag_eigenval)):
            if np.allclose(np.abs(purely_imag_eigenvec[:,i]), np.abs(purely_imag_eigenvec[:,j])):
                purely_imag_eigenval[i+1], purely_imag_eigenval[j] = purely_imag_eigenval[j], purely_imag_eigenval[i+1]
                purely_imag_eigenvec[:, [i+1, j]] = purely_imag_eigenvec[:, [j, i+1]]

    for i in range (0, len(complex_eigenval), 2):
        for j in range(i+1, len(complex_eigenval)):
            l = 1
            if np.allclose(np.abs(complex_eigenvec[:,i]), np.abs(complex_eigenvec[:,j])):
                complex_eigenval[i+l], complex_eigenval[j] = complex_eigenval[j], complex_eigenval[i+l]
                complex_eigenvec[:, [i+l, j]] = complex_eigenvec[:, [j, i+l]]
                l += 1 
        for j in range(i+1, len(complex_eigenval)):
            if np.allclose(complex_eigenval[i], -complex_eigenval[j]):
                complex_eigenval[i+1], complex_eigenval[j] = complex_eigenval[j], complex_eigenval[i+1]
                complex_eigenvec[:, [i+1, j]] = complex_eigenvec[:, [j, i+1]]


    # Normalize the eigenvectors under the standard symplectic inner product x.T @ J @ y and rearrange the elements to obtain x.T @ J @ y = 1

    # Real pairs 
    for i in range (0, len(real_eigenval), 2):
        a = np.sqrt(np.abs(real_eigenvec[:,i].T @ J @ real_eigenvec[:,i+1]))
        real_eigenvec[:,i] = real_eigenvec[:,i]*(1/a)
        real_eigenvec[:,i+1] = real_eigenvec[:,i+1]*(1/a)
        
        if np.allclose(real_eigenvec[:,i].T @ J @ real_eigenvec[:,i+1], -1): # Rearrange the elements to obtain x.T @ J @ y = 1
            real_eigenval[i], real_eigenval[i+1] = real_eigenval[i+1], real_eigenval[i]
            real_eigenvec[:, [i, i+1]] = real_eigenvec[:, [i+1, i]]

        assert np.allclose(real_eigenvec[:,i].T @ J @ real_eigenvec[:,i+1], 1), "There is an error in the orthonormalization of an eigenvector from a real eigenvalue"

    # Purely imaginary pairs
    for i in range (0, len(purely_imag_eigenval), 2):
        a = np.sqrt(np.abs(purely_imag_eigenvec[:,i].T @ J @ purely_imag_eigenvec[:,i+1]))
        purely_imag_eigenvec[:,i] = purely_imag_eigenvec[:,i]*(1/a)
        purely_imag_eigenvec[:,i+1] = purely_imag_eigenvec[:,i+1]*(1/a)

        if np.allclose(purely_imag_eigenvec[:,i].T @ J @ purely_imag_eigenvec[:,i+1], -1j): # Rearrange the elements to obtain x.T @ J @ y = 1j
            purely_imag_eigenval[i], purely_imag_eigenval[i+1] = purely_imag_eigenval[i+1], purely_imag_eigenval[i]
            purely_imag_eigenvec[:, [i, i+1]] = purely_imag_eigenvec[:, [i+1, i]]

        assert np.allclose(purely_imag_eigenvec[:,i].T @ J @ purely_imag_eigenvec[:,i+1], 1j), "There is an error in the orthonormalization of an eigenvector from a purely imaginary eigenvalue"

    # Complex quadruplets
    for i in range (0, len(complex_eigenval), 2):
        a = np.sqrt(complex_eigenvec[:,i].T @ J @ complex_eigenvec[:,i+1])
        complex_eigenvec[:,i] = complex_eigenvec[:,i]*(1/a)
        complex_eigenvec[:,i+1] = complex_eigenvec[:,i+1]*(1/a)
        
        if np.allclose(complex_eigenvec[:,i].T @ J @ complex_eigenvec[:,i+1], -1): # Rearrange the elements to obtain x.T @ J @ y = 1
            complex_eigenval[i], complex_eigenval[i+1] = complex_eigenval[i+1], complex_eigenval[i]
            complex_eigenvec[:, [i, i+1]] = complex_eigenvec[:, [i+1, i]]

        assert np.allclose(complex_eigenvec[:,i].T @ J @ complex_eigenvec[:,i+1], 1), "There is an error in the orthonormalization of an eigenvector from a complex eigenvalue"
    

    # Construct the normal form transfromation matrix T such that H_normal = T.T @ H @ T
    T_plus = np.empty((dimension, 0))
    T_minus = np.empty((dimension, 0))
    T = np.empty((dimension, 0))

    # Real pairs, c = 1
    for i in range(len(real_eigenval)):
        if i % 2 == 0:
            T_plus = np.hstack((T_plus, real_eigenvec[:,i].reshape(-1,1)))
        if i % 2 == 1:
            T_minus = np.hstack((T_minus, real_eigenvec[:,i].reshape(-1,1)))

    # Complex quadruplets, c = 2
    for i in range(len(complex_eigenval)):

        if i % 4 == 0:
            T_plus = np.hstack((T_plus, np.sqrt(2)*(complex_eigenvec[:,i].real).reshape(-1,1)))
        if i % 4 == 2:
            T_plus = np.hstack((T_plus, np.sqrt(2)*(complex_eigenvec[:,i].imag).reshape(-1,1)))

        if i % 4 == 1:
            T_minus = np.hstack((T_minus, np.sqrt(2)*(complex_eigenvec[:,i].real).reshape(-1,1)))
        if i % 4 == 3:
            T_minus = np.hstack((T_minus, (-1)*np.sqrt(2)*(complex_eigenvec[:,i].imag).reshape(-1,1)))

    # Purely imaginary pairs, c = 6
    sigma = 1j # x.T @ J @ y = sigma = 1j
    for i in range(len(purely_imag_eigenval)):
        if i % 2 == 0:
            T_plus = np.hstack((T_plus, np.sqrt(2)*(purely_imag_eigenvec[:,i].real).reshape(-1,1)))
        if i % 2 == 1:
            T_minus = np.hstack((T_minus, np.sqrt(2)*((np.conjugate(purely_imag_eigenvec[:,i]*(-sigma))).real).reshape(-1,1)))

    # Create the total matrix T
    T  = np.hstack((T, T_plus))
    T  = np.hstack((T, T_minus))
    assert T.shape[0] == T.shape[1], "There is an error in the construction of the normal form transfromation matrix T. It must be square"
    assert T.shape[0] == dimension, "There is an error in the construction of the normal form transfromation matrix T. It must have the same dimension as the Hamiltonian"
    assert np.allclose(J, T.T @ J @ T), "There is an error in the construction of the normal form transfromation matrix T. It must satisfy T.T @ J @ T = J"
    assert np.allclose(T.imag, 0), "There is an error in the construction of the normal form transfromation matrix T. It must be real"
    

    # Calculate the Hamiltonian matrix in its normalm form, H_n
    H_n = T.T @ H @ T

    # Prints for testing
    print("The initial Hmailtonian:")
    print(H)
    print("The normal Hmailtonian:")
    print(H_n)

    print("The normal form transfromation matrix T:")
    print(T)

    print("The eigenvalues of K")
    print(purely_imag_eigenval)
    
    print("The product T.T @ J @ T, that must be J:")
    print(T.T @ J @ T)

 

  

    
        
        
    





    



    




    



    

