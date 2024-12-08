"""
circuit.py contains the classes for the circuit and their properties
"""

from typing import Any, Sequence, Optional, Union

import numpy as np

from .elements import (
    Capacitor,
    Inductor,
    Junction,
    Loop,
)

from .elements import Junction

from .algebra import *

Edge = tuple[int, int, object]

class Circuit:
    """
    Class that contains circuit properties and use Kirchoff laws to obtain the circuit equations


    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each branch
            of the circuit.

    """

    elements: list[Edge]
    no_elements: int
    node_dictionary: dict[Any, int]

    def __init__(self, elements: list[Edge]) -> None:
        """Define a circuit from a list of edges and circuit elements."""
        nodes = set([a for a, _, _ in elements] + [b for _, b, _ in elements])
        node_dictionary = {a: i for i, a in enumerate(nodes)}

        complete_elements = []
        for a, b, elt in elements:
            if isinstance(elt, Junction) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])
        for a, b, elt in elements:
            if isinstance(elt, Junction) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt.cap])
            elif isinstance(elt, Capacitor) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])
        for a, b, elt in elements:
            if isinstance(elt, Inductor) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])

        self.elements = complete_elements

        self.no_elements = len(self.elements)
        self.node_dict = node_dictionary
        self.no_nodes = len(node_dictionary)


        self.Fcut, self.Floop, self.F, self.K = self.Kirchhoff()

        self.omega_2B, self.omega_symplectic, self.V, self.no_independent_variables = self.omega_function()

        self.linear_quadratic_hamiltonian, self.interaction_quadratic_hamiltonian, self.nonlinear_quadratic_hamiltonian, self.vector_JJ = self.classical_hamiltonian_function()

        self.linear_canonical_hamiltonian, self.T = self.linear_hamiltonian_quantization()

        self.linear_quantum_hamiltonian, self.G = self.linear_hamiltonian_second_quantization()

        self.nonlinear_hamiltonian_quantization()

    def Kirchhoff(self):
        """
        Contructs the total Kirchhoff matrix F of the circuit and its kernel K.

        Returns
        ----------
        F_cut:
            Kirchhoff matrix with respect to the Kirchhoff current law (KCL).
        F_loop:
            Kirchhoff matrix with respect to the Kirchhoff voltage law (KVL).
        F:
            Total Kirchhoff matrix.
        K:
            Kernel of the total Kirchhoff matrix.
        """

        # Preallocate the F_cut matrix
        Fcut = np.zeros((self.no_nodes, self.no_elements))

        # Construct the F_cut matrix according to KCL
        for n_edge, (orig_node, dest_node, _) in enumerate(self.elements):
            Fcut[orig_node, n_edge] = -1
            Fcut[dest_node, n_edge] = +1

        Fcut, order = GaussJordan(Fcut)
        self.elements = [self.elements[i] for i in order]
        Fcut = reverseGaussJordan(remove_zero_rows(Fcut))

        # Fcut = [1, A]
        # Floop = [-A.T, 1]
        n = len(Fcut)
        A = Fcut[:, n:]
        Floop = np.hstack((-A.T, np.eye(A.shape[1])))

        # F = [[Floop, 0], [0, Fcut]] = Kirchhoff matrix
        F = np.block(
            [
                [Floop, np.zeros((Floop.shape[0], Fcut.shape[1]))],
                [np.zeros((Fcut.shape[0], Floop.shape[1])), Fcut],
            ]
        )

        # K = [[Fcut.T, 0], [0, Floop.T]] = Kernel matrix
        K = np.block(
            [
                [Fcut.T, np.zeros((Fcut.shape[1], Floop.shape[0]))],
                [np.zeros((Floop.shape[1], Fcut.shape[0])), Floop.T],
            ]
        )

        # Make sure K is correct
        assert K.shape[1] == F.shape[1] - np.linalg.matrix_rank(K), "There is an error in the construction of the Kernell"
        assert np.allclose(F @ K, np.zeros((F.shape[0], K.shape[1]))) == True, "There is an error in the construction of the Kernell"

        return Fcut, Floop, F, K

    def omega_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It calculates the symplectic form of 
        the two-form omega and the basis change matrix.
        
        Returns
        ----------
        omega_2B:
            Matrix expression of the two-form omega.
        omega_symplectic:
            Symplectic expression of the two-form omega.
        V:
            Basis change matrix that transform omega to its symplectic form.
        number_of_pairs:
            Number of pairs of non-zero conjugate eigenvalues the two-form omega has. 
        """
        # Obtain omega_2B matrix
        omega_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Junction) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

            if isinstance(elem[2], Capacitor) == True:
                omega_2B[i, i + self.no_elements] = -0.5
                omega_2B[i + self.no_elements, i] = 0.5

            elif isinstance(elem[2], Inductor) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

        # Obtain omega matrix in the Kirchhoff equations basis
        omega_non_symplectic = self.K.T @ omega_2B @ self.K

        # Obtain the symplectic form of the omega matrix and the basis change matrix
        omega_symplectic, V = symplectic_transformation(omega_non_symplectic,  no_flux_variables = self.Fcut.shape[0], Omega = True)

        # Remove the zeros columns and rows from omega_symplectic
        no_independent_variables = np.linalg.matrix_rank(omega_symplectic)
        omega_symplectic = omega_symplectic[:no_independent_variables, :no_independent_variables]

        return omega_2B, omega_symplectic, V, no_independent_variables

    def classical_hamiltonian_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It constructs the symplified 
        quadratic Hamiltonian matrices from the energy function of the Lagrangian.

        Returns
        ----------
        linear_quadratic_hamiltonian:
            Matrix expression of the quadratic linear Hamiltonian.
        interaction_quadratic_hamiltonian:
            Matrix expression of interaction between the linear and non linear quadratic Hamiltonian.
        nonlinear_quadratic_hamiltonian:
            Matrix expression of the quadratic non-linear Hamiltonian.
        vector_JJ:
            Vector of the non-linear variables that go inside the cosine in the final Hamiltonian expression.
        """

        # Calculate the initial quadratic total energy function matrix (prior to the change of variable given by the Kirchhoff's equtions)
        quadratic_energy = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Inductor) == True:
                inductor = elem[2]
                quadratic_energy[i, i] = 2 * inductor.energy() # Energy of the inductor in GHz by default

            elif isinstance(elem[2], Capacitor) == True:
                capacitor = elem[2]
                quadratic_energy[i + self.no_elements, i + self.no_elements] = 2 * capacitor.energy() # Energy of the capacitor in GHz by default

        # Calculate the quadratic energy function matrix after the change of variables given by the Kirchhoff's equations
        quadratic_energy_after_Kirchhoff = self.K.T @ quadratic_energy @ self.K

        # Calculate the quadratic energy function matrix after the change of variables given by the symplectic form of omega
        quadratic_energy_symplectic_basis = self.V.T @ quadratic_energy_after_Kirchhoff @ self.V


        # Construct the initial vectors of the Josephson Juntion energy such that E = -Ej cos(vector.T @ R)
        vector_JJ = np.empty((quadratic_energy.shape[0], 0))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Junction) == True:
                aux = np.zeros((quadratic_energy.shape[0], 1))
                aux[i,0] = 1
                vector_JJ = np.hstack((vector_JJ, aux))

        # Calculate the JJ vector under the change of variables given by the Kirchhoff's equations and the symplectic form of omega
        vector_JJ = self.V.T @ self.K.T @ vector_JJ

        # Verify, JJ vector consider only dynamical variables and delete the entries corresponding to the non dynamical variables
        if  np.allclose(vector_JJ[self.no_independent_variables:,:], 0) == False:
            raise ValueError("The Energy of the Josephson Junction depends on non-dynamical variables. We cannot solve the circuit.")
        
        vector_JJ = vector_JJ[:self.no_independent_variables,:]


        # If the size of the new Total_energy_symplectic_basis matrix is equal to 2*self.number_of_pairs, this matrix is the Hamiltonian
        if len(quadratic_energy_symplectic_basis) == self.no_independent_variables:
            quadratic_hamiltonian = quadratic_energy_symplectic_basis

        # If the previous condition does not happend, we need to solve d(Total_energy_symplectic_basis)/dw = 0
        else:

            # Decompose the Total_energy_symplectic_basis matrix into 4 blocks: ((TEF_11, TEF_12);(TEF_21, TEF_22)), according to the matrix E_symplectic
            TEF_11 = quadratic_energy_symplectic_basis[:self.no_independent_variables, :self.no_independent_variables]
            TEF_12 = quadratic_energy_symplectic_basis[:self.no_independent_variables, self.no_independent_variables:]
            TEF_21 = quadratic_energy_symplectic_basis[self.no_independent_variables:, :self.no_independent_variables]
            TEF_22 = quadratic_energy_symplectic_basis[self.no_independent_variables:, self.no_independent_variables:]

            assert np.allclose(TEF_12, TEF_21.T) == True, "There is an error in the decomposition of the total energy function matrix in blocks"

            # Verify that the equation dH/dw = 0 has a solution by testing that TEF_22 has a inverse form
            try: 
                TEF_22_inv = pseudo_inv(TEF_22, tol = 1e-15)
            except np.linalg.LinAlgError:
                raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

            # If there is solution, calculate the final matrix expression for the total energy function, which is the Hamiltonian
            quadratic_hamiltonian = TEF_11 - TEF_12 @ TEF_22_inv @ TEF_21


        # Verify the resulting quadratic Hamiltonian is block diagonal
        assert np.allclose(quadratic_hamiltonian[quadratic_hamiltonian.shape[0]:, :quadratic_hamiltonian.shape[1]], 0) and \
            np.allclose(quadratic_hamiltonian[:quadratic_hamiltonian.shape[0], quadratic_hamiltonian.shape[1]:], 0), \
            'The classical Hamiltonian matrix must be block diagonal. There could be an error in the construction of the basis change matrix V'
        
        # Separate the quadratic Hamiltonian matrix into three components: linear, non-linear and interaction

        flux_nonlinear_indexes = nonzero_indexes(vector_JJ[:self.no_independent_variables//2])
        charge_nonlinear_indexes = nonzero_indexes(vector_JJ[self.no_independent_variables//2:self.no_independent_variables])

        nonlinear_indexes_set1 = flux_nonlinear_indexes + [self.no_independent_variables//2 + x for x in flux_nonlinear_indexes]
        nonlinear_indexes_set2 = charge_nonlinear_indexes + [self.no_independent_variables//2 + x for x in charge_nonlinear_indexes]

        nonlinear_indexes = sorted(set(nonlinear_indexes_set1 + nonlinear_indexes_set2))
        linear_indexes = np.setdiff1d(np.arange(self.no_independent_variables), nonlinear_indexes)
        
        # Linear quadratic Hamiltonian
        linear_quadratic_hamiltonian = quadratic_hamiltonian[np.ix_(linear_indexes, linear_indexes)]

        # Non-linear quadratic Hamiltonian
        nonlinear_quadratic_hamiltonian = quadratic_hamiltonian[np.ix_(nonlinear_indexes, nonlinear_indexes)]

        # Interaction quadratic Hamiltonian
        interaction_quadratic_hamiltonian = quadratic_hamiltonian.copy()
        for i in nonlinear_indexes:
            for j in nonlinear_indexes:
                interaction_quadratic_hamiltonian[i,j] = 0.
        for i in linear_indexes:
            for j in linear_indexes:
                interaction_quadratic_hamiltonian[i,j] = 0.
    

        np.set_printoptions(precision=2)
        print('vector_JJ:')
        print(vector_JJ)
        print('-----------------------------')

        print('Quadratic Hamiltonian:')
        print(quadratic_hamiltonian)
        print('-----------------------------')

        print('linear Quadratic Hamiltonian:')
        print(linear_quadratic_hamiltonian)
        print('-----------------------------')

        print('nonlinear Quadratic Hamiltonian:')
        print(nonlinear_quadratic_hamiltonian)
        print('-----------------------------')

        print('interaction quadratic Hamiltonian')
        print(interaction_quadratic_hamiltonian)
        print('-----------------------------')

        # Returns
        return linear_quadratic_hamiltonian, interaction_quadratic_hamiltonian, nonlinear_quadratic_hamiltonian, vector_JJ
    

    def linear_hamiltonian_quantization(self):
        """
        Calculates the linear quantum Hamiltonian in its canonical form.
        
        Returns
        ----------
        linear_canonical_hamiltonian:
            Cannical matrix expression of the quantum Hamiltonian.
        T:
            Basis change matrix that brings the Hamiltonian to its quantum canonical form
        
        """

        # Define the linear quadratic Hamiltonian
        linear_quadratic_hamiltonian = self.linear_quadratic_hamiltonian
        dimension = linear_quadratic_hamiltonian.shape[0]

        #print(linear_quadratic_hamiltonian)

        # Get the quantum canonical Hamiltonian and the basis change matrix
        J = np.block([[np.zeros((dimension//2, dimension//2)), np.eye(dimension//2)],
                      [-np.eye(dimension//2), np.zeros((dimension//2, dimension//2))]])
        
        dynamical_matrix = J @ linear_quadratic_hamiltonian
        _, T = symplectic_transformation(dynamical_matrix, no_flux_variables=linear_quadratic_hamiltonian.shape[0]//2, Omega=False)
        linear_canonical_hamiltonian = T.T @ linear_quadratic_hamiltonian @ T

        return linear_canonical_hamiltonian, T


    def linear_hamiltonian_second_quantization(self):
        """
        Calculates the quantum Hamiltonian in the ladder operators basis.
        
        Returns
        ----------
        quantum_hamiltonian:
            Matrix expression of the quantum Hamiltonian in the ladder operators basis.
        G:
            Basis change matrix that perform the second quantization of the Hamiltonian
        """ 

        # Define the canonical Hamiltonian
        linear_canonical_hamiltonian = self.linear_canonical_hamiltonian

        # Define the basis change matrix G
        I = np.eye(len(linear_canonical_hamiltonian)//2)
        G = (1 / np.sqrt(2)) * np.block([[I, I], [-1j * I, 1j * I]])

        # Calculate the matrix expression of the Quantum Hamiltonian in the ladder operators basis.
        linear_quantum_hamiltonian = np.conj(G.T) @ linear_canonical_hamiltonian @ G

        assert np.allclose(linear_quantum_hamiltonian, linear_canonical_hamiltonian), \
        "The matrix expression for the Hamiltonian in the ladder operators basis must be the same as the canonical Hamiltonian matrix."

        return linear_quantum_hamiltonian, G


    def nonlinear_hamiltonian_quantization(self):

        nonlinear_quadratic_hamiltonian = self.nonlinear_quadratic_hamiltonian
        dimension = nonlinear_quadratic_hamiltonian.shape[0]

        # If the nonlinear flux variables does not appear in the quadratic non-linear Hamiltonian -> Proceed with the charge basis
        if np.allclose(nonlinear_quadratic_hamiltonian[:dimension//2, :dimension//2], 0) and \
            np.allclose(nonlinear_quadratic_hamiltonian[:dimension//2, dimension//2:dimension], 0):

            print('Caso: 1')

        # If the nonlinear flux variables appears also in the quadratic non-linear Hamiltonian -> 
        elif np.allclose(nonlinear_quadratic_hamiltonian[:dimension//2, dimension//2:dimension], 0):
            print('Caso: 2')

        else:
            print('Caso: 3')





        
        
