"""
circuit.py contains the classes for the circuit and their properties
"""

from typing import Any, Sequence, Optional, Union

import numpy as np

from .elements import (
    Capacitor,
    Inductor,
    Loop,
)

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

        self.elements = [
            (node_dictionary[a], node_dictionary[b], elt) for a, b, elt in elements
        ]
        self.no_elements = len(self.elements)
        self.node_dict = node_dictionary
        self.no_nodes = len(node_dictionary)
        

        self.Fcut, self.Floop, self.F, self.K = self.Kirchhoff()

        self.omega_2B, self.omega_symplectic, self.V, self.no_independent_variables = self.omega_function()

        self.Total_energy_2B, self.Total_energy_symplectic_basis, self.classical_hamiltonian = self.quadratic_classical_hamiltonian_function()

        self.canonical_hamiltonian, self.T = self.hamiltonian_canonization()

        self.quantum_hamiltonian, self.G = self.hamiltonian_second_quantization()

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
            if isinstance(elem[2], Capacitor) == True:
                omega_2B[i, i + self.no_elements] = -0.5
                omega_2B[i + self.no_elements, i] = 0.5

            elif isinstance(elem[2], Inductor) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

        # Obtain omega matrix in the Kirchhoff equations basis
        omega_non_symplectic = self.K.T @ omega_2B @ self.K

        # Obtain the symplectic form of the omega matrix and the basis change matrix
        omega_symplectic, V = symplectic_transformation(omega_non_symplectic, Omega = True, no_flux_variables = self.Fcut.shape[0])

        # Remove the zeros columns and rows from omega_symplectic
        no_independent_variables = np.linalg.matrix_rank(omega_symplectic)
        omega_symplectic = omega_symplectic[:no_independent_variables, :no_independent_variables]

        return omega_2B, omega_symplectic, V, no_independent_variables

    def quadratic_classical_hamiltonian_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It constructs the symplified 
        classical Hamiltonian matrix from the energy function of the Lagrangian.

        Returns
        ----------
        Total_energy_2B:
            Matrix expression of the total energy in the initial basis.
        Total_energy_symplectic_basis:
            Matrix expression of the total energy in the symplectic basis of omega.
        classical_hamiltonian:
            Matrix expression of the classical Hamiltonian of the circuit.
        """
        # IMPORTANT -> This is the fuction that calculates the Hamiltonian for the lineal elements of the circuit, and considering circuits with only linear elements

        # Calculate the initial total energy function matrix (prior to the change of variable given by the Kirchhoff's equtions)
        Total_energy_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Inductor) == True:
                inductor = elem[2]
                Total_energy_2B[i, i] = 2 * inductor.energy() # Energy of the inductor in GHz by default

            elif isinstance(elem[2], Capacitor) == True:
                capacitor = elem[2]
                Total_energy_2B[i + self.no_elements, i + self.no_elements] = 2 * capacitor.energy() # Energy of the capacitor in GHz by default

        # Calculate the total energy function matrix after the change of variable given by the Kirchhoff's equations
        Total_energy_after_Kirchhoff = self.K.T @ Total_energy_2B @ self.K
        
        # Calculate the total energy function matrix after the change of variable given by the symplectic form of omega
        Total_energy_symplectic_basis = self.V.T @ Total_energy_after_Kirchhoff @ self.V

        # If the size of the new Total_energy_symplectic_basis matrix is equal to 2*self.number_of_pairs, this matrix is the Hamiltonian
        if len(Total_energy_symplectic_basis) == self.no_independent_variables:
            hamiltonian = Total_energy_symplectic_basis

            return Total_energy_2B, Total_energy_symplectic_basis, hamiltonian
        
        # If the previous condition does not happend, we need to solve d(Total_energy_symplectic_basis)/dw = 0

        # Decompose the Total_energy_symplectic_basis matrix into 4 blocks: ((TEF_11, TEF_12);(TEF_21, TEF_22)), according to the matrix E_symplectic
        TEF_11 = Total_energy_symplectic_basis[:self.no_independent_variables, :self.no_independent_variables]
        TEF_12 = Total_energy_symplectic_basis[:self.no_independent_variables, self.no_independent_variables:]
        TEF_21 = Total_energy_symplectic_basis[self.no_independent_variables:, :self.no_independent_variables]
        TEF_22 = Total_energy_symplectic_basis[self.no_independent_variables:, self.no_independent_variables:]

        assert np.allclose(TEF_12, TEF_21.T) == True, "There is an error in the decomposition of the total energy function matrix in blocks"

        # Verify that the equation dH/dw = 0 has a solution by testing that TEF_22 has a inverse form
        try: 
            TEF_22_inv = pseudo_inv(TEF_22, tol = 1e-15)
        except np.linalg.LinAlgError:
            raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

        # If there is solution, calculate the final matrix expression for the total energy function, which is the Hamiltonian
        classical_hamiltonian = TEF_11 - TEF_12 @ TEF_22_inv @ TEF_21

        # Verify the Hamiltonian is block diagonal
        assert np.allclose(classical_hamiltonian[classical_hamiltonian.shape[0]:, :classical_hamiltonian.shape[1]], 0) and \
            np.allclose(classical_hamiltonian[:classical_hamiltonian.shape[0], classical_hamiltonian.shape[1]:], 0), \
            'The classical Hamiltonian matrix must be block diagonal. There could be an error in the construction of the basis change matrix V'

        return Total_energy_2B, Total_energy_symplectic_basis, classical_hamiltonian
    

    def hamiltonian_canonization(self):
        """
        Calculates the quantum Hamiltonian in its canonical form.
        
        Returns
        ----------
        canonical_hamiltonian:
            Cannical matrix expression of the quantum Hamiltonian.
        T:
            Basis change matrix that brings the Hamiltonian to its quantum canonical form
        
        """

        # Define the classical Hamiltonian
        classical_hamiltonian = self.classical_hamiltonian

        # Get the quantum canonical Hamiltonian and the basis change matrix
        J = self.omega_symplectic
        dynamical_matrix = J @ classical_hamiltonian
        _, T = symplectic_transformation(dynamical_matrix, Omega=False)
        canonical_hamiltonian = T.T @ classical_hamiltonian @ T

        return canonical_hamiltonian, T


    def hamiltonian_second_quantization(self):
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
        canonical_hamiltonian = self.canonical_hamiltonian

        # Define the basis change matrix G
        I = np.eye(len(canonical_hamiltonian)//2)
        G = (1 / np.sqrt(2)) * np.block([[I, I], [-1j * I, 1j * I]])

        # Calculate the matrix expression of the Quantum Hamiltonian in the ladder operators basis.
        quantum_hamiltonian = np.conj(G.T) @ canonical_hamiltonian @ G

        assert np.allclose(quantum_hamiltonian, canonical_hamiltonian), \
        "The matrix expression for the Hamiltonian in the ladder operators basis must be the same as the canonical Hamiltonian matrix."

        return quantum_hamiltonian, G





        
        
