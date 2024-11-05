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

        self.omega_2B, self.omega_symplectic, self.symplectic_basis_change, self.number_of_pairs = self.omega_function()

        self.Total_energy_2B, self.Total_energy_symplectic_basis, self.hamiltonian = self.hamiltonian_function_quadratic()

    def Kirchhoff(self):
        # Calculate the full Fcut
        Fcut = np.zeros((self.no_nodes, self.no_elements))

        # This matrix accounts for the "intensity" variables. In the
        # Kirchhoff equations, for a given node, the contribution of an
        # intensity is negative or positive depending on whether the node
        # is the origin or destination of this oriented element.
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

        # F = [[Fcut, 0], [0, Floop]] = Kirchhoff matrix
        F = np.block(
            [
                [Fcut, np.zeros((Fcut.shape[0], Floop.shape[1]))],
                [np.zeros((Floop.shape[0], Fcut.shape[1])), Floop],
            ]
        )

        # K = [[Floop.T, 0], [0, Fcut.T]] = Kernel matrix
        K = np.block(
            [
                [Floop.T, np.zeros((Floop.shape[1], Fcut.shape[0]))],
                [np.zeros((Fcut.shape[1], Floop.shape[0])), Fcut.T],
            ]
        )

        # Make sure K is correct
        assert K.shape[1] == F.shape[1] - np.linalg.matrix_rank(K), "There is an error in the construction of the Kernell"
        assert np.allclose(F @ K, np.zeros((F.shape[0], K.shape[1]))) == True, "There is an error in the construction of the Kernell"

        return Fcut, Floop, F, K

    def omega_function(self):
        # Obtain omega_2B matrix
        omega_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Capacitor) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

            if isinstance(elem[2], Inductor) == True:
                omega_2B[i, i + self.no_elements] = -0.5
                omega_2B[i + self.no_elements, i] = 0.5

        # Obtain omega matrix in its non symplectic form
        omega_non_symplectic = self.K.T @ omega_2B @ self.K

        # Obtain the symplectic form of the omega matrix and the basis change matrix
        omega_symplectic, symplectic_basis_change, number_of_pairs = symplectic_form(omega_non_symplectic)

        # Remove the zeros columns and rows from omega_symplectic
        omega_symplectic = omega_symplectic[:2 * number_of_pairs, :2 * number_of_pairs]

        return omega_2B, omega_symplectic, symplectic_basis_change, number_of_pairs

    def hamiltonian_function_quadratic(self):
        # IMPORTANT -> This is the fuction that calculates the Hamiltonian for the lineal elements of the circuit, and considering circuits with only linear elements

        # Calculate the initial total energy function matrix (prior to the change of variable given by the Kirchhoff's equtions)
        Total_energy_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Capacitor) == True:
                capacitor = elem[2]
                Total_energy_2B[i, i] = capacitor.energy() # Energy of the capacitor in GHz by default

            if isinstance(elem[2], Inductor) == True:
                inductor = elem[2]
                Total_energy_2B[i + self.no_elements, i + self.no_elements] = inductor.energy() # Energy of the inductor in GHz by default

        # Calculate the total energy function matrix after the change of variable given by the Kirchhoff's equations
        Total_energy_after_Kirchhoff = self.K.T @ Total_energy_2B @ self.K
        
        # Calculate the total energy function matrix after the change of variable given by the symplectic form of omega
        Total_energy_symplectic_basis = self.symplectic_basis_change.T @ Total_energy_after_Kirchhoff @ self.symplectic_basis_change

        # If the size of the new Total_energy_symplectic_basis matrix is equal to 2*self.number_of_pairs, this matrix is the Hamiltonian
        if len(Total_energy_symplectic_basis) == 2*self.number_of_pairs:
            hamiltonian = Total_energy_symplectic_basis

            return Total_energy_2B, Total_energy_symplectic_basis, hamiltonian
        
        # If the previous condition does not happend, we need to solve d(Total_energy_symplectic_basis)/dw = 0

        # Decompose the Total_energy_symplectic_basis matrix into 4 blocks: ((TEF_11, TEF_12);(TEF_21, TEF_22)), according to the matrix E_symplectic
        TEF_11 = Total_energy_symplectic_basis[:2*self.number_of_pairs, :2*self.number_of_pairs]
        TEF_12 = Total_energy_symplectic_basis[:2*self.number_of_pairs, 2*self.number_of_pairs:]
        TEF_21 = Total_energy_symplectic_basis[2*self.number_of_pairs:, :2*self.number_of_pairs]
        TEF_22 = Total_energy_symplectic_basis[2*self.number_of_pairs:, 2*self.number_of_pairs:]

        assert np.allclose(TEF_12, TEF_21.T) == True, "There is an error in the decomposition of the total energy function matrix in blocks"

        # Verify that the equation dH/dw = 0 has a solution by testing that TEF_22 has a inverse form
        try: 
            TEF_22_inv = pseudo_inv(TEF_22, tol = 1e-15)
        except np.linalg.LinAlgError:
            raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

        # If there is solution, calculate the final matrix expression for the total energy function, which is the Hamiltonian
        hamiltonian = TEF_11 - TEF_12 @ TEF_22_inv @ TEF_21

        return Total_energy_2B, Total_energy_symplectic_basis, hamiltonian
    

    def hamiltonian_quantization(self):

        # Define the classical Hamiltonian
        classical_H = self.hamiltonian

        # Get the quantum canonical Hamiltonian and the basis change matrix
        canonical_H, T = canonical_transformation_quadratic_hamiltonian(classical_H)

        # Proceed with the second quantization of the Hamiltonian

        

        
