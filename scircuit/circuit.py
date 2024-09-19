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

        self.E_2B, self.E_symplectic, self.symplectic_basis_change, self.number_of_pairs = self.omega_function()

        self.Total_energy_2B, self.Total_energy_symplectic_basis, self.hamiltonian = self.hamiltonian_function_lineal_elements()

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
        assert K.shape[1] == F.shape[1] - np.linalg.matrix_rank(K)
        assert np.allclose(F @ K, np.zeros((F.shape[0], K.shape[1]))) == True

        return Fcut, Floop, F, K

    def omega_function(self):
        # Obtain E_2B matrix
        E_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Capacitor) == True:
                E_2B[i, i + self.no_elements] = 0.5
                E_2B[i + self.no_elements, i] = -0.5

            if isinstance(elem[2], Inductor) == True:
                E_2B[i, i + self.no_elements] = -0.5
                E_2B[i + self.no_elements, i] = 0.5

        # Obtain omega matrix, E, in its non symplectic form
        E_non_symplectic = self.K.T @ E_2B @ self.K

        # Obtain the symplectic form of the omega matrix, E, and the basis change matrix
        E_symplectic, symplectic_basis_change, number_of_pairs = symplectic_form(E_non_symplectic)

        return E_2B, E_symplectic, symplectic_basis_change, number_of_pairs

    def hamiltonian_function_lineal_elements(self):
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

        # Remove from the previous matrix the rows and columns that correspond to variables without dynamics
        number_rows, _ = Total_energy_symplectic_basis.shape
        rows_columns_to_delete = []

        for i in range(2*self.number_of_pairs, number_rows):
            if np.all(np.abs(Total_energy_symplectic_basis[i,:]) <= 1e-12) and np.all(np.abs(Total_energy_symplectic_basis[:,i]) <= 1e-12):
                rows_columns_to_delete.append(i)
        
        Total_energy_symplectic_basis = np.delete(Total_energy_symplectic_basis, rows_columns_to_delete, axis=0)
        Total_energy_symplectic_basis = np.delete(Total_energy_symplectic_basis, rows_columns_to_delete, axis=1)

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
        rank_TEF_22 = np.linalg.matrix_rank(TEF_22)
        rows_TEF_22, columns_TEF_22 = TEF_22.shape

        if rank_TEF_22 < min(rows_TEF_22, columns_TEF_22):
            raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

        # If there is solution, calculate the final matrix expression for the total energy function, which is the Hamiltonian
        hamiltonian = TEF_11 - TEF_12@np.linalg.inv(TEF_22)@TEF_21

        return Total_energy_2B, Total_energy_symplectic_basis, hamiltonian
