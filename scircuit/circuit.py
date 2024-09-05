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

        self.E_2B, self.E_canonical, self.canonical_basis_change, self.number_of_pairs = self.omega_function()

        self.hamiltonian_2B, self.hamiltonian, self.hamiltonian_xi = self.hamiltonian_function_lineal_elements()

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

        # Obtain omega matrix, E, in its non canonical form
        E_non_canonical = self.K.T @ E_2B @ self.K

        # Obtain the canonical form of the omega matrix, E, and the basis change matrix
        E_canonical, canonical_basis_change, number_of_pairs = canonical_form(E_non_canonical)

        return E_2B, E_canonical, canonical_basis_change, number_of_pairs

    def hamiltonian_function_lineal_elements(self):
        # IMPORTANT -> This is the Hamiltonian for the lineal elements of the circuit, and considering circuits with only linear elements

        # Calculate the Hamiltonian prior to the change of variable given by the Kirchhoff's equtions
        hamiltonian_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Capacitor) == True:
                capacitor = elem[2]
                hamiltonian_2B[i, i] = 1 / (2 * capacitor.value())

            if isinstance(elem[2], Inductor) == True:
                inductor = elem[2]
                hamiltonian_2B[i + self.no_elements, i + self.no_elements] = 1 / (
                    2 * inductor.value()
                )

        # Calculate the Hamiltonian after the change of variable given by the Kirchhoff's equations
        hamiltonian_after_Kirchhoff = self.K.T @ hamiltonian_2B @ self.K

        # Calculate the Hamiltonian after the change of variable given by the canonical form of omega
        hamiltonian = self.canonical_basis_change @ hamiltonian_after_Kirchhoff @ self.canonical_basis_change.T

        # Decompose the Hamiltonian matrix into 4 blocks: ((H_11, H_12);(H_21, H_22)), according to the matrix E_canonical
        hamiltonian_11 = hamiltonian[:2*self.number_of_pairs, :2*self.number_of_pairs]
        hamiltonian_12 = hamiltonian[:2*self.number_of_pairs, 2*self.number_of_pairs:]
        hamiltonian_21 = hamiltonian[2*self.number_of_pairs:, :2*self.number_of_pairs]
        hamiltonian_22 = hamiltonian[2*self.number_of_pairs:, 2*self.number_of_pairs:]

        assert np.allclose(hamiltonian_12, hamiltonian_21.T) == True, "There is an error in the decomposition of the Hamiltonian in blocks"

        # Verify that the equation dH/dw = 0 has a solution by testing that hamiltonian_22 has a pseudo-inverse form
        rank_hamiltonian_22 = np.linalg.matrix_rank(hamiltonian_22)
        rows_hamiltonian_22, columns_hamiltonian_22 = hamiltonian_22.shape

        if rank_hamiltonian_22 < min(rows_hamiltonian_22, columns_hamiltonian_22):
            raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

        # If there is solution, calculate the final matrix expression for the hamiltonian, hamiltonian_xi
        hamiltonian_xi = hamiltonian_11 - hamiltonian_12@np.linalg.pinv(hamiltonian_22)@hamiltonian_21

        return hamiltonian_2B, hamiltonian, hamiltonian_xi
