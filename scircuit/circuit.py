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

        self.omega_2B, self.omega = self.omega_function()

        self.hamiltonian_2B, self.hamiltonian = self.hamiltonian_function()
        


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
        F = np.bmat(
            [
                [Fcut, np.zeros((Fcut.shape[0], Floop.shape[1]))],
                [np.zeros((Floop.shape[0], Fcut.shape[1])), Floop],
            ]
        )

        # K = [[Floop.T, 0], [0, Fcut.T]] = Kernel matrix
        K = np.bmat(
            [
                [Floop.T, np.zeros((Floop.shape[1], Fcut.shape[0]))],
                [np.zeros((Fcut.shape[1], Floop.shape[0])), Fcut.T],
            ]
        )

        # Make sure K is correct
        assert K.shape[1] == F.shape[1]-np.linalg.matrix_rank(K)
        assert np.allclose(F@K, np.zeros((F.shape[0], K.shape[1]))) == True

        return Fcut, Floop, F, K
    
    def omega_function(self):
        
        # Obtain omega_2B matrix
        omega_2B = np.zeros((2*self.no_elements,2*self.no_elements)) 
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Capacitor) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

            if isinstance(elem[2], Inductor) == True:
                omega_2B[i, i + self.no_elements] = -0.5
                omega_2B[i + self.no_elements, i] = 0.5
                    
        # Obtain omega_alpha_beta matrix
        omega = self.K.T @ omega_2B @ self.K

        return omega_2B, omega
    
    def hamiltonian_function(self):

        # Calculate the Hamiltonian prior to the change of variable

        # IMPORTANT -> This is the Hamiltonian for the lineal elements of the circuit, and considering circuits with only linear elements
        hamiltonian_2B = np.zeros((2*self.no_elements, 2*self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Capacitor) == True:
                cap = elem[2]
                hamiltonian_2B[i, i] = 1/(2*cap.value())
            
            if isinstance(elem[2], Inductor) == True:
                ind = elem[2]
                hamiltonian_2B[i + self.no_elements, i + self.no_elements] = 1/(2*ind.value())

        # Calculate the Hamiltonian after the change of variable
        hamiltonian = self.K.T @ hamiltonian_2B @ self.K

        return hamiltonian_2B, hamiltonian


def GaussJordan(M):
    # Obtain the matrix dimensions
    nrows, ncolumns = M.shape
    assert nrows <= ncolumns
    M = M.copy()
    order = np.arange(ncolumns)
    for ii in range(nrows):
        k = np.argmax(np.abs(M[ii, ii:]))
        if k != 0:
            Maux = M.copy()
            M[:, ii], M[:, ii + k] = Maux[:, ii + k], Maux[:, ii]
            order[ii], order[ii + k] = order[ii + k], order[ii]
        for jj in range(ii + 1, nrows):
            M[jj, :] -= M[ii, :] * M[jj, ii] / M[ii, ii]

    return M, order


def reverseGaussJordan(M):
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
    row_norm_1 = np.sum(np.abs(M), -1)
    M = M[(row_norm_1 > tol), :]
    return M


