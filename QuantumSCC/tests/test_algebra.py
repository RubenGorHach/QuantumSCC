import unittest

import numpy as np

from QuantumSCC.algebra import *

class Test_Gauss_Jordan_method(unittest.TestCase):

    def test_direct_Gauss_Jordan(self):

        M_before_GJ = np.array([[-1., -1.,  0.,  0.,  1.,],
                                [ 1.,  1., -1.,  0.,  0.,],
                                [ 0.,  0.,  1., -1.,  0.,],
                                [ 0.,  0.,  0.,  1., -1.,]])
        
        M_after_GJ = np.array([[-1.,  0.,  0., -1.,  1.,],
                               [ 0., -1.,  0.,  0.,  1.,],
                               [ 0.,  0., -1.,  0.,  1.,],
                               [ 0.,  0.,  0.,  0.,  0.,]])
        
        M_test, _ = GaussJordan(M_before_GJ)

        self.assertTrue(np.allclose(M_test, M_after_GJ))

    def test_reverse_Gauss_Jordan(self):

        M_before_reverse_GJ = np.array([[-1.,  0.,  0., -1.,  1.],
                                        [ 0., -1.,  0.,  0.,  1.],
                                        [ 0.,  0., -1.,  0.,  1.]])
        
        M_after_reverse_GJ = np.array([[ 1.,  0.,  0.,  1., -1.],
                                       [ 0.,  1.,  0.,  0., -1.],
                                       [ 0.,  0.,  1.,  0., -1.]])
        
        M_test = reverseGaussJordan(M_before_reverse_GJ)
        
        self.assertTrue(np.allclose(M_test, M_after_reverse_GJ))

class Test_symplectic_form_function(unittest.TestCase):

    def test_symplectic_form_transformation(self):

        matrix_before_transformation = np.array([[ 0.,  0.,  0.,  1., -1.],
                                                 [ 0.,  0.,  0.,  0., -1.],
                                                 [ 0.,  0.,  0.,  0.,  0.],
                                                 [-1.,  0.,  0.,  0.,  0.],
                                                 [ 1.,  1.,  0.,  0.,  0.]])
        
        matrix_after_transformation = np.array([[ 0.,  0.,  1.,  0.,  0.],
                                                [ 0.,  0.,  0.,  1.,  0.],
                                                [-1., -0.,  0.,  0.,  0.],
                                                [-0., -1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.]])
        
        matrix_test, _, _ = omega_symplectic_transformation(matrix_before_transformation, no_compact_flux_variables=0, no_flux_variables=2)
        
        self.assertTrue(np.allclose(matrix_after_transformation, matrix_test))

    def test_symplectic_form_basis_change(self):

        matrix_before_transformation = np.array([[ 0.,  0.,  0.,  1., -1.],
                                                 [ 0.,  0.,  0.,  0., -1.],
                                                 [ 0.,  0.,  0.,  0.,  0.],
                                                 [-1.,  0.,  0.,  0.,  0.],
                                                 [ 1.,  1.,  0.,  0.,  0.]])
        
        canonical_matrix, canonical_basis_change, _ = omega_symplectic_transformation(matrix_before_transformation, no_compact_flux_variables=0, no_flux_variables=2)

        self.assertTrue(np.allclose(canonical_matrix, canonical_basis_change.T @ matrix_before_transformation @ canonical_basis_change))

class Test_canonical_transformation_quadratic_hamiltonian(unittest.TestCase):

    def test_basis_change_matrix_T_dimensions(self):

        hamiltonian = np.array([[ 5.963e-01,  1.491e-01, -1.342e-16,  1.573e-17],
                                [ 1.491e-01,  5.963e-01, -6.702e-17,  3.462e-17],
                                [-1.342e-16, -6.702e-17,  5.963e-01, -1.491e-01],
                                [ 1.573e-17,  3.462e-17, -1.491e-01,  5.963e-01]])
        
        J = np.block([[ np.zeros((2,2)), np.eye(2)], 
                      [-np.eye(2), np.zeros((2,2))]])
        
        _, T = symplectic_transformation(J @ hamiltonian, no_flux_variables=2)

        self.assertTrue(T.shape[0] == hamiltonian.shape[0] and T.shape[1] == hamiltonian.shape[1])

    def test_basis_change_matrix_T_symplectic(self):

        hamiltonian = np.array([[ 5.963e-01,  1.491e-01, -1.342e-16,  1.573e-17],
                                [ 1.491e-01,  5.963e-01, -6.702e-17,  3.462e-17],
                                [-1.342e-16, -6.702e-17,  5.963e-01, -1.491e-01],
                                [ 1.573e-17,  3.462e-17, -1.491e-01,  5.963e-01]])
        
        J = np.block([[ np.zeros((2,2)), np.eye(2)], 
                      [-np.eye(2), np.zeros((2,2))]])
        
        _, T = symplectic_transformation(J @ hamiltonian, no_flux_variables=2)
        
        self.assertTrue(np.allclose(J, T.T @ J @ T))

    def test_basis_change_matrix_T_real(self):

        hamiltonian = np.array([[ 5.963e-01,  1.491e-01, -1.342e-16,  1.573e-17],
                                [ 1.491e-01,  5.963e-01, -6.702e-17,  3.462e-17],
                                [-1.342e-16, -6.702e-17,  5.963e-01, -1.491e-01],
                                [ 1.573e-17,  3.462e-17, -1.491e-01,  5.963e-01]])
        
        J = np.block([[ np.zeros((2,2)), np.eye(2)], 
                      [-np.eye(2), np.zeros((2,2))]])
        
        _, T = symplectic_transformation(J @ hamiltonian, no_flux_variables=2)

        self.assertTrue(np.allclose(T.imag, 0))


        


