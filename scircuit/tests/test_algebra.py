import unittest

import numpy as np

from scircuit.algebra import *

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

        matrix_before_transformation = np.array([[ 0.,  0., -1.,  0.],
                                                 [ 0.,  0., -1.,  0.],
                                                 [ 1.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  0.,  0.]])
        
        matrix_after_transformation = np.array([[ 0.,  1.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.]])
        
        matrix_test, _, _ = symplectic_form(matrix_before_transformation)
        
        self.assertTrue(np.allclose(matrix_after_transformation, matrix_test))

    def test_symplectic_form_basis_change(self):

        matrix_before_transformation = np.array([[ 0.,  0., -1.,  0.],
                                                 [ 0.,  0., -1.,  0.],
                                                 [ 1.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  0.,  0.]])
        
        canonical_matrix, canonical_basis_change, _ = symplectic_form(matrix_before_transformation)

        self.assertTrue(np.allclose(canonical_matrix, canonical_basis_change.T @ matrix_before_transformation @ canonical_basis_change))

    def test_symplectic_form_random_antisymmetric_matrix(self):

        size = 3
        random_block = np.random.normal(size=(size, size))
        random_antisymmetric_matrix_by_blocks = np.zeros((2*size, 2*size))
        random_antisymmetric_matrix_by_blocks[:size,  size:] = random_block
        random_antisymmetric_matrix_by_blocks[size:, :size] = -random_block.T

        symplectic_matrix, symplectic_basis_change, _ = symplectic_form(random_antisymmetric_matrix_by_blocks)

        self.assertTrue(np.allclose(symplectic_matrix, symplectic_basis_change.T @ random_antisymmetric_matrix_by_blocks @ symplectic_basis_change))

        


