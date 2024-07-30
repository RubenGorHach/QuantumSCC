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

class Test_canonical_form_function(unittest.TestCase):

    def test_canonical_form_transformation(self):

        matrix_before_transformation = np.array([[ 0.,  0., -1.,  0.],
                                                 [ 0.,  0., -1.,  0.],
                                                 [ 1.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  0.,  0.]])
        
        matrix_after_transformation = np.array([[ 0.,  1.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.]])
        
        matrix_test, _ = canonical_form(matrix_before_transformation)
        
        self.assertTrue(np.allclose(matrix_after_transformation, matrix_test))

    def test_canonical_form_basis_change(self):

        matrix_before_transformation = np.array([[ 0.,  0., -1.,  0.],
                                                 [ 0.,  0., -1.,  0.],
                                                 [ 1.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  0.,  0.]])
        
        canonical_matrix, canonical_basis_change = canonical_form(matrix_before_transformation)

        self.assertTrue(np.allclose(canonical_matrix, canonical_basis_change @ matrix_before_transformation @ canonical_basis_change.T))

        


