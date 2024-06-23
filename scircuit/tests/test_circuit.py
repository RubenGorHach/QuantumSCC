import unittest
from assertpy import assert_that

import numpy as np

from scircuit.circuit import (
    Circuit, 
    GaussJordan, 
    reverseGaussJordan
)
from scircuit.elements import (
    Capacitor, 
    Inductor
)

class Test_Kirchhoff_matrix(unittest.TestCase):

    def test_Fcut_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        Fcut = np.array([[ 1.,  0.,  0.,  0., -1.],
                         [ 0.,  1.,  0.,  0., -1.],
                         [ 0.,  0.,  1.,  1., -1.]])
        
        self.assertTrue(np.array_equal(cr.Fcut, Fcut))

    def test_Floop_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]
        cr = Circuit(elements)

        Floop = np.array([[ 0.,  0., -1.,  1.,  0.], 
                          [ 1.,  1.,  1.,  0.,  1.]])

        self.assertTrue(np.array_equal(cr.Floop, Floop))

    def test_F_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        F = np.array([[ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  0.,  1.,  1., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.], 
                      [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.]])
        
        self.assertTrue(np.array_equal(cr.F, F))

    def test_Kernel(self):
        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        K = np.array([[ 0.,  1.,  0.,  0.,  0.],
                      [ 0.,  1.,  0.,  0.,  0.],
                      [-1.,  1.,  0.,  0.,  0.],
                      [ 1.,  0.,  0.,  0.,  0.],
                      [ 0.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  1.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  0.],
                      [ 0.,  0.,  0.,  0.,  1.],
                      [ 0.,  0.,  0.,  0.,  1.],
                      [ 0.,  0., -1., -1., -1.]])
        
        self.assertTrue(np.array_equal(cr.K, K))


class Test_Gauss_Jordan_method(unittest.TestCase):

    def test_direct_Gauss_Jordan(self):

        M_before_GJ = np.array([[-1., -1.,  0.,  0.,  1.,],
                                [ 1.,  1., -1.,  0.,  0.,],
                                [ 0.,  0.,  1., -1.,  0.,],
                                [ 0.,  0.,  0.,  1., -1.,]])
        
        M_after_GJ = np.array([[-1.,  0.,  0.,  0.,  1.,],
                               [ 0., -1.,  0.,  0.,  1.,],
                               [ 0.,  0., -1., -1.,  1.,],
                               [ 0.,  0.,  0.,  0.,  0.,]])
        
        M_test, _ = GaussJordan(M_before_GJ)

        self.assertTrue(np.array_equal(M_test, M_after_GJ))

    def test_reverse_Gauss_Jordan(self):

        M_before_reverse_GJ = np.array([[-1.,  0.,  0.,  0.,  1.],
                                        [ 0., -1.,  0.,  0.,  1.],
                                        [ 0.,  0., -1., -1.,  1.]])
        
        M_after_reverse_GJ = np.array([[ 1.,  0.,  0.,  0., -1.],
                                       [ 0.,  1.,  0.,  0., -1.],
                                       [ 0.,  0.,  1.,  1., -1.]])
        
        M_test = reverseGaussJordan(M_before_reverse_GJ)
        
        self.assertTrue(np.array_equal(M_test, M_after_reverse_GJ))


class Test_omega_function(unittest.TestCase):

    def test_omega_2B(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        omega_2B = np.array([[ 0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5],
                             [-0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0. ]])
        
        self.assertTrue(np.array_equal(cr.omega_2B, omega_2B))

    def test_omega_alpha_beta(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        omega_alpha_beta = np.array([[ 0.,  0.,  0.,  0.,  0.],
                                     [ 0.,  0.,  1.,  0.,  0.],
                                     [ 0., -1.,  0.,  0.,  0.],
                                     [ 0.,  0.,  0.,  0.,  0.],
                                     [ 0.,  0.,  0.,  0.,  0.]])
        
        self.assertTrue(np.array_equal(cr.omega_alpha_beta, omega_alpha_beta))



if __name__ == '__main__':
    unittest.main()

