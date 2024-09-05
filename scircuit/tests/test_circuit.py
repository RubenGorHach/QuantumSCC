import unittest

import numpy as np

from scircuit.circuit import (
    Circuit, 
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

        Fcut = np.array([[ 1.,  0.,  0.,  1., -1.],
                         [ 0.,  1.,  0.,  0., -1.],
                         [ 0.,  0.,  1.,  0., -1.]])
        
        self.assertTrue(np.allclose(cr.Fcut, Fcut))

    def test_Floop_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]
        cr = Circuit(elements)

        Floop = np.array([[-1.,  0.,  0.,  1.,  0.], 
                          [ 1.,  1.,  1.,  0.,  1.]])

        self.assertTrue(np.allclose(cr.Floop, Floop))

    def test_F_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        F = np.array([[ 1.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.], 
                      [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.], 
                      [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.]])
        
        self.assertTrue(np.allclose(cr.F, F))

    def test_Kernel(self):
        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        self.assertTrue(cr.K.shape[1] == cr.F.shape[1]-np.linalg.matrix_rank(cr.K))
        self.assertTrue(np.allclose(cr.F@cr.K, np.zeros((cr.F.shape[0], cr.K.shape[1]))))


class Test_omega_function(unittest.TestCase):

    def test_omega_2B(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        E_2B = np.array([[ 0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0. ],
                         [ 0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0.,   0.,   0. ],
                         [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0.,   0. ],
                         [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5,  0. ],
                         [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.5],
                         [-0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                         [ 0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                         [ 0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                         [ 0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0.,   0. ],
                         [ 0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.,   0. ]])
        
        self.assertTrue(np.allclose(cr.E_2B, E_2B))

    def test_omega_canonical(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        E_canonical = np.array([[ 0.,  1.,  0.,  0.,  0.],
                                [-1.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.]])
        
        self.assertTrue(np.allclose(cr.E_canonical, E_canonical))



if __name__ == '__main__':
    unittest.main()

