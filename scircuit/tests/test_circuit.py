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
        
        self.assertTrue(np.allclose(cr.omega_2B, omega_2B))

    def test_omega_symplectic(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)

        omega_symplectic = np.array([[ 0.,  1.],
                                     [-1.,  0.]])
        
        self.assertTrue(np.allclose(cr.omega_symplectic, omega_symplectic))

class Test_Hamiltonian_function(unittest.TestCase):

    def test_Total_energy_2B(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,2,C), (0,1,L), (1,2,L)] 

        cr = Circuit(elements)

        Total_energy_2B = np.array([[1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 1.]])
        
        self.assertTrue(np.allclose(cr.Total_energy_2B, Total_energy_2B))

    def test_Total_energy_symplectic_basis(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,2,C), (0,1,L), (1,2,L)] 

        cr = Circuit(elements)

        Total_energy_symplectic_basis = np.array([[1., 0., 0.],
                                                  [0., 1., 1.],
                                                  [0., 1., 2.]])
        
        self.assertTrue(np.allclose(cr.Total_energy_symplectic_basis, Total_energy_symplectic_basis))

    def test_hamiltonian(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,2,C), (0,1,L), (1,2,L)] 

        cr = Circuit(elements)

        hamiltonian = np.array([[1.,  0.],
                                [0., 0.5]])
        
        self.assertTrue(np.allclose(cr.classical_hamiltonian, hamiltonian))







if __name__ == '__main__':
    unittest.main()

