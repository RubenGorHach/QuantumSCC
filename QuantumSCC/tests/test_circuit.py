import unittest

import numpy as np

from QuantumSCC.circuit import (
    Circuit, 
)

from QuantumSCC.elements import (
    Capacitor, 
    Inductor
)

class Test_Kirchhoff_matrix(unittest.TestCase):

    def test_Fcut_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]

        cr = Circuit(elements)
        
        Fcut = np.array([[ 1.,  1.,  0.,  0., -1.],
                         [ 0.,  0.,  1.,  0., -1.],
                         [ 0.,  0.,  0.,  1., -1.]])
        
        self.assertTrue(np.allclose(cr.Fcut, Fcut))

    def test_Floop_matrix(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]
        cr = Circuit(elements)

        Floop = np.array([[-1.,  1., -0., -0.,  0.],
                          [ 1.,  0.,  1.,  1.,  1.]])

        self.assertTrue(np.allclose(cr.Floop, Floop))

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

        omega_2B = np.array([[ 0.,   0.,   0.,   0.,   0.,  -0.5,  0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.5,  0. ],
                             [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.5],
                             [ 0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,  -0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,  -0.5,  0.,   0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,  -0.5,  0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.,   0.,   0.,   0.,  -0.5,  0.,   0.,   0.,   0.,   0. ]])
        
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

    def test_hamiltonian(self):

        C = Capacitor(value = 1, unit='GHz')
        L = Inductor(value = 1, unit = 'GHz')
        elements = [(0,2,C), (0,1,L), (1,2,L)] 

        cr = Circuit(elements)

        hamiltonian = np.array([[1.,  0.],
                                [0.,  2.]])
        
        self.assertTrue(np.allclose(cr.quadratic_hamiltonian, hamiltonian))


class Test_linear_examples(unittest.TestCase):

    def test_LC_oscillator(self):

        C = Capacitor(value = 1, unit='pF')
        L = Inductor(value = 1, unit = 'nH')

        LC = [(0,1,L), (0,1,C)]

        cr = Circuit(LC)

        omega = 1e-9/(np.sqrt(C.cValue*1e-12*L.lValue*1e-9))
        expected_hamiltonian = np.array([[omega, 0], [0, omega]])

        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected_hamiltonian))

    def test_2C_and1L_parallel(self):

        C = Capacitor(value = 1, unit='pF')
        L = Inductor(value = 1, unit = 'nH')

        circuit_2C_1L_parallel = [(0,1,C),(0,1,C),(0,1,L)]

        cr = Circuit(circuit_2C_1L_parallel)

        omega = 1e-9/(np.sqrt(2*C.cValue*1e-12*L.lValue*1e-9))
        expected_hamiltonian = np.array([[omega, 0], [0, omega]])

        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected_hamiltonian))

    def test_2C_and1L_series(self):

        C = Capacitor(value = 1, unit='pF')
        L = Inductor(value = 1, unit = 'nH')

        circuit_2C_1L_series = [(0,1,C),(1,2,C),(2,0,L)]

        cr = Circuit(circuit_2C_1L_series)

        omega = 1e-9/(np.sqrt(0.5*C.cValue*1e-12*L.lValue*1e-9))
        expected_hamiltonian = np.array([[omega, 0], [0, omega]])

        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected_hamiltonian))
    
    def test_capacitance_coupled_oscillators(self):

        C1 = Capacitor(value = 1, unit='pF')
        C2 = Capacitor(value = 1, unit='pF')
        Cg = Capacitor(value = 2, unit='pF')
        L1 = Inductor(value = 1, unit = 'nH')
        L2 = Inductor(value = 1, unit = 'nH')

        coupled_oscillators = [(0,1,L1), (1,2,Cg), (2,0,L2), (0,1,C1),(2,0,C2)]

        cr = Circuit(coupled_oscillators)

        omega1 = 10 * np.sqrt(2)
        omega2 = 10 * np.sqrt(10)
        expected_hamiltonian = np.array([[omega1, 0, 0, 0], 
                                         [0, omega2, 0, 0], 
                                         [0, 0, omega1, 0], 
                                         [0, 0, 0, omega2]])

        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected_hamiltonian))

    def test_star_circuit(self):

        C = Capacitor(value = 1, unit='pF')
        L = Inductor(value = 1, unit = 'nH')

        symmetric_starcircuit = [(0,1,C), (1,2,C), (2,0,C), (0,3,L), (1,3,L), (2,3,L)]

        cr = Circuit(symmetric_starcircuit)

        omega = 18.2574110
        expected_hamiltonian = np.array([[omega, 0, 0, 0], 
                                         [0, omega, 0, 0], 
                                         [0, 0, omega, 0], 
                                         [0, 0, 0, omega]])

        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected_hamiltonian))








if __name__ == '__main__':
    unittest.main()

