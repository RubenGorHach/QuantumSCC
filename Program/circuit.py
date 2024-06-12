"""
circuit.py contains the classes for the circuit and their properties
"""

from typing import Dict, Tuple, List, Sequence, Optional, Union

import numpy as np

import units as unt
from elements import (Capacitor, Inductor, Junction, Loop, Charge,
                                VerySmallCap, VeryLargeCap)


class Circuit:
    """
    Class that contains circuit properties and use Kirchoff laws to obtain the circuit equations


    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each branch
            of the circuit.


    """

    def __init__(
            self,
            elements: List[Tuple[int,int,object]]
            ) -> None:
        
        #######################################################################
        # General circuit attributes
        #######################################################################
        
        self.elements = elements

        # number of elements
        self.ne = len(self.elements)

        # number of nodes
        nnlist = []
        for ii in range(0,self.ne):
            nnlist.append(self.elements[ii][0])
            nnlist.append(self.elements[ii][1])
        nnlist = list(set(nnlist))
        self.nn = len(nnlist)
        # Error message if nodes are incorrectly named
        if max(nnlist) != self.nn-1:
            raise ValueError('The labels of the nodes are not correct. Remember use numbers 0,1,2... and so on.')


        # Kirchhoff matrix test -> Eliminate 
        self.K = self.Kirchhoff()

    
    #######################################################################
    # Circuit functions
    #######################################################################

    # ----------------------------------------------------------------------------------------------------------------

    def Kirchhoff(self):

        # Calculate the full Fcut
        Fcut = np.zeros((self.nn, self.ne))

        for ii in range (0,self.ne):
            Fcut[self.elements[ii][0]][ii] = -1
            Fcut[self.elements[ii][1]][ii] = 1

        # See if Fcut is linearly dependent and delete an ecuation if it is ¡¡WARNING!! I am eliminating always the last equation, 
        #                                                                               I don't know if it is correct 
        rank_Fcut = np.linalg.matrix_rank(Fcut)
        while rank_Fcut < len(Fcut):
            Fcut = np.delete(Fcut, len(Fcut)-1, axis=0) # Delete the last row of Fcut
            rank_Fcut = np.linalg.matrix_rank(Fcut)   # Obtain again the rank

        # Return an error message if the Fcut(n x n) submatrix has rank < n
        n = len(Fcut)
        Fcut_sub = np.zeros((n,n))

        for ii in range (0, n):
            for jj in range (0, n):
                Fcut_sub[ii][jj] = Fcut[ii][jj]

        if np.linalg.matrix_rank(Fcut_sub) < n:
            raise ValueError('The elements are oredered in a wrong way. Try to separate parallel elements.')

        # Apply Gauss-Jordan to obtain the matrix F such Fcut = (1|F)

        IdFcut = self.GaussJordan(Fcut)
        F = IdFcut[:, n:]

        # Calculate Floop as Floop = (-F.T|1)
        mFt = F.T * (-1)
        identity = np.eye(len(mFt))
        Floop = np.hstack((mFt, identity))

        # Calculate the final Kirchhoff matrix K
        nfc, ncc = Fcut.shape
        nfl, ncl = Floop.shape

        zeros1 = np.zeros((nfc,ncl))
        zeros2 = np.zeros((nfl,ncc))

        Fcut_zeros = np.hstack((Fcut, zeros1))
        zeros_Floop = np.hstack((zeros2, Floop))

        K = np.vstack((Fcut_zeros, zeros_Floop))

        return K


    # ----------------------------------------------------------------------------------------------------------------

    def GaussJordan(self, M):
        # Obtain the matrix dimensions
        nf, nc = M.shape

        Maux = M.copy()

        for ii in range (0, 2): # Iteration for the firts nf columns -> transform them to the identity     nf

            for jj in range (0, nf): # Iterate for each row
                for kk in range (0, nc): # Iterate for each column
                    if M[jj][ii] != 0:
                        M[jj][kk] = M[jj][kk]/Maux[jj][ii]
                        
            Maux = M.copy() # Copy the new matrix M in Maux


            for jj in range (0, nf): # Iterate for each row
                for kk in range (0, nc): # Iterate for each column
                    if jj != ii:
                        if Maux[jj][ii] != 0:
                            M[jj][kk] = M[ii][kk]-M[jj][kk]

            Maux = M.copy() # Copy the new matrix M in Maux

        return M

    # ----------------------------------------------------------------------------------------------------------------
    

        






    
    
# -----------------Test the program--------------------------
#      
loop1 = Loop()
C = Capacitor(value = 1, unit='GHz')
L = Inductor(value = 1, unit = 'GHz',loops=[loop1])  

#Define the circuit
elements1 = [(0,1,C), (1,2,L), (2,3,L), (3,0,L)]
elements2 = [(0,1,C), (0,1,L), (1,2,L), (2,3,L), (3,0,L)]
elements3 = [(0,1,C), (1,2,L), (2,3,L), (3,0,L), (0,1,L),]

cr = Circuit(elements1)

print(cr.K)






