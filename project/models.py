#!/usr/bin/env python
'''File containing the implementation of the models I am
    going to use
'''
import numpy as np
import pandas as pd
import time
import pickle
import h5py

import quimb as qu
import quimb.tensor as qtn
from itertools import product

#import warnings
#from tables import NaturalNameWarning
# avoid name warning to save files
#warnings.filterwarnings('ignore', category=NaturalNameWarning)


class SpinChain:
    '''Class implementing the spin chain with PBC

    Parameters
    ----------
    L : int
        Length of the spin chain
    omega : float
        Rabi frequency
    beta : float
        Inverse temperature
    potential : float
        Interaction strength between the spins
    T : float
        Total time of the evolution
    dt : float
        Time step for observable measurement
    cutoff : float
        Cutoff for the TEBD algorithm
    tolerance : float
        Trotter tolerance for TEBD algorithm
    '''

    def __init__(self, L, omega=1, beta=0.01, potential=0.1, T=10,
                 dt=0.1, cutoff=1e-10, tolerance=1e-3, verbose=True):
        # setting th parameters
        self.L = L
        self.beta = beta
        self.vv = potential
        self.t = [i for i in np.arange(0, T, dt)]
        self.cutoff = cutoff
        self.tolerance = tolerance
        self._verbose = verbose

        # creating verboseprint
        self.verboseprint = print if verbose else lambda *a, **k: None

        # create the MPS of the spin chain
        self.verboseprint(f'System for beta = {self.beta}, potential = {self.vv}')
        self.verboseprint('Building the spin chain MPS: \n')
        B = np.array([1, 0, 0, 1])/np.sqrt(2)
        arrays = [B]*L
        self.psi = qtn.MPS_product_state(arrays, cyclic=True)#, site_ind_id='s{}')
        if self._verbose:
            self.psi.show()
            print('\n')

        # build the Hamiltonian of the system
        self.verboseprint('Building the Hamiltonian of the system \n')

        # dims = [2]*L # overall space of L qbits

        I = qu.pauli('I')
        X = qu.pauli('Y')
        Z = qu.pauli('Z')

        O_Rabi = (omega/2)*X & I
        N = (I + Z)/2 & I

        # the hamiltonian
        H1 = {i: O_Rabi for i in range(L)}
        H2 = {None: self.vv*N&N,
              (L-1, 0): self.vv*N&N} # for safety

        self.H = qtn.LocalHam1D(L=L, H2=H2, H1=H1, cyclic=True)

        # results
        self.results = None

    def thermalize(self):
        '''
        Perform imaginary time evolution so to
        obtain the thermal state rho = e^(-beta*H)
        '''

        self.verboseprint('Imaginary time evolution \n')

        # create the object
        tebd = qtn.TEBD(self.psi, self.H, imag=True)

        # cutoff for truncating after each infinitesimal-time operator application
        tebd.split_opts['cutoff'] = 1e-12

        tebd.update_to(self.beta/2, tol=1e-6)
        self.psi_th = tebd.pt

        if self._verbose:
            self.psi_th.show()
            print('\n')

    def evolve(self):
        '''Perform time evolution of the SystemError
        '''

        self.verboseprint('Performing the time evolution \n')

        # initial coditions ootained by means of random unitary
        Rand1 = qu.gen.rand.rand_uni(2) & qu.pauli('I')
        Rand2 = qu.gen.rand.rand_uni(2) & qu.pauli('I')

        self.psi_init = self.psi_th.gate(Rand1&Rand2, (0,1), contract='swap+split')

        start = time.time()

        # first I build the observables and results dictionaries
        observables = {}
        self.results = {}
        for ob1, ob2 in product(['I', 'X', 'Y', 'Z'], repeat=2):
            key = ob1 + '1' + ob2 + '2'
            observables[key] = []
            self.results[key] = []

        # dropping the identity
        observables.pop('I1I2')
        self.results.pop('I1I2')

        # create the object
        tebd = qtn.TEBD(self.psi_init, self.H)

        # cutoff for truncating after each infinitesimal-time operator application
        tebd.split_opts['cutoff'] = self.cutoff

        keys = self.results.keys()

        for psit in tebd.at_times(self.t, tol=self.tolerance):
            for key in keys:
                ob1 = qu.pauli(key[0]) & qu.pauli('I')
                ob2 = qu.pauli(key[2]) & qu.pauli('I')
                self.results[key].append((psit.H @ psit.gate(ob1 & ob2, (0, 1))).real)

        end = time.time()
        self.verboseprint(f'It took:{int(end - start)}s')

    def return_results(self):
        '''Return the results, which are the evolution of
        the coherence vector, as a pandas dataframe
        '''
        if self.results == None:
            raise Exception('The object have not been evolved jet')
        else:
            return pd.DataFrame(data=self.results, dtype=np.float32)
