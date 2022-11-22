import numpy as np
#import matplotlib.pyplot as plt
import time
import sys

import quimb as qu
# import quimb.tensor as qtn
# import quimb.linalg.base_linalg as la
# from itertools import product
#
from utils import get_params_from_cmdline


default_params = {'L' : 10,             # length of spin chain
                  'omega' : 1,          # Rabi frequency
                  'beta' : 0.01,        # inverse temperature
                  'potential' : 0.1,    # interaction of subsystem's S spins
                  'potential_' : 0.1,   # interaction of bath spins
                  'T' : 10,             # total time for the evolution
                  'cutoff' : 1e-8,      # cutoff for TEBD algorithm
                  'tolerance' : 1e-3,   # Trotter tolerance for TEBD algorithm
                  'verbose' : True,     # verbosity of the script
                  'wd' : './data/data1'    # current working directory
                  }

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
        Interaction strength of the subsystem's spins
    potential_ : float
        Interaction of the bath spins
    T : float
        Total time of the evolution
    cutoff : float
        Cutoff for the TEBD algorithm
    tolerance : float
        Trotter tolerance for TEBD algorithm
    wd : path
        Path to the working directory to save data
    '''

    def __init__(self, L, omega=1, beta=0.01, potential=0.1, potential_=0.1,
                 T=10, cutoff=1e-10, tolerance=1e-3, verbose=True, wd=None):
        # setting th parameters
        self.L = L
        self.beta = beta
        self.T = T
        self.cutoff = cutoff
        self.tolerance = tolerance
        self.wd = wd
        self._verbose = verbose

        # create the MPS of the spin chain
        verboseprint('Building the spin chain MPS: \n')
        B = np.array([1, 0, 0, 1])/np.sqrt(2)
        arrays = [B for i in range(L)]
        self.psi = qtn.MPS_product_state(arrays, cyclic=True)#, site_ind_id='s{}')
        if self._verbose:
            self.psi.show()
            print('\n')

        # build the Hamiltonian of the system
        verboseprint('Building the Hamiltonian of the system \n')

        dims = [2]*L # overall space of L qbits

        I = qu.pauli('I')
        X = qu.pauli('Y')
        Z = qu.pauli('Z')

        O_Rabi = (omega/2)*X & I
        N = (I + Z)/2 & I

        # the hamiltonian
        H1 = {i: O_Rabi for i in range(L)}
        H2 = {None: potential_*N&N,
              (L-1, 0): potential*N&N,
              (0, 1): potential*N&N,
              (1, 2): potential*N&N}

        self.H = qtn.LocalHam1D(L=L, H2=H2, H1=H1, cyclic=True)

    def thermalize(self):
        '''
        Perform imaginary time evolution so to
        obtain the thermal state rho = e^(-beta*H)
        '''

        verboseprint('Imaginary time evolution \n')

        # create the object
        tebd = qtn.TEBD(self.psi, self.H, imag=True)

        # cutoff for truncating after each infinitesimal-time operator application
        tebd_th.split_opts['cutoff'] = 1e-12

        tebd_th.update_to(self.beta/2, tol=1e-6)
        self.psi_th = tebd_th.pt

        if self._verbose:
            psi_th.show()
            print('\n')

    def evolve(self):
        '''Perform time evolution of the SystemError
        '''

        verboseprint('Performing the time evolution \n')

        # initial coditions ootained by means of random unitary
        Rand1 = qu.gen.rand.rand_uni(2) & qu.pauli('I')
        Rand2 = qu.gen.rand.rand_uni(2) & qu.pauli('I')

        psi_init = self.psi_th.gate(Rand1&Rand2, (0,1), contract='swap+split')

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

        keys = results_tebd.keys()

        for psit in tebd.at_times(t, tol=self.tolerance):
            for key in keys:
                ob1 = qu.pauli(key[0]) & qu.pauli('I')
                ob2 = qu.pauli(key[2]) & qu.pauli('I')
                self.results[key].append((psit.H @ psit.gate(ob1 & ob2, (0, 1))).real)

        end = time.time()
        verboseprint(f'It took:{int(end - start)}s')

if __name__ == '__main__':

    # check if custom parameters are given
    argv = sys.argv
    if len(argv) == 1:
        print('Working with default params \n')
        prms = default_params
    else:
        prms = get_params_from_cmdline(argv, default_params)
        print('Problem parameters')
    print('   -----------------------')
    for key in default_params.keys():
        print(f'   |{key} : {default_params[key]}')
    print('   -----------------------\n')

    # creating verboseprint
    verboseprint = print if prms['verbose'] else lambda *a, **k: None

    # evolution of the spin chain
    system = SpinChain(**prms)
    system.thermalize()
    system.evolve()
