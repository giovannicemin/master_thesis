#!/usr/bin/env python
'''File for the data generation
'''
import sys

from models import SpinChain
from utils import get_params_from_cmdline

def generate_data(argv):
    '''Here I generate the data for the spin chain
    as in models.SpinChain

    Parameter
    --------
    argv : array
        Output from sys.argv, custom parameters
    '''
    default_params = {'L' : 10,                # length of spin chain
                      'sites' : [0, 1],        # sites of the subsystem S spins
                      'omega' : 1,             # Rabi frequency
                      # inverse temperature
                      'beta' : [0.001, 0.005, 0.01, 0.05, 0.1],
                      # interaction of subsystem's S spins
                      'potential' : [0.1, 0.2, 0.3, 0.4, 0.5],
                      'potential_' : None,     # interaction of bath spins, if None same as potential
                      'T' : 10,                # total time for the evolution
                      'dt' : 0.1,              # interval every which save the data
                      'cutoff' : 1e-8,         # cutoff for TEBD algorithm
                      'tolerance' : 1e-3,      # Trotter tolerance for TEBD algorithm
                      'verbose' : True,        # verbosity of the script
                      # file to save the data
                      'fname' : './data/data_tebd.hdf5'
                      }

    # check if custom parameters are given
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

    sys_prms = prms
    sys_prms.pop('sites')
    sys_prms.pop('potential_')

    ### ACTUAL GENERATION

    for vv in prms['potential']:
        for beta in prms['beta']:

            sys_prms['potential'] = vv
            sys_prms['beta'] = beta

            # evolution of the spin chain
            system = SpinChain(**sys_prms)
            system.thermalize()
            system.evolve()
            system.save_results()

if __name__ == '__main__':
    argv = sys.argv

    generate_data(argv)
