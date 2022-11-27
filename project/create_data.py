#!/usr/bin/env python
'''File for the data generation
'''
import pandas as pd
import sys
import os.path

from models import SpinChain
from utils import get_params_from_cmdline

prms = {'L' : 40,                # length of spin chain
        'sites' : [0, 1],        # sites of the subsystem S spins
        'omega' : 1,             # Rabi frequency
        # inverse temperature
        'beta' : [0.001, 0.005, 0.01, 0.05, 0.1],
        # interaction of subsystem's S spins
        'potential' : [0.1, 0.2, 0.3, 0.4, 0.5],
        'potential_' : None,     # interaction of bath spins, if None same as potential
        'T' : 10,                # total time for the evolution
        'dt' : 0.02,             # interval every which save the data
        'cutoff' : 1e-5,         # cutoff for TEBD algorithm
        'im_cutoff' : 1e-7,      # cutoff for TEBD algorithm, img t-e
        'tolerance' : 1e-5,      # Trotter tolerance for TEBD algorithm
        'verbose' : True,        # verbosity of the script
        'num_traj' : 20,         # how many trajectories to do
        # file to save the data
        'fname' : './data/data_tebd.hdf5'
        }

def generate_data(default_params, argv=[1]):
    '''Here I generate the data for the spin chain
    as in models.SpinChain

    Parameter
    --------
    argv : array
        Output from sys.argv, custom parameters
        default [1]
    default_params : dict
        Contains all the parameters necessary for the generation, e.g.
        {'L' : 10,                # length of spin chain
         'sites' : [0, 1],        # sites of the subsystem S spins
         'omega' : 1,             # Rabi frequency
          # inverse temperature
         'beta' : [0.001, 0.005, 0.01, 0.05, 0.1],
          # interaction of subsystem's S spins
         'potential' : [0.1, 0.2, 0.3, 0.4, 0.5],
         'potential_' : None,     # interaction of bath spins, if None same as potential
         'T' : 10,                # total time for the evolution
         'dt' : 0.02,             # interval every which save the data
         'cutoff' : 1e-5,         # cutoff for TEBD algorithm
         'im_cutoff' : 1e-7,      # cutoff for TEBD algorithm, im t-e
         'tolerance' : 1e-5,      # Trotter tolerance for TEBD algorithm
         'verbose' : True,        # verbosity of the script
         'num_traj' : 20,         # how many trajectories to do
         # file to save the data
         'fname' : './data/data_tebd.hdf5'
         }
    '''
    # check if custom parameters are given
    if len(argv) == 1:
        print('Working with default params \n')
        prms = default_params.copy()
    else:
        prms = get_params_from_cmdline(argv, default_params)
        print('Working with custom parameters')

    print('   -----------------------')
    for key in prms.keys():
        print(f'   |{key} : {prms[key]}')
    print('   -----------------------\n')

    # check if the file alreay exists
    if os.path.isfile(prms['fname']):
        i = 1
        fname = prms['fname'][:-5] + '_' + str(i) + '.hdf5'
        while(os.path.isfile(fname)):
            i += 1
            fname = prms['fname'][:-5] + '_' + str(i) + '.hdf5'
        prms['fname'] = fname
        print(f'File already exists, saving as {fname}')
    # saving the txt file with psecifications
    with open(prms['fname'][:-5]+'.txt', 'w') as f:
        print(prms, file=f)

    # system parameters
    sys_prms = prms.copy()
    sys_prms.pop('sites')
    sys_prms.pop('potential_')
    sys_prms.pop('num_traj')
    sys_prms.pop('fname')
    sys_prms['verbose'] = False

    ### ACTUAL GENERATION
    store = pd.HDFStore(prms['fname'])

    n_simulations = len(prms['potential']) * len(prms['beta'])
    count = 1
    for vv in prms['potential']:
        for beta in prms['beta']:

            sys_prms['potential'] = vv
            sys_prms['beta'] = beta

            # group name in hdf5 file
            gname = 'cohVec_L_' + str(prms['L']) + \
                '_V_' + str(int(vv*1e3)).zfill(4) + \
                '_beta_' + str(int(beta*1e3)).zfill(4) + \
                '_dt_' + str(int(prms['dt']*1e3)).zfill(4)

            for i in range(prms['num_traj']):
                print(f'===== {count}/{n_simulations}, trajectory: {i}')
                print(f'== beta = {beta}, potential = {vv}')

                # evolution of the spin chain
                system = SpinChain(**sys_prms)
                system.thermalize()
                system.evolve()

                store.append(gname, system.return_results())
            count += 1
    store.close()

if __name__ == '__main__':
    argv = sys.argv
    generate_data(prms, argv)
