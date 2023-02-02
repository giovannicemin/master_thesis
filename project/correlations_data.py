#!/usr/bin/env python
'''File for the data generation of data to
    study the correlations
'''
import sys
import h5py

from models import SpinChain

S = [1, 2, 3, 4, 5, 6, 7, 8, 9]
W = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
SW = W + S

prms = {'L' : [20],              # length of spin chain
        'sites' : [0, 1],        # sites of the subsystem S spins
        'omega' : 1,             # Rabi frequency
        # inverse temperature
        'beta' : [1],
        # interaction of subsystem's S spins
        'potential' : W,
        'potential_' : None,     # interaction of bath spins, if None same as potential
        'T' : 10,                # total time for the evolution
        'dt' : 0.01,             # interval every which save the data
        'cutoff' : 1e-5,         # cutoff for TEBD algorithm
        'im_cutoff' : 1e-10,      # cutoff for TEBD algorithm, img t-e
        'tolerance' : 1e-3,      # Trotter tolerance for TEBD algorithm
        'verbose' : True,        # verbosity of the script
        'num_traj' : 1,          # how many trajectories to do
        # file to save the data
        'fname' : './data/data_correlations.hdf5'
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
    # printing general info
    print('   -----------------------')
    for key in prms.keys():
        print(f'   |{key} : {prms[key]}')
    print('   -----------------------\n')

    # system parameters
    sys_prms = prms.copy()
    sys_prms.pop('sites')
    sys_prms.pop('potential_')
    sys_prms.pop('num_traj')
    sys_prms.pop('fname')
    sys_prms['verbose'] = False
    sys_prms['beta'] = 1


    ### ACTUAL GENERATION
    n_simulations = len(prms['potential']) * len(prms['L'])
    count = 1
    for vv in prms['potential']:
        for ll in prms['L']:

            sys_prms['potential'] = vv
            sys_prms['L'] = ll

            print(f'==== {count}/{n_simulations}')
            print(f'== L = {ll}, potential = {vv}')

            # evolution of the spin chain
            system = SpinChain(**sys_prms)
            system.thermalize()
            cx, cy, cz = system.calculate_correlations(site=0)

            # save to file
            file = h5py.File(prms['fname'], 'a')

            # group name in hdf5 file
            gname = 'correlations_L_' + str(ll) + \
                '_V_' + str(int(vv*1e3)).zfill(4) + \
                '_T_' + str(int(sys_prms['T'])).zfill(2)
            # create the subgroup
            subg = file.create_group(gname)

            subg.create_dataset('x', data=cx)
            subg.create_dataset('y', data=cy)
            subg.create_dataset('z', data=cz)

            file.close()

            count += 1

def execute_trajectories(sys_prms, seed):
    '''Little function needed for the
    parallelization
    '''
    # evolution of the spin chain
    system = SpinChain(**sys_prms)
    system.thermalize()
    system.evolve(seed)

    return system.return_results()

if __name__ == '__main__':
    argv = sys.argv
    generate_data(prms, argv)
