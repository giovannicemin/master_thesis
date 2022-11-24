import numpy as np
import torch
from torch import nn
from itertools import product
import opt_einsum as oe
import getopt
import sys

def get_arch_from_layer_list(input_dim, output_dim, layers):
    ''' Function returning the NN architecture from layer list
    '''
    layers_module_list = nn.ModuleList([])
    # layers represent the structure of the NN
    layers = [input_dim] + layers + [output_dim]
    for i in range(len(layers)-1):
        layers_module_list.append(nn.Linear(layers[i], layers[i+1]))
    return layers_module_list

def pauli_s_const_test():
    '''Function returning the structure constants
    for 2 spin algebra
    '''
    s_x = np.array([[ 0,  1 ], [ 1,  0 ]], dtype=np.complex64)
    s_z = np.array([[ 1,  0 ], [ 0, -1 ]], dtype=np.complex64)
    s_y = np.array([[ 0, -1j], [ 1j, 0 ]], dtype=np.complex64)
    Id  = np.eye(2)
    pauli_dict = {
       'X' : s_x,
       'Y' : s_y,
       'Z' : s_z,
       'I' : Id
    }

    # creating the elements of the base
    base_F = []
    for i, j in product(['I', 'X', 'Y', 'Z'], repeat=2):
        base_F.append( np.kron(pauli_dict[i], pauli_dict[j]))

    base_F.pop(0) # don't want the identity
    abc = oe.contract('aij,bjk,cki->abc', base_F, base_F, base_F )
    acb = oe.contract('aij,bki,cjk->abc', base_F, base_F, base_F )

    f = np.real( 1j*0.25*(abc-acb))
    d = np.real(0.25*(abc+acb))

    # return as a torch tensor
    return torch.from_numpy(f).float(), torch.from_numpy(d).float()


def get_params_from_cmdline(argv, default_params=None):
    '''Function that parse command line argments to dicitonary
    Parameters
    ----------
    argv : argv from sys.argv
    default_params : dict
        Dicionary to update

    Return
    ------
    Updated dictionary as in input
    '''
    arg_help = '{0} -L <length of spin chain> -b <beta> -V <potential> -w <working dir>'

    if default_params == None:
        raise Exception('Missing default parameters')

    try:
        opts, args = getopt.getopt(argv[1:], 'hL:b:V:w:', ['help', 'length', 'beta=', 'potential=', 'working_dir='])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
        elif opt in ('-L', '--length'):
            default_params['L'] = arg
        elif opt in ('-b', '--beta'):
            default_params['beta'] = arg
        elif opt in ('-p', '--potential'):
            default_params['potential'] = arg
        elif opt in ('-w', '--working_dir'):
            default_params['working_dir'] = arg

    return default_params
