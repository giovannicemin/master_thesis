import numpy as np
import torch
import math
from torch import nn
import os
from itertools import product
import opt_einsum as oe
import getopt
import sys
from torch.utils.data.sampler import SubsetRandomSampler

# def custom_pruning_unstructured(module, name, threshold):
#     """Prunes tensor corresponding to parameter called `name` in `module`
#     by setting to zero if < threshold
#     Modifies module in place (and also return the modified module)
#     by:
#     1) adding a named buffer called `name+'_mask'` corresponding to the
#     binary mask applied to the parameter `name` by the pruning method.
#     The parameter `name` is replaced by its pruned version, while the
#     original (unpruned) parameter is stored in a new parameter named
#     `name+'_orig'`.

#     Args:
#         module (nn.Module): module containing the tensor to prune
#         name (string): parameter name within `module` on which pruning
#                 will act.

#     Returns:
#         module (nn.Module): modified (i.e. pruned) version of the input
#             module

#     Examples:
#         >>> m = nn.Linear(3, 4)
#         >>> custum_pruning_unstructured(m, name='bias')
#     """
#     pruning = ThresholdPruning()
#     print('ciao')
#     pruning.apply(module, name)

#     return module

# class ThresholdPruning(prune.BasePruningMethod):
#     PRUNING_TYPE = "unstructured"

#     def __init__(self):
#         self.threshold = 1e-2

#     def compute_mask(self, tensor, default_mask):
#         mask = torch.abs(tensor[:, :]) < self.threshold
#         default_mask[mask] = 0
#         return default_mask
#         #return torch.abs(tensor) > self.threshold

class FourierLayer(nn.Module):
    """ Custom Layer to construct time function starting from
    the Fourier transform

    Parameters
    ----------
    frequencies : array float
        Init values for the frequencies
    out_dim : int
        Dimention of the output
    threshold : float
        Threshold for pruning
    """
    def __init__(self, frequencies, out_dim:int, threshold:float,
                 impose_positivity:bool = False):
        super().__init__()
        N_freq = len(frequencies)
        #frequencies = 0.5*torch.ones(N_freq)
        self.frequencies = nn.Parameter(frequencies, requires_grad=True)

        weights = torch.ones((out_dim, 2*(N_freq)))
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(out_dim)
        self.bias = nn.Parameter(bias)

        self.out_dim = out_dim
        self.threshold = threshold
        self.pos = impose_positivity

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=15) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, t):
        F = torch.cat((torch.cos(t*self.frequencies),\
                       torch.sin(t*self.frequencies)), dim=-1)

        # w -> index for the omega vector
        # b -> batch index
        # n -> Fourier decomposition element
        if len(F.shape) == 2:
            w = torch.einsum('wn,bn->bw', self.weights, F)
        else:
            w = torch.einsum('wn,n->w', self.weights, F)

        if self.pos:
            return torch.abs(torch.add(w, self.bias))
        else:
            return torch.add(w, self.bias)


class Snake(nn.Module):
    def __init__(self, a = 0.5):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor([a]))

    def forward(self, x):
        return 0 - (0.5/self.a)*torch.cos(2.*self.a*x) + 0.5/self.a

@torch.no_grad()
def init_weights(m):
    '''Function to initialize the weights of NN
    '''
    if isinstance(m, nn.Linear):
        #d = m.weight.shape[0]
        #nn.init.uniform_(m.weight, -d, d)
        nn.init.kaiming_uniform_(m.weight, a=5)
        #m.weight = 0.1*m.weight
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #bound = 1 / np.sqrt(fan_in)
        #nn.init.uniform_(m.bias, -bound, bound)

        #nn.init.normal_(m.weight, 0, 0.05)
        #nn.init.constant_(m.bias, 0)
        #nn.init.uniform_(m.bias, -0.01, 0.01)


def calculate_error(results_ml, results_tebd, T, dt):
    '''Function to calculate the error defined as
    the normalized norm squared of the difference
    of the two coherence vectors averaged over time

    Perameters
    ----------
    results_ml : array
        Vector of vectors containing the dynamics
        predicted by the model
    results_tebd : array
        Vector of vectors containing the dynamics
        calculated using TEBD
    T : int
        Total time of the dynamics
    dt : float
        Time increase

    Return
    ------
        Return the error
    '''
    integral = 0

    # to do thigs rigth first and last element shoul be *1/2
    for v_ml, v_tebd in zip(results_ml, results_tebd):
        integral += (np.linalg.norm(v_ml - v_tebd) / np.linalg.norm(v_tebd) )**2

    return integral * (dt/T)

def get_arch_from_layer_list(input_dim, output_dim, layers):
    ''' Function returning the NN architecture from layer list
    '''
    layers_module_list = nn.ModuleList([])
    # layers represent the structure of the NN
    layers = [input_dim] + layers + [output_dim]
    for i in range(len(layers)-1):
        layers_module_list.append(nn.Linear(layers[i], layers[i+1]))
    return layers_module_list

def pauli_s_const():
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

    f = np.real( 1j*0.5*(abc - acb) )
    d = np.real( 0.5*(abc + acb) )

    # return as a torch tensor
    return torch.from_numpy(f).float(), torch.from_numpy(d).float()

def ensure_empty_dir(directory):
    if len(os.listdir(directory)) != 0:
        raise Exception('Model dir not empty!')

def load_data(path, L, beta, potential, dt, T_train,
              num_traj, batch_size, validation_split, resize=False):
    '''Function to load the NORMALIZED data from hdf5 file.
    Reshuffling of data is performed. Then separates train
    from validation and return the iterables.
    NOTE: this functions takes both beta and potential as arrays
    in case one wants to have wider training sets.

    Parameters
    ----------
    path : str
        Path to the hdf5 file
    beta : array
    potential : float
        Array of betas an potentials for the group name
    resize : bool
        To either resize or not based on T
    T_train : int
        Time used in the training procedure
    num_traj : int
    batch_size : int
    validation_split : float
        Number 0 < .. < 1 which indicates the relative
        sizes of validation and train

    Return
    ------
    train and validation loaders
    '''
    # put import here to avoid circular imports
    from ml.classes import CustomDatasetFromHDF5

    # list of group names
    gname = ['cohVec_L_' + str(L) + \
        '_V_' + str(int(p*1e3)).zfill(4) + \
        '_beta_' + str(int(b*1e3)).zfill(4) + \
        '_dt_' + str(int(dt*1e3)).zfill(4) for b in beta for p in potential]

    dataset = CustomDatasetFromHDF5(path, gname, T_train, dt, num_traj, resize)

    # creating the indeces for training and validation split
    dataset_size = len(dataset)
    indeces = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # shuffling the datesets
    np.random.seed(42)
    np.random.shuffle(indeces)
    train_indices, val_indices = indeces[split:], indeces[:split]

    # Creating PT data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=valid_sampler)
    return train_loader, val_loader


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
