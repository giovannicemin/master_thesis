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

class FourierLayer(nn.Module):
    """ Custom Layer that outputs the Fourier components.

    Given the number of frequency the model outputs
    A*cos(t*f + phi) + c   first half
    A*sin(t*f + phi) + c   second half

    This should be Fourier decomposition of the functions.

    Parameters
    ----------
    frequencies : array float
        Init values for the frequencies
    require_amplitude : bool
        Whether to set amplitude != 1. Becomes usefull
        in case one is applying the non-linear function.
    require_constant : bool
        Whether to have constant != 0. Becomes usefull
        in case one is ammplying the non-linear function.
    require_phase : bool
        Whether to put the phase, bounded between -pi/2, pi/2
    """
    def __init__(self, frequencies, require_amplitude=False,
                 require_constant=False, require_phase=False,
                 learnable_f=False):
        super().__init__()
        N_freq = len(frequencies)

        self.frequencies = nn.Parameter(frequencies, requires_grad=learnable_f)

        amplitude = torch.ones(2*N_freq)
        if require_amplitude:
            self.amplitude = nn.Parameter(amplitude)
            nn.init.normal_(self.amplitude, 0, 0.1)
        else:
            self.amplitude = amplitude

        const = torch.zeros_like(amplitude)
        if require_constant:
            self.const = nn.Parameter(const)
            nn.init.normal_(self.const, 0, 0.1)
        else:
            self.const = const

        phase = torch.zeros(N_freq)
        if require_phase:
            self.phase = nn.Parameter(phase)
            nn.init.normal_(self.phase, 0, 0.1)
        else:
            self.phase = phase

    def forward(self, t):
        batch_size = t.shape[0]
        # bound the frequencies
        #frequencies = self.frequencies.clamp(1e-2, 20)
        # bound the phases
        phase = self.phase.clamp(-0.5*torch.pi, 0.5*torch.pi)

        t_f_product = torch.einsum('b,f -> bf', t, self.frequencies)
        argument = t_f_product + phase.repeat(batch_size, 1)
        F = torch.cat((torch.cos(argument),\
                       torch.sin(argument)), dim=-1)

        return torch.einsum('bf, f -> bf', F, self.amplitude) + \
            self.const.repeat(batch_size, 1)


class Square(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.square(x)


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
        #nn.init.kaiming_uniform_(m.weight, a=5)
        #m.weight = 0.1*m.weight
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #bound = 1 / np.sqrt(fan_in)
        #nn.init.uniform_(m.bias, -bound, bound)

        nn.init.normal_(m.weight, 0, 0.1)
        nn.init.constant_(m.bias, 0)
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

    # added -, and put 0.25 instead of 0.5
    f = np.real( -1j*0.25*(abc - acb) )
    d = np.real( 0.25*(abc + acb) )

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

class SinusoidalLayer(nn.Module):
    """ Custom Layer implementing sinusoids linear combination.

    This layer take tie as input and output the
    A * cos(t*w + ph) + c
        where w  = frequency
              ph = phase
              A, c = amplitude and constant
    Arguments
    ---------
    T : float
        Maximum time
    m : int
        Number of independent periodic functions.
    """
    def __init__(self, T, m) -> None:
        super().__init__()
        self.m = m

        self.frequency = nn.Parameter(torch.Tensor([1.5]))

        amplitude = torch.ones(m)
        self.amplitude = nn.Parameter(amplitude)

        phase = torch.zeros(m)
        self.phase = nn.Parameter(phase)

        const = torch.zeros(m)
        self.const = nn.Parameter(const)

        # init
        nn.init.uniform_(self.amplitude, 0.1)

    def forward(self, t):
        batch_size = t.shape[0]

        # i -> index for the output vector
        # b -> batch index

        # putting everything in the right dimension
        t_f_product = (t*self.frequency).unsqueeze(1)
        t_f_product = t_f_product.repeat(1, self.m)
        phase = self.phase.repeat(batch_size, 1)

        p_function = torch.cos(t_f_product + phase)

        product = torch.einsum("bi,i -> bi", p_function, self.amplitude)

        const = self.const.repeat(batch_size, 1)

        return product + const


class C_inf_Layer(nn.Module):
    """ Custom layer implementing C^inf periodic functions

    Arguments
    ---------
    T : float
        Maximum time, the minimim frequency is caluclated accordingly
    m : int
        Number of indepdendent periodic functions
    n : int
        Number of nodes = output dimension
    """
    def __init__(self, T, n, m) -> None:
        super().__init__()

        self.layer = nn.Sequential(SinusoidalLayer(T, m),
                                   nn.Tanh(),
                                   nn.Linear(m, n),
                                   nn.Tanh())
    def forward(self, t):
        return self.layer.forward(t)


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

class FourierLayer_backup(nn.Module):
    """ Custom Layer that outputs the Fourier components.

    For each output the model learns the 0 frequency (const) term
    and the real anc complex part for each frequency in input.
    Note the frequencies are learnable parameters!

    Because of the use of periodic functions the learning procedure of this
    model is really delicate, and the risk of ending in a local minima very high.

    Parameters
    ----------
    frequencies : array float
        Init values for the frequencies
    out_dim : int
        Dimention of the output
    threshold : float
        Threshold for pruning
    impose_positivity : bool
        Whether or not to have a >=0 output.
    """
    def __init__(self, frequencies, constant=False, phase=False):
        super().__init__()
        N_freq = len(frequencies)

        # create different set of frequencies for each output
        frequencies = frequencies.repeat(out_dim, 1)
        self.frequencies = nn.Parameter(frequencies, requires_grad=True)

        weight = torch.ones((out_dim, 2*N_freq))
        self.weight = nn.Parameter(weight)
        const = torch.ones_like(weight)
        self.const = nn.Parameter(const)

        self.out_dim = out_dim
        self.threshold = threshold

        # initialize weights
        nn.init.normal_(self.weight, 0, 0.1)
        nn.init.normal_(self.const, 0, 0.1)

    def forward(self, t):
        batch_size = t.shape[0]
        # bound the frequencies
        #frequencies = self.frequencies.clamp(1e-2, 20)

        t_f_product = torch.einsum('b,wf -> bwf', t, self.frequencies)
        F = torch.cat((torch.cos(t_f_product),\
                       torch.sin(t_f_product)), dim=-1)
        # add the linear term
        #linear_term = torch.einsum('w,b->bw', self.linear, t)
        #F = torch.cat((F, linear_term.unsqueeze_(-1)), dim=-1)

        # w -> index for the output vector
        # b -> batch index
        # n -> Fourier decomposition element
        # w = torch.einsum('wn,bwn->bw', self.weight, F)
        w = torch.einsum('wn,bwn->bwn', self.weight, F)

        # I want to constrain the constant term to be positive
        const = self.const.repeat(batch_size, 1, 1)

        # sum ove the Fourier components
        return torch.einsum('bwn->bw', w+const)
