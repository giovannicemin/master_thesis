'''Where all the classes used for the ML part of the project
are stored.
'''
import numpy as np
import pandas as pd
import torch
import math
from torch import nn
import opt_einsum as oe
import h5py
from torch.nn.modules import normalization
from torch.utils.data.dataset import Dataset

from ml.utils import pauli_s_const, get_arch_from_layer_list

class CustomDatasetFromHDF5(Dataset):
    '''Class implementing the Dataset object, for
    the data, to the pass to DataLoader. It directly takes
    the data from the hdf5 file
    NOTE: here I also normalize the data respect to beta!

    Parameters
    ----------
    path : str
        Path to where hdf5 file is
    group : array of str
        Group name or names of the desired data
    resize : bool
        To resize or not the dataset based on T_train
    T_train : int
        Max time to reach
    dt : float
        Time increment used in data generation
    num_traj : int
        Number of trajectories
    '''

    def __init__(self, path, group, T_train, dt, num_traj, resize=False):
        with h5py.File(path, 'r') as f:
            self.X = []
            self.y = []
            self.V = []

            for g in group:
                # extract beta from the name
                beta = int(g[24:28])*1e-3
                normalization = 1 - math.e**(-beta/2)

                if resize:
                    data_short_X = []
                    data_short_y = []

                    # calculate how much to include for each traj
                    tt = int(T_train/dt)

                    for n in range(num_traj):
                        data_short_X.extend(f[g + '/X'][n*999:n*999+tt])
                        data_short_y.extend(f[g + '/y'][n*999:n*999+tt])

                    # normalizing and appending
                    self.X.extend([x/normalization for x in data_short_X])
                    self.y.extend([y/normalization for y in data_short_y])

                else:
                    self.X.extend(f[g + '/X'][()] / normalization)
                    self.y.extend(f[g + '/y'][()] / normalization)

                print(len(self.X))
                # I have to extract the potential from the name
                # and add a vector of the same length
                self.V.extend([int(g[14:18])*1e-3]*len(f[g + '/X'][()]))

    def __getitem__(self, index):
        # return the potential and the vector at t and t+dt
        # as tensors
        return torch.tensor(self.V[index]), \
            torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)

class MLLP(nn.Module):
    '''Machine learning model to parametrize the Lindbladian operator

    Parametes
    ---------
    mlp_params : dict
        Dictionary containing all parameters needed to exp_LL
    potential : float
        Potential appearing in the H
    '''

    def __init__(self, mlp_params):
        '''Init function
        Here i need the temperature to normalize data accordingly
        '''
        super().__init__()
        self.MLP = exp_LL(**mlp_params)  # multi(=1) layer perceptron
        #self.MLP = exp_LL_custom_V(**mlp_params)

        self.dt = mlp_params['dt']

    def forward(self, x):
        '''Forward step of the model
        '''
        return self.MLP.forward(x)

    def generate_trajectory(self, v_0, T, beta):
        '''Function that generates the time evolution of
        the system, namely the trajectory of v(t) coherence
        vector

        Parameters
        ----------
        v_0 : array
            Initial conditions
        T : int
            Total time of the simulation
        beta : float
            Inverse temperature of the initial condition

        Return
        ------
        vector of vectors representing the v(t) at each
        instant of time.
        '''
        normalization = 1 - math.e**(-beta/2)
        results = [v_0]

        X = torch.tensor(v_0/normalization, dtype=torch.double)

        length = int(T/self.dt)

        for i in range(length-1):
            with torch.no_grad():
                y = self.forward(X.float())
                X = y.clone()
                results.extend([y.numpy()*normalization])

        return results

    def trace_loss(self, x, recon_x):
        '''Function
        '''
        paulis = torch.tensor([[[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,1.,0.],
                                [0.,0.,0.,1.]],
                               [[0.,0.,1.,0.],
                                [0.,0.,0.,1.],
                                [1.,0.,0.,0.],
                                [0.,1.,0.,0.]],
                               [[0.,0.,0.,1.],
                                [0.,0.,-1.,0.],
                                [0.,-1.,0.,0.],
                                [1.,0.,0.,0.]],
                               [[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,-1.,0.],
                                [0.,0.,0.,-1.]] ])
        batch_size = len(x)
        p = recon_x-x
        #x = x.reshape(batch_size,4,4)
        #recon_x = recon_x.reshape(batch_size,4,4)
        s_mn = torch.zeros(4,4,16,16)
        for m in range(4):
            for n in range(4):
                s_mn[m,n] = torch.kron(paulis[m],paulis[n])
        s_mn = s_mn.reshape(16,16,16)[1:]
        pp = oe.contract('bx,xkl->bkl', p, s_mn)
        #sigma,U = torch.symeig(pp , eigenvectors = True )
        #sigma   = (torch.abs(sigma)+sigma)/2.
        #sigma  /= sigma.sum()
        #rt = oe.contract('bik,bk,bkl->bil',U,sigma,torch.transpose(U,1,2))
        e,_ = torch.symeig(pp , eigenvectors = True )
        loss = torch.sum(torch.abs(e),1)
        return torch.mean(loss)


class exp_LL(nn.Module):
    ''' Custom Liouvillian layer to ensure positivity of the rho

    Parameters
    ----------
    data_dim : int
        Dimension of the input data
    layers : arr
        Array containing for each layer the number of neurons
        (can be empty)
    nonlin : str
        Activation function of the layer(s)
    output_nonlin : str
        Activation function of the output layer
    dt : float
        Time step for the input data
    '''

    def __init__(self, data_dim, layers, nonlin, output_nonlin, dt):
        super().__init__()
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.data_dim = data_dim
        self.dt = dt

        # I want to build a single layer NN
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        # ?
        self.n = int(np.sqrt(data_dim+1))
        # structure constants
        self.f, self.d = pauli_s_const()

        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        # (v is Z on the notes)
        v_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        v_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.v_x = nn.Parameter(v_re)
        self.v_y = nn.Parameter(v_im)

        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim])
        self.omega = nn.Parameter(omega).float()

        # initialize omega and v
        nn.init.kaiming_uniform_(self.v_x, a=1)
        nn.init.kaiming_uniform_(self.v_y, a=1)
        # rescaling to avoid too big initial values
        self.v_x.data = 0.001*self.v_x.data
        self.v_y.data = 0.001*self.v_y.data
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init

    def forward(self, x):
        # second argument would be potential but here not needed

        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        c_re = torch.add(torch.einsum('ki,kj->ij', self.v_x, self.v_x),\
                         torch.einsum('ki,kj->ij', self.v_y, self.v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', self.v_x, self.v_y),\
                         -torch.einsum('ki,kj->ij', self.v_y, self.v_x) )

        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics.
        # Einsum not optimized in torch: https://optimized-einsum.readthedocs.io/en/stable/

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
        im_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
        im_2 =  4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = -4.*torch.einsum('imj,ij ->m', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, self.omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

class exp_LL_custom_V(nn.Module):
    ''' Custom Liouvillian layer to ensure positivity of the rho.
    The fact the potential explicitly appear (should) make the
    model independant of the potential.

    Parameters
    ----------
    data_dim : int
        Dimension of the input data
    layers : arr
        Array containing for each layer the number of neurons
        (can be empty)
    nonlin : str
        Activation function of the layer(s)
    output_nonlin : str
        Activation function of the output layer
    dt : float
        Time step for the input data
    '''

    def __init__(self, data_dim, layers, nonlin, output_nonlin, dt):
        super().__init__()
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.data_dim = data_dim
        self.dt = dt

        # I want to build a single layer NN
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        # ?
        self.n = int(np.sqrt(data_dim+1))
        # structure constants
        self.f, self.d = pauli_s_const()

        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        # (v is Z on the notes)
        v_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        v_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.v_x = nn.Parameter(v_re)
        self.v_y = nn.Parameter(v_im)

        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim])
        self.omega = nn.Parameter(omega).float()
        self.omega_int = nn.Parameter(omega).float()

        # initialize omega and v
        nn.init.kaiming_uniform_(self.v_x, a=1)
        nn.init.kaiming_uniform_(self.v_y, a=1)
        # rescaling to avoid too big initial values
        self.v_x.data = 0.001*self.v_x.data
        self.v_y.data = 0.001*self.v_y.data
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init
        nn.init.uniform_(self.omega_int, -bound, bound)  # bias init

    def forward(self, x, potential):
        # at input also the potential appearing in the Hamiltonian
        #
        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        c_re = torch.add(torch.einsum('ki,kj->ij', self.v_x, self.v_x),\
                         torch.einsum('ki,kj->ij', self.v_y, self.v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', self.v_x, self.v_y),\
                         -torch.einsum('ki,kj->ij', self.v_y, self.v_x) )

        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics.
        # Einsum not optimized in torch: https://optimized-einsum.readthedocs.io/en/stable/

        # dummy index
        dummy = torch.ones(len(potential))

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
        im_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
        im_2 =  4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = -4.*torch.einsum('imj,ij ->m', self.f, c_im )

        # dissipative part
        dissipative = torch.zeros(self.data_dim+1, self.data_dim+1)
        dissipative[1:, 1:] = d_super_x
        dissipative[1:, 0] = tr_id

        # hamiltonian part
        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, self.omega)
        h_commutator_x_int = 4.* torch.einsum('ijk,k->ji', self.f, self.omega_int)

        hamiltonian = torch.zeros(self.data_dim+1, self.data_dim+1)
        hamiltonian_int = torch.zeros(self.data_dim+1, self.data_dim+1)
        hamiltonian[1:, 1:] = h_commutator_x
        hamiltonian_int[1:, 1:] = h_commutator_x_int

        # adding the potential index
        # L   = id  H   + V  H  + (V^2)  D
        #  ijk    i  jk    i  jk       i  jk
        hamiltonian = torch.einsum('i,jk -> ijk', dummy, hamiltonian)
        hamiltonian_int = torch.einsum('i,jk -> ijk', 100*potential, hamiltonian_int)
        dissipative = torch.einsum('i,jk -> ijk', potential**2, dissipative)

        # building the Lindbladian operator
        L = torch.add(hamiltonian, hamiltonian_int)
        L = torch.add(L, dissipative)

        exp_dt_L = torch.matrix_exp(self.dt*L )

        # x must be 2D
        if len(x.shape) == 1:
            x.resize_(1, 15)
        # have to add identity to x
        x = torch.cat((torch.ones(x.shape[0], 1), x), 1)

        # calculating the result
        y = torch.einsum('ikj,ij -> ik', exp_dt_L, x)

        return y[:, 1:]
