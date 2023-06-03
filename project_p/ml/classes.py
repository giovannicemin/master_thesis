'''Where all the classes used for the ML part of the project
are stored.
'''
import numpy as np
import quimb as qu
import torch
import math
from torch import nn
import opt_einsum as oe
import h5py
#from torch.nn.modules import normalization
from torch.utils.data.dataset import Dataset

from ml.utils import pauli_s_const, get_arch_from_layer_list, \
    init_weights, Snake, FourierLayer, C_inf_Layer, Square

class CustomDatasetFromHDF5(Dataset):
    '''Class implementing the Dataset object from HDF5 file.

    This class loads the data from HDF5 file.

    This class is implemented such that it can be passed to
    torch.utils.data.DataLoader.

    Parameters
    ----------
    path : str
        Path to where hdf5 file is
    group : str
        Group name or names of the desired data
    '''

    def __init__(self, path, group):
        with h5py.File(path, 'r') as f:
            self.X = []
            self.y = []
            self.t = []
            self.V = [] # dummy vector

            self.X.extend(f[group + '/X'][()])
            self.y.extend(f[group + '/y'][()])
            self.t.extend(f[group + '/t'][()])
            self.V.extend([0]*len(f[group + '/X'][()]))

    def __getitem__(self, index):
        # return the potential and the vector at t and t+dt
        # as tensors
        return torch.tensor(self.V[index]), torch.tensor(self.t[index]), \
            torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)


class MLLP(nn.Module):
    '''Machine learning model to parametrize the Lindbladian operator.

    Parametes
    ---------
    mlp_params : dict
        Dictionary containing all parameters needed to exp_LL
    potential : float
        Potential appearing in the H
    time_dependent = bool, default False
        Wheter or not to use the time dependent exp_LL
    '''

    def __init__(self, mlp_params, potential, time_dependent=False):
        super().__init__()
        if time_dependent:
            self.MLP = exp_LL_td_2(**mlp_params)
        else:
            self.MLP = exp_LL(**mlp_params)  # multi(=1) layer perceptron

        self.dt = mlp_params['dt']
        self.potential = potential
        self.td = time_dependent

    def forward(self, **kwargs):
        '''Forward step of the model'''
        return self.MLP.forward_t(**kwargs)

    def generate_trajectory(self, v_0, T):
        '''Function that generates the trajectory v(t).

        Given an initial condition v_0, this function generates the trajectory,
        namely the time evolution v(t), using the learned model.

        Parameters
        ----------
        v_0 : array
            Initial conditions
        T : int
            Total time of the simulation
            
        Return
        ------
        vector of vectors representing the v(t) at each instant of time.
        '''
        X = torch.Tensor(v_0)

        length = int(T/self.dt)

        results = [v_0]
        with torch.no_grad():
            if self.td:
                for i in range(length-1):
                    # if the L is time dep. I have to evolve step by step
                    y = self.MLP.predict(torch.Tensor([i*self.dt]), X.float())
                    X = y.clone()
                    results.extend([y.numpy()])
            else:
                # if the L is const I do in one step
                Lindblad = self.MLP.get_L()
                for i in range(length-1):
                    exp_dt_L = torch.linalg.matrix_exp(i*self.dt*Lindblad )
                    y = torch.add(0.5*exp_dt_L[1:,0], X @ torch.transpose(exp_dt_L[1:,1:],0,1))
                    results.extend([y.numpy()])

        return results


class exp_LL(nn.Module):
    '''Custom Liouvillian layer to ensure positivity of the rho.

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
        # structure constants
        self.f, self.d = pauli_s_const()

        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        # (v is Z on the notes)
        v_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        v_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.v_x = nn.Parameter(v_re)
        self.v_y = nn.Parameter(v_im)
        self.normalization = 1

        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim])
        self.omega = nn.Parameter(omega).float() # torch.Tensor([1,0,0.25,1,0,0,0,0,0,0,0,0.25,0,0,0.25])

        # initialize omega and v
        nn.init.kaiming_uniform_(self.v_x, a=1)
        nn.init.kaiming_uniform_(self.v_y, a=1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init

    def get_L(self):
        ''' Function that returns teh Lindbladian learned.
        '''
        v_x = self.v_x#/self.normalization
        v_y = self.v_y#/self.normalization
        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                         torch.einsum('ki,kj->ij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                         -torch.einsum('ki,kj->ij', v_y, v_x) )

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
        im_1 =  4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
        im_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im)

        tr_id = 2.*torch.einsum('imj,ij ->m', self.f, c_im )

        h_commutator_x = -4.* torch.einsum('ijk,k->ij', self.f, self.omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        return L

    def forward(self, t, x):
        """Forward step of the Layer.
        The step: t -> t+dt

        Time is not used but present from compatibility reasons.
        """
        L = self.get_L()

        exp_dt_L = torch.matrix_exp(self.dt*L ).float()
        return torch.add(0.5*exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

    def predict(self, t, x):
        '''Dummy function for compatibility with exp_LL_td'''
        return self.forward(t, x)

    def forward_t(self, t, x):
        """Forward step of the Layer.
        The step: 0 -> t
        """
        L = self.get_L()

        exp_dt_L = torch.matrix_exp( torch.einsum('b,ij->bij', t, L) ).float()
        return torch.add(0.5*exp_dt_L[:,1:,0],
                         torch.einsum('bi,bji->bj', x, exp_dt_L[:,1:,1:]))
                         #x @ torch.transpose(exp_dt_L[:,1:,1:],1,2))
    def gap(self):
        '''Function to calculate the Lindblad gap, meaning
        the smallest real part of the modulus of the spectrum.
        '''
        L = self.get_L()
        # take the real part of the spectrum
        e_val = np.linalg.eigvals(L.detach().numpy()).real

        e_val.sort()
        return np.abs(e_val[-2])


class exp_LL_td_2(nn.Module):
    ''' Custom Liouvillian **time-depenent** layer.

    In this case the model learns the Fouriere decomposition of omega vector,
    and the rates. The unitary matrix to obtain the Kossakowski matrix is
    learned separately.

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
        self.data_dim = data_dim
        self.dt = dt

        # structure constants
        self.f, self.d = pauli_s_const()

        # Dissipative matrix is learned as u gamma u^T
        # where u is a unitary matrix and gamma a diagonal matrix of
        # time-dependent functions, of which the model learns the Fourier
        # decomposition.

        # the model learn separately the real and complex part of the matrix at
        # the exponent
        u_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        u_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.u_re = nn.Parameter(u_re)
        self.u_im = nn.Parameter(u_im)

        nn.init.kaiming_uniform_(self.u_re, a=10)
        nn.init.kaiming_uniform_(self.u_im, a=10)

        # NOTE: if the frequencies are learnable params it is important to initialize
        # again the tensor, otherwise the model treats them equally (updated in the
        # same manner)
        #frequencies = torch.Tensor([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        frequencies = list(np.arange(0.4, 40, 0.1))
        # frequencies = [i for i in np.arange(0.5, 5, 0.5)]
        # frequencies = [1.5]
        self.gamma_net = nn.Sequential(FourierLayer(frequencies=torch.Tensor(frequencies),
                                                    T=5,
                                                    learnable_f=True,
                                                    require_amplitude=True,
                                                    require_constant=True,
                                                    require_phase=False),
                                       # nn.ELU(),
                                       nn.Linear(2*len(frequencies), self.data_dim),
                                       nn.ReLU())
                                       # Square())
        self.gamma_normalization = 1

        self.omega_net = nn.Sequential(FourierLayer(frequencies=torch.Tensor(frequencies),
                                                    T=5,
                                                    learnable_f=True,
                                                    require_amplitude=True,
                                                    require_constant=True,
                                                    require_phase=False),
                                       # nn.ELU(),
                                       nn.Linear(2*len(frequencies), self.data_dim),
                                       )
        nn.init.uniform_(self.omega_net[0].const, -0.5, 0.5)
        self.omega_normalization = 1

        init_weights(self.omega_net)
        init_weights(self.gamma_net)

        #frequencies = torch.Tensor([i for i in range(11)])
        #frequencies = torch.Tensor([i for i in np.arange(0.5, 40, 0.5)])
        #frequencies = torch.Tensor([0.1, 0.5, 1])
        # self.gamma_net = nn.Sequential(FourierLayer(frequencies=frequencies,
        #                                             out_dim=self.data_dim,
        #                                             threshold=1e-2),
        #                                nn.ReLU())
        # self.gamma_net = nn.Sequential(C_inf_Layer(T=10, n=self.data_dim, m=1),
        #                                nn.ReLU())
        # self.gamma_normalization = 1#100

        # Hamiltonian parameters omega, also this is a vector of time-dependent
        # functions.
        #frequencies = torch.Tensor([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2])
        #frequencies = torch.Tensor([i for i in range(5)])
        #frequencies = torch.Tensor([0.1, 0.5, 1])
        # self.omega_net = FourierLayer(frequencies=frequencies,
        #                               out_dim=self.data_dim, threshold=1e-2)
        # self.omega_net = C_inf_Layer(T=10, n=self.data_dim, m=1)
        # self.omega_normalization = 1#5

    def get_L(self, t):
        """Forward step of the model. """
        batch_size = x.shape[0]

        u_re = self.u_re
        u_im = self.u_im
        gamma = self.gamma_net(t)/self.gamma_normalization
        omega = self.omega_net(t)/self.omega_normalization

        theta = u_re + 1j*u_im + u_re.T - 1j*u_im.T
        u = torch.linalg.matrix_exp(1j*theta)
        c = torch.einsum('ij,sj,jl->sil', u, gamma.type(torch.complex64), u.H)
        c_re = c.real.float()
        c_im = c.imag.float()

        # NOTE: s index is batch index
        # c_re = torch.einsum('ij,sj,jl->sil', u_re, gamma, u_re.T) + \
        #     torch.einsum('ij,sj,jl->sil', u_im, gamma, u_im.T)
        # c_im = torch.einsum('ij,sj,jl->sil', u_im, gamma, u_re.T) - \
        #     torch.einsum('ij,sj,jl->sil', u_re, gamma, u_im.T)

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,sij->smn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,sij->smn', self.f, self.f, c_re )
        im_1 =  4.*torch.einsum('mjk,nik,sij->smn', self.f, self.d, c_im )
        im_2 = -4.*torch.einsum('mik,njk,sij->smn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = 2.*torch.einsum('imj,sij->sm', self.f, c_im )

        h_commutator_x = -4.* torch.einsum('kij,sk->sij', self.f, omega).unsqueeze_(0)

        # building the Lindbladian operator
        L = torch.zeros(batch_size, self.data_dim+1, self.data_dim+1)
        L[:, 1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[:, 1:,0] = tr_id

        return L

    def forward_t(self, t, x):

        t_ = 0
        while t_ < t:
            L = self.get_L(t_)
            exp_dt_L = torch.matrix_exp(self.dt*L )
            #print( torch.einsum('si,sji->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)).shape )
            x = torch.add(0.5*exp_dt_L[:,1:,0], torch.einsum('si,sij->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)))
            t_ += self.dt

        return x

    def get_omega(self, t):
        """Function returning the omega vector. """
        t = torch.Tensor([t])
        return self.omega_net(t).squeeze().detach().numpy()

    def get_rates(self, t):
        """Function returning the gamma vector. """
        t = torch.Tensor([t])
        return self.gamma_net(t).squeeze().detach().numpy()

    def predict(self, t, x):
        """Same function as forward, but works sithout batch dimension. """

        u_re = self.u_re
        u_im = self.u_im
        gamma = self.gamma_net(t).squeeze()/self.gamma_normalization
        omega = self.omega_net(t).squeeze()/self.omega_normalization

        theta = u_re + 1j*u_im + u_re.T - 1j*u_im.T
        u = torch.linalg.matrix_exp(1j*theta)
        c = torch.einsum('ij,j,jl->il', u, gamma.type(torch.complex64), u.H)
        c_re = c.real.float()
        c_im = c.imag.float()

        # NOTE: s index is batch index
        # c_re = torch.einsum('ij,j,jl->il', u_re, gamma, u_re.T) + \
        #     torch.einsum('ij,j,jl->il', u_im, gamma, u_im.T)
        # c_im = torch.einsum('ij,j,jl->il', u_im, gamma, u_re.T) - \
        #     torch.einsum('ij,j,jl->il', u_re, gamma, u_im.T)

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
        im_1 =  4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
        im_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = 4.*torch.einsum('imj,ij->m', self.f, c_im )

        h_commutator_x = -4.* torch.einsum('ijk,k->ij', self.f, omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

    def get_L(self, t):
        '''Function that calculate the Lindbladian. '''
        t = torch.Tensor([t])
        with torch.no_grad():
            u_re = self.u_re
            u_im = self.u_im
            gamma = self.gamma_net(t).squeeze()/self.gamma_normalization
            omega = self.omega_net(t).squeeze()/self.omega_normalization

            #theta = u_re + 1j*u_im + u_re.T - 1j*u_im.T
            #u = torch.linalg.matrix_exp(1j*theta)
            #c = torch.einsum('ij,sj,jl->sil', u, gamma.type(torch.complex64), u.H)
            #c_re = c.real
            #c_im = c.imag

            # NOTE: s index is batch index
            c_re = torch.einsum('ij,j,jl->il', u_re, gamma, u_re.T) + \
                    torch.einsum('ij,j,jl->il', u_im, gamma, u_im.T)
            c_im = torch.einsum('ij,j,jl->il', u_im, gamma, u_re.T) - \
                    torch.einsum('ij,j,jl->il', u_re, gamma, u_im.T)

            # Here I impose the fact c_re is symmetric and c_im antisymmetric
            re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
            re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
            im_1 =  4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
            im_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
            d_super_x_re = torch.add(re_1, re_2 )
            d_super_x_im = torch.add(im_1, im_2 )
            d_super_x = torch.add(d_super_x_re, d_super_x_im )

            tr_id = 4.*torch.einsum('imj,ij->m', self.f, c_im )

            h_commutator_x = -4.* torch.einsum('ijk,k->ij', self.f, omega)

            # building the Lindbladian operator
            L = torch.zeros(self.data_dim+1, self.data_dim+1)
            L[1:,1:] = torch.add(h_commutator_x, d_super_x)
            L[1:,0] = tr_id

            return L

    def gap(self, t):
        '''Function to calculate the Lindblad gap,
        meaning the smallest real part of spectrum in modulus
        '''
        L = self.get_L(t)
        # take the real part of the spectrum
        e_val = np.linalg.eigvals(L.detach().numpy()).real

        e_val.sort()
        return np.abs(e_val[-2])
