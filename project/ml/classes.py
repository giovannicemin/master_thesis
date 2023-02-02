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
#from torch.nn.modules import normalization
from torch.utils.data.dataset import Dataset

from ml.utils import pauli_s_const, get_arch_from_layer_list, init_weights

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
    T_train : int
        Max time to reach
    dt : float
        Time increment used in data generation
    num_traj : int
        Number of trajectories
    resize : bool
        To resize the dataset according to T_train
    '''

    def __init__(self, path, group, T_train, dt, num_traj, resize=False):
        with h5py.File(path, 'r') as f:
            self.X = []
            self.y = []
            self.t = []
            self.V = []

            for g in group:
                # extract beta from the name
                beta = int(g[24:28])*1e-3
                potential = int(g[14:18])*1e-3
                normalization = 1 - math.e**(-beta/2)
                if potential > 1:
                    # divide the normalization to multiply the data
                    normalization /= potential**2

                if resize:
                    data_short_X = []
                    data_short_y = []

                    # have to calculate how long each block is
                    block = int(f[g+'/X'][()].shape[0]/num_traj)
                    # calculate how much to include for each traj
                    tt = int(T_train/dt)

                    for n in range(num_traj):
                        data_short_X.extend(f[g + '/X'][n*block:n*block+tt])
                        data_short_y.extend(f[g + '/y'][n*block:n*block+tt])

                    # normalizing and appending
                    self.X.extend([x/normalization for x in data_short_X])
                    self.y.extend([y/normalization for y in data_short_y])
                    # creating the time vector based on T_train
                    for _ in range(num_traj):
                        self.t.extend([i*dt for i in range(int(T_train/dt))])

                else:
                    self.X.extend(f[g + '/X'][()] / normalization)
                    self.y.extend(f[g + '/y'][()] / normalization)


                    # creating the time vector based on T_train
                    for _ in range(num_traj):
                        self.t.extend([i*dt for i in range(int(T_train/dt) - 1)])

                print(len(self.X))
                # I have to extract the potential from the name
                # and add a vector of the same length
                self.V.extend([int(g[14:18])*1e-3]*len(f[g + '/X'][()]))

    def __getitem__(self, index):
        # return the potential and the vector at t and t+dt
        # as tensors
        return torch.tensor(self.V[index]), torch.tensor(self.t[index]), \
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

    def __init__(self, mlp_params, potential, time_dependent=False):
        '''Init function
        Here i need the temperature to normalize data accordingly
        '''
        super().__init__()
        if time_dependent:
            self.MLP = exp_LL_td(**mlp_params)
            #self.MLP = exp_LL_td_plus(**mlp_params)
        else:
            self.MLP = exp_LL(**mlp_params)  # multi(=1) layer perceptron
        #self.MLP = exp_LL_custom_V(**mlp_params)

        self.dt = mlp_params['dt']
        self.potential = potential
        self.td = time_dependent

    def forward(self, t, x):
        '''Forward step of the model
        '''
        return self.MLP.forward(t, x)

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
        if self.potential > 1:
            normalization /= self.potential**2
        results = [v_0]

        X = torch.tensor(v_0/normalization, dtype=torch.double)

        length = int(T/self.dt)

        for i in range(length-1):
            with torch.no_grad():
                y = self.MLP.predict(torch.Tensor([i*self.dt]), X.float())
                X = y.clone()
                results.extend([y.numpy()*normalization])

        return results

    def thermal_state(self, v_0, beta):
        '''Function that runs the time evolution for 10/gap
        This I consider thermalized

        Parameters
        ----------
        v_0 : array
            Initial conditions
        beta : float
            Inverse temperature of the initial condition

        Return
        ------
        final coherence vetor
        '''
        normalization = 1 - math.e**(-beta/2)
        if self.potential > 1:
            normalization /= self.potential**2

        #X = torch.zeros(16, dtype=torch.float)
        #X[0] = 0.5
        X = torch.tensor(v_0, dtype=torch.float)
        X /= normalization
        #X = torch.tensor(v_0/normalization, dtype=torch.float)

        if self.td:
            # if time dependent, have to do every dt
            time = 0.0
            gap = 1

            while( (time/gap) < 100):
                Lindblad = self.MLP.get_L(time)
                # e_val, e_vec = np.linalg.eig(Lindblad.detach().numpy())
                gap = self.MLP.gap(time)

                exp_dt_L = torch.matrix_exp(self.dt*Lindblad )
                X = torch.add(exp_dt_L[1:,0], X @ torch.transpose(exp_dt_L[1:,1:],0,1))

                time += self.dt

        else: # if not time dependant, things are much easier
            # get the Lindbladian and the gap
            Lindblad = self.MLP.get_L()
            # e_val, e_vec = np.linalg.eig(Lindblad)
            gap = self.MLP.gap()

            # get the exp and apply to X
            # NOTE: done step by step to avoid numerical errors
            exp_dt_L = torch.matrix_exp((1./gap)*Lindblad )

            for i in range(100):
                X = torch.add(exp_dt_L[1:,0], X @ torch.transpose(exp_dt_L[1:,1:],0,1))
                #X = torch.matrix_exp( (1./gap)*Lindblad ) @ X
            #y = torch.add(exp_dt_L[1:,0], X @ torch.transpose(exp_dt_L[1:,1:],0,1))

        print(f'Time {100./gap}')

        return X.detach().numpy()*normalization

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
        self.v_x.data = 0.01*self.v_x.data
        self.v_y.data = 0.01*self.v_y.data
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init

    def forward(self, _, x):
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

    def predict(self, t, x):
        '''Dummy function for compatibility with exp_LL_td'''
        return self.forward(t, x)

    def get_L(self):
        ''' Function that calculate the Lindbaldian
        '''
        with torch.no_grad():
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

        return L

    def gap(self):
        '''Function to calculate the Lindblad gap,
        meaning the smallest real part of spectrum in modulus
        '''

        L = self.get_L()
        # take the real part of the spectrum
        e_val = np.linalg.eigvals(L.detach().numpy()).real

        e_val.sort()
        return np.abs(e_val[-2])


class exp_LL_td(nn.Module):
    ''' Custom Liouvillian **time-depenent** layer
    to ensure positivity of the rho

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
        self.v_x_net = nn.Sequential(nn.Linear(1, 2),
                                     nn.ReLU(),
                                     nn.Linear(2, 4),
                                     nn.ReLU(),
                                     nn.Linear(4, 4),
                                     nn.ReLU(),
                                     #nn.Tanh(),
                                     nn.Linear(4, self.data_dim**2),
                                     nn.Unflatten(-1, (self.data_dim, self.data_dim))).float()
        self.v_y_net = nn.Sequential(nn.Linear(1, 2),
                                     nn.ReLU(),
                                     nn.Linear(2, 4),
                                     nn.ReLU(),
                                     nn.Linear(4, 4),
                                     nn.ReLU(),
                                     #nn.Tanh(),
                                     nn.Linear(4, self.data_dim**2),
                                     nn.Unflatten(-1, (self.data_dim, self.data_dim))).float()

        # Hamiltonian parameters omega
        #self.omega_net = nn.Sequential(nn.Linear(1, self.data_dim)).float()
        omega = torch.zeros([data_dim])
        self.omega = nn.Parameter(omega).float()


        # initialize omega and v
        self.v_x_net.apply(init_weights)
        self.v_y_net.apply(init_weights)
        #self.omega_net.apply(init_weights)

        # keeping the biases initialized as the previous case
        # nn.init.kaiming_uniform_(self.v_x_net, a=1)
        # nn.init.kaiming_uniform_(self.v_y, a=1)
        # # rescaling to avoid too big initial values
        # self.v_x.data = 0.01*self.v_x.data
        # self.v_y.data = 0.01*self.v_y.data
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)

        # bound = 1. / np.sqrt(fan_in)
        # nn.init.uniform_(self.omega, -bound, bound)  # bias init

        nn.init.normal_(self.v_x_net[-2].bias, 0.0, 0.001)
        nn.init.normal_(self.v_y_net[-2].bias, 0.0, 0.001)
        #nn.init.normal_(self.omega_net[-1].bias, 0.0, 0.001)
        nn.init.normal_(self.omega, 0.0, 0.001)


    def forward(self, t, x):
        batch_size = x.shape[0]

        # making the time tensor the right dimension
        t.unsqueeze_(1)

        v_x = self.v_x_net(t)
        v_y = self.v_y_net(t)
        #omega = self.omega_net(t)
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        # NOTE: s index is batch index
        c_re = torch.add(torch.einsum('ski,skj->sij', v_x, v_x),\
                         torch.einsum('ski,skj->sij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ski,skj->sij', v_x, v_y),\
                         -torch.einsum('ski,skj->sij', v_y, v_x) )

        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics.
        # Einsum not optimized in torch: https://optimized-einsum.readthedocs.io/en/stable/

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,sij->smn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,sij->smn', self.f, self.f, c_re )
        im_1 = -4.*torch.einsum('mjk,nik,sij->smn', self.f, self.d, c_im )
        im_2 =  4.*torch.einsum('mik,njk,sij->smn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = -4.*torch.einsum('imj,sij->sm', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, self.omega).unsqueeze_(0)

        # building the Lindbladian operator
        L = torch.zeros(batch_size, self.data_dim+1, self.data_dim+1)
        L[:, 1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[:, 1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        #print( torch.einsum('si,sji->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)).shape )
        return torch.add(exp_dt_L[:,1:,0], torch.einsum('si,sij->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)))


    def predict(self, t, x):
        v_x = self.v_x_net(t)
        v_y = self.v_y_net(t)
        #omega = self.omega_net(t)

        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        # NOTE: s index is batch index
        c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                         torch.einsum('ki,kj->ij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                         -torch.einsum('ki,kj->ij', v_y, v_x) )

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

        tr_id = -4.*torch.einsum('imj,ij->m', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, self.omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

    def get_L(self, t):
        '''Function that calculate the Lindbladian
        '''
        with torch.no_grad():
            t = torch.Tensor([t])
            v_x = self.v_x_net(t).reshape(self.data_dim, self.data_dim)
            v_y = self.v_y_net(t).reshape(self.data_dim, self.data_dim)
            omega = self.omega_net(t).reshape(self.data_dim)

            # Structure constant for SU(n) are defined
            #
            # We define the real and imaginary part od the Kossakowsky's matrix c.
            #       +
            # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
            #              k    ki   kj     ki  kj         ki  kj   ki  kj
            # NOTE: s index is batch index
            c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                             torch.einsum('ki,kj->ij', v_y, v_y)  )
            c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                             -torch.einsum('ki,kj->ij', v_y, v_x) )

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

            tr_id = -4.*torch.einsum('imj,ij->m', self.f, c_im )

            h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, omega)

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

class exp_LL_td_plus(nn.Module):
    ''' Custom Liouvillian **time-dependent** layer
    to ensure positivity of the rho
    plus = also dependent on the coherence vector!

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
        self.v_x_net = nn.Sequential(nn.Linear(self.data_dim+1, 16),
                                     nn.ReLU(),
                                     nn.Linear(16, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     #nn.Tanh(),
                                     nn.Linear(1024, self.data_dim**2),
                                     nn.Unflatten(-1, (self.data_dim, self.data_dim))).float()
        self.v_y_net = nn.Sequential(nn.Linear(self.data_dim+1, 16),
                                     nn.ReLU(),
                                     nn.Linear(16, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     #nn.Tanh(),
                                     nn.Linear(1024, self.data_dim**2),
                                     nn.Unflatten(-1, (self.data_dim, self.data_dim))).float()

        # Hamiltonian parameters omega
        self.omega_net = nn.Sequential(nn.Linear(1, self.data_dim)).float()

        # initialize omega and v
        self.v_x_net.apply(init_weights)
        self.v_y_net.apply(init_weights)
        self.omega_net.apply(init_weights)

        # keeping the biases initialized as the previous case
        # nn.init.kaiming_uniform_(self.v_x_net, a=1)
        # nn.init.kaiming_uniform_(self.v_y, a=1)
        # # rescaling to avoid too big initial values
        # self.v_x.data = 0.01*self.v_x.data
        # self.v_y.data = 0.01*self.v_y.data
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        # bound = 1. / np.sqrt(fan_in)
        # nn.init.uniform_(self.omega, -bound, bound)  # bias init

        nn.init.normal_(self.v_x_net[-2].bias, 0.0, 0.001)
        nn.init.normal_(self.v_y_net[-2].bias, 0.0, 0.001)
        nn.init.normal_(self.omega_net[-1].bias, 0.0, 0.001)


    def forward(self, t, x):
        batch_size = x.shape[0]

        # making the time tensor the right dimension
        t.unsqueeze_(1)

        v_x = self.v_x_net(torch.cat((t,x), 1))
        v_y = self.v_y_net(torch.cat((t,x), 1))
        omega = self.omega_net(t)
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        # NOTE: s index is batch index
        c_re = torch.add(torch.einsum('ski,skj->sij', v_x, v_x),\
                         torch.einsum('ski,skj->sij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ski,skj->sij', v_x, v_y),\
                         -torch.einsum('ski,skj->sij', v_y, v_x) )

        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics.
        # Einsum not optimized in torch: https://optimized-einsum.readthedocs.io/en/stable/

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,sij->smn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,sij->smn', self.f, self.f, c_re )
        im_1 = -4.*torch.einsum('mjk,nik,sij->smn', self.f, self.d, c_im )
        im_2 =  4.*torch.einsum('mik,njk,sij->smn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im )

        tr_id = -4.*torch.einsum('imj,sij->sm', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,sk->sji', self.f, omega)

        # building the Lindbladian operator
        L = torch.zeros(batch_size, self.data_dim+1, self.data_dim+1)
        L[:, 1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[:, 1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        #print( torch.einsum('si,sji->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)).shape )
        return torch.add(exp_dt_L[:,1:,0], torch.einsum('si,sij->sj', x, torch.transpose(exp_dt_L[:,1:,1:],1,2)))


    def predict(self, t, x):
        #t.unsqueeze_(1)
        v_x = self.v_x_net(torch.cat((t,x)))
        v_y = self.v_y_net(torch.cat((t,x)))
        omega = self.omega_net(t)

        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        # NOTE: s index is batch index
        c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                         torch.einsum('ki,kj->ij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                         -torch.einsum('ki,kj->ij', v_y, v_x) )

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

        tr_id = -4.*torch.einsum('imj,ij->m', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        exp_dt_L = torch.matrix_exp(self.dt*L )
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

    def get_L(self, t):
        '''Function that calculate the Lindbladian
        '''
        t = torch.Tensor([t])
        v_x = self.v_x_net(t).reshape(self.data_dim, self.data_dim)
        v_y = self.v_y_net(t).reshape(self.data_dim, self.data_dim)
        omega = self.omega_net(t).reshape(self.data_dim)

        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        # NOTE: s index is batch index
        c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                         torch.einsum('ki,kj->ij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                         -torch.einsum('ki,kj->ij', v_y, v_x) )

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

        tr_id = -4.*torch.einsum('imj,ij->m', self.f, c_im )

        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, omega)

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

    def __init__(self, data_dim, layers, nonln, output_nonlin, dt):
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
