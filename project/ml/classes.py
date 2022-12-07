'''Where all the classes used for the ML part of the project
are stored.
'''
import numpy as np
import pandas as pd
import torch
from torch import nn
import opt_einsum as oe
from torch.utils.data.dataset import Dataset

from sfw.constraints import create_simplex_constraints
from ml.utils import pauli_s_const, get_arch_from_layer_list

class CustomDatasetFromHDF5(Dataset):
    '''Class implementing the Dataset object of
    the data to the pass to DataLoader

    Parameters
    ----------
    path : str
        Path to where hdf5 file is
    group : str
        Group name of the desired data
    '''

    def __init__(self, path, group):
        self.data = pd.read_hdf(path, group)

    def __getitem__(self, index):
        # get the vector at t and t+dt
        self.v = torch.tensor(self.data.iloc[index])
        self.v1 = torch.tensor(self.data.iloc[index+1])
        return self.v, self.v1

    def __len__(self):
        # return the shape -1, because I have to account
        # for the fact last point has no t+dt :(
        return self.data.shape[0] - 1

class MLLP(nn.Module):
    '''Machine learning model to parametrize the Lindbladian operator

    Parametes
    ---------
    mlp_params : dict
        Dictionary containing all parameters needed to exp_LL
    rev_loss_fn : loss function
        Loss function to employ in the model,
        default value torch.nn.MSELoss
    '''

    def __init__(self, mlp_params):
        super().__init__()
        self.MLP = exp_LL(**mlp_params)  # multi(=1) layer perceptron

    def set(self, loss, optimizer):
        '''Function to set loss and optimizer of the model
        '''
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        '''Forward step of the model
        '''
        return self.MLP(x)

    def train_model(self, train_loader, n_epochs, device):
        '''Function to train the model
        '''

        for epoch in range(n_epochs):
            self.train()
            print('= Starting epoch ', epoch, '/', n_epochs)

            summed_train_loss = np.array([])
            # Train
            for batch_index, (batch_in, batch_out) in enumerate(train_loader):

                constraints = create_simplex_constraints(self)

                X = batch_in.float().to(device)
                y = batch_out.float().to(device)

                # set gradients to zero to avoid using old data
                self.optimizer.zero_grad()

                # apply the model
                recon_y = self.forward(X)

                # calculate the loss
                loss = self.loss(recon_y, y)

                # sum to the loss per epoch
                summed_train_loss = np.append(summed_train_loss, loss.item())

                # backpropagate = calculate derivatives
                loss.backward()

                # update values
                self.optimizer.step(constraints)

            print('=== Mean train loss: {:.12f}'.format(summed_train_loss.mean()))

    def eval_model(self, eval_loader, device):
        self.eval()
        summed_eval_loss = np.array([])

        for batch_index, (batch_in, batch_out) in enumerate(eval_loader):

            X = batch_in.float().to(device)
            y = batch_out.float().to(device)

            # apply the model
            recon_y = self.forward(X)

            # calculate the loss
            loss = self.loss(recon_y, y)

            summed_eval_loss = np.append(summed_eval_loss, loss.item())

        print('=== Test set loss: {:.12f}'.format(summed_eval_loss.mean()))
        return {'loss': summed_eval_loss}

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

    def __init__(self, data_dim, layers, nonlin, output_nonlin, dt=0.01):
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
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init

    def forward(self, x):
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
        # memo: first element of x is identity
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))
