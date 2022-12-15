'''Where the core operation for the machine learning part are stored.
'''
import numpy as np
import math

from sfw.constraints import create_simplex_constraints
from ml.utils import ensure_empty_dir

def train(model, criterion, optimizer, train_loader, n_epochs, device, beta):
    '''Function to train the model in place

    Parameters
    ----------
    model : nn.Module subclass
        Class implementing the model to train
    criterion : nn.loss
        Some function implementing the loss
    optimizer : torch.optim
        Optimizer for the training procedure
    train_loader : torch.utils.data.DataLoader
        Loader for the training data
    n_epochs : int
        Number of epochs
    device : str
        Device to send the computations
    '''
    # normalization of the data
    normalization = 1 - math.e**(-beta/2)

    for epoch in range(n_epochs):
        model.train()
        print('= Starting epoch ', epoch, '/', n_epochs)

        summed_train_loss = np.array([])

        # Train
        for batch_index, (batch_in, batch_out) in enumerate(train_loader):

            constraints = create_simplex_constraints(model)

            X = batch_in.float().to(device)/normalization
            y = batch_out.float().to(device)/normalization

            # set gradients to zero to avoid using old data
            optimizer.zero_grad()

            # apply the model
            recon_y = model.forward(X)

            # calculate the loss
            loss = criterion(recon_y, y)

            # sum to the loss per epoch
            summed_train_loss = np.append(summed_train_loss, loss.item())

            # backpropagate = calculate derivatives
            loss.backward()

            # update values
            optimizer.step(constraints)

        print('=== Mean train loss: {:.12f}'.format(summed_train_loss.mean()))

def eval(model, criterion, eval_loader, device, beta):
    '''Function to evaluate the model
    '''
    # normalization of the data
    normalization = 1 - math.e**(-beta/2)

    model.eval()
    summed_eval_loss = np.array([])

    for batch_index, (batch_in, batch_out) in enumerate(eval_loader):

        X = batch_in.float().to(device)/normalization
        y = batch_out.float().to(device)/normalization

        # apply the model
        recon_y = model.forward(X)

        # calculate the loss
        loss = criterion(recon_y, y)

        summed_eval_loss = np.append(summed_eval_loss, loss.item())

    print('=== Test set loss:   {:.12f}'.format(summed_eval_loss.mean()))
    return {'loss': summed_eval_loss}
