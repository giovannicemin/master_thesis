'''Where the core operation for the machine learning part are stored.
'''
import numpy as np
import torch

from sfw.constraints import create_simplex_constraints

def train(model, criterion, optimizer, scheduler, train_loader, n_epochs, device,
          epochs_to_prune, alpha_1, alpha_2):
    '''Function to train the model in place

    Parameters
    ----------
    model : nn.Module subclass
        Class implementing the model to train
    criterion : nn.loss
        Some function implementing the loss
    optimizer : torch.optim
        Optimizer for the training procedure
    scheduler : torch.optim
        Schedule the learning rate behavour
    train_loader : torch.utils.data.DataLoader
        Loader for the training data
    n_epochs : int
        Number of epochs
    device : str
        Device to send the computations
    epochs_to_prune : array
        At which epochs pruning is done
    alpha_1 : float
        Parameter setting the L1 regularization strength
    alpha_2 : float
        Parameter setting the L2 regularization strength
    '''
    mean_train_loss = []

    for epoch in range(1, n_epochs+1):
        model.train()
        print('= Starting epoch ', epoch, '/', n_epochs)

        summed_train_loss = np.array([])

        # Train
        for batch_index, (vv, t, batch_in, batch_out) in enumerate(train_loader):

            constraints = create_simplex_constraints(model)

            X = batch_in.float().to(device)
            y = batch_out.float().to(device)

            # set gradients to zero to avoid using old data
            optimizer.zero_grad()

            # apply the model
            recon_y = model.forward(t=t, x=X)

            # calculate the loss
            loss = 1e4*criterion(recon_y, y)
            # sum to the loss per epoch
            summed_train_loss = np.append(summed_train_loss, loss.item())

            # regularization for U to be unitary
            # u_re = model.MLP.u_re
            # u_im = model.MLP.u_im
            # matrix_1 = torch.mm(u_re, u_re.T) + torch.mm(u_im, u_im.T) - torch.eye(u_re.shape[0])
            # matrix_1 = torch.abs(matrix_1)
            # matrix_2 = torch.mm(u_im, u_re.T) - torch.mm(u_re, u_im.T)
            # matrix_2 = torch.abs(matrix_2)
            # loss += (1e-5)*( torch.sum(matrix_1) + torch.sum(matrix_2) )
            # L2 regularization
            # norm_weights = torch.norm(model.MLP.omega_net.weights)*\
            #     torch.norm(model.MLP.gamma_net.weights)
            # loss /= 1 + lamda*norm_weights

            #--> regularization
            l2_regularizer = torch.norm(model.MLP.omega_net.weight)*\
                            torch.norm(model.MLP.gamma_net[0].weight)
            l1_regularizer = torch.norm(model.MLP.omega_net.weight, 1) + \
                            torch.norm(model.MLP.gamma_net[0].weight, 1)
                            #torch.norm(model.MLP.u_re, 1) + \
                            #torch.norm(model.MLP.u_im, 1)
            # Elastic Net
            #loss += 0.01*( torch.norm(model.MLP.u_re) + torch.norm(model.MLP.u_im) )
            loss += alpha_1*l1_regularizer
            loss /= 1 + alpha_2*l2_regularizer
            #print(f"L1 {alpha_1*l1_regularizer}")

            # backpropagate = calculate derivatives
            loss.backward(retain_graph=True)

            # update values
            optimizer.step(constraints)

        scheduler.step()

        # prune the model
        #if epoch >= epoch_to_prune and epoch%40 == 0:
        if epoch in epochs_to_prune:
            with torch.no_grad():
                # here I set to 0 the minimum value along each line
                w = model.MLP.omega_net.weights
                mask = (w != w.min(dim=-1).values.unsqueeze(-1))
                model.MLP.omega_net.weights *= mask
                w = model.MLP.gamma_net.weights
                mask = (w != w.min(dim=-1).values.unsqueeze(-1))
                model.MLP.gamma_net.weights *= mask

                # here I set to 0 the values below therehold
                # mask = torch.abs(model.MLP.omega_net.weights[:, :]) < model.MLP.omega_net.threshold
                # model.MLP.omega_net.weights[mask] = 0
                # mask = torch.abs(model.MLP.omega_net.weights[:, :]) < model.MLP.gamma_net.threshold
                # model.MLP.gamma_net.weights[mask] = 0
                print('prune')
        #custom_pruning_unstructured(model.MLP.omega_net, 'weights', threshold=1e-2)

        print('=== Mean train loss: {:.12f}'.format(summed_train_loss.mean()))
        print('=== lr: {:.5f}'.format(scheduler.get_last_lr()[0]))
        #print(f'frequency gamma {model.MLP.gamma_net.frequencies}, omega {model.MLP.omega_net.frequencies}')
        mean_train_loss.append(summed_train_loss.mean())

    return mean_train_loss

def eval(model, criterion, eval_loader, device):
    '''Function to evaluate the model
    '''

    model.eval()
    summed_eval_loss = np.array([])

    for batch_index, (vv, t, batch_in, batch_out) in enumerate(eval_loader):

        X = batch_in.float().to(device)
        y = batch_out.float().to(device)

        # apply the model
        recon_y = model.forward(t=t, x=X)

        # calculate the loss
        loss = criterion(recon_y, y)

        summed_eval_loss = np.append(summed_eval_loss, loss.item())

    print('=== Test set loss:   {:.12f}'.format(summed_eval_loss.mean()))
    return {'loss': summed_eval_loss}
