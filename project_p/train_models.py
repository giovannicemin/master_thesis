import torch
import matplotlib.pyplot as plt
from pathlib import Path


from ml.classes import MLLP
from ml.utils import ensure_empty_dir, load_data
from ml.core import train, eval
from sfw.optimizers import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CyclicLR

S = [1, 2, 3, 4, 5, 6, 7, 8, 9]
W = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
SW = W + S

data_gen_params = {'L': 20,               # length of spin chain
                   'sites': [0, 1],        # sites of the subsystem S spins
                   'omega': 1,             # Rabi frequency
                   # interaction of subsystem's S spins
                   'potential': [0.1],
                   'potential_': None,     # interaction of bath spins, if None same as potential
                   'T': 10,                # total time for the evolution
                   'dt': 0.01,             # interval every which save the data
                   'cutoff': 1e-5,         # cutoff for TEBD algorithm
                   'im_cutoff': 1e-10,      # cutoff for TEBD algorithm, img t-e
                   'tolerance': 1e-3,      # Trotter tolerance for TEBD algorithm
                   'verbose': True,        # verbosity of the script
                   'num_traj': 30,         # how many trajectories to do
                   # file to save the data
                   'fname': './data/data_tebd.hdf5'
                  }

ml_params = {'model_dir': './data/trained/',  # folder where the metadata of the training are stored
             'validation_split': 0,
             'batch_size': 256,
             'time_dependent': False,
             # 'batches_per_epoch': 256,
             'n_epochs': 1000,
             'device': 'cpu',
             'mlp_params': {
                 'data_dim': 15,
                 'layers': [],
                 'nonlin': 'id',
                 'output_nonlin': 'id',
                 'dt': 0.01
                 },
             }

if __name__ == '__main__':
    '''Here I train different models for different parameters'''

    # check if the model laready exists
    # ensure_empty_dir(ml_params['model_dir'])

    # then I train the model for all potentials
    prms = data_gen_params

    beta = prms["beta"] # one or more values, to select the training data (MUST be array)
    total_time = prms['T']

    print(f'Training data with beta = {beta} and T = {total_time}')

    for potential in prms["potential"]:
        print('====================================================\n')
        print(f'=== Training the model for V = {potential}')

        # load the data
        train_loader, eval_loader = load_data(prms['fname'], prms['L'], beta, [potential],
                                              prms['dt'], prms['T'],
                                              prms['num_traj'],
                                              ml_params['batch_size'],
                                              ml_params['validation_split'],
                                              resize=False) #OCIO check
        # create the model
        model = MLLP(ml_params['mlp_params'], potential=potential,
                     time_dependent=ml_params['time_dependent']).to(ml_params['device'])
        # name the model
        name = 'model_L_' + str(prms['L']) + \
                '_V_' + str(int(potential*1e3)).zfill(4) + \
                '_dt_' + str(int(prms['dt']*1e3)).zfill(4) + \
                '_T' + str(prms['T']).zfill(2) + 'r'

        # load existing model
        # model.load_state_dict(torch.load(Path('./data/trained_noisy/' + name)))

        criterion = torch.nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        #optimizer = Adam(model.parameters(), lr=0.01)
        scheduler = ExponentialLR(optimizer, 1)
        # scheduler = MultiStepLR(optimizer, milestones=[150], gamma=0.1)
        #scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3,
        #                     step_size_up=100, step_size_down=900,
        #                     mode='triangular', cycle_momentum=False)

        # train the model
        loss = train(model, criterion, optimizer, scheduler, train_loader,
              ml_params['n_epochs'], ml_params['device'],
                     epochs_to_prune=[], alpha_1=[], alpha_2=[])

        # save the model
        torch.save(model.state_dict(), ml_params['model_dir'] + name)
