import torch

from ml.classes import MLLP
from ml.utils import ensure_empty_dir, load_data
from sfw.optimizers import Adam

data_gen_params = {'L' : 10,               # length of spin chain
                  'sites' : [0, 1],        # sites of the subsystem S spins
                  'omega' : 1,             # Rabi frequency
                  # inverse temperature
                  'beta' : [0.001, 0.005, 0.01, 0.05, 0.1],
                  # interaction of subsystem's S spins
                  'potential' : [0.1, 0.2, 0.3, 0.4, 0.5],
                  'potential_' : None,     # interaction of bath spins, if None same as potential
                  'T' : 10,                # total time for the evolution
                  'dt' : 0.02,             # interval every which save the data
                  'cutoff' : 1e-5,         # cutoff for TEBD algorithm
                  'im_cutoff' : 1e-7,      # cutoff for TEBD algorithm, img t-e
                  'tolerance' : 1e-5,      # Trotter tolerance for TEBD algorithm
                  'verbose' : True,        # verbosity of the script
                  'num_traj' : 20,         # how many trajectories to do
                  # file to save the data
                  'fname' : './data/data_tebd.hdf5'
                  }

ml_params = {'model_dir': './data/trained_model', # folder where the metadata of the training are stored
             'validation_split' : 0.2,
             'batch_size': 256,
             #'batches_per_epoch': 256,
             'n_epochs': 20,
             'device': 'cpu',
             'mlp_params': {
                 'data_dim': 15,
                 'layers': [],
                 'nonlin': 'id',
                 'output_nonlin': 'id'
             },
         }

if __name__ == '__main__':

    # check if the model laready exists
    ensure_empty_dir(ml_params['model_dir'])

    # then I train the model for all combinations of params
    prms = data_gen_params
    for beta in prms["beta"]:
        for potential in prms["potential"]:
            print('====================================================\n')
            print(f'===Training the model for beta = {beta} and V = {potential}')

            # load the data
            train_loader, val_loader = load_data(prms['fname'], prms['L'], beta, potential,
                                                 prms['dt'],
                                                 ml_params['batch_size'],
                                                 ml_params['validation_split'])
            # create the model
            model = MLLP(ml_params['mlp_params']).to(ml_params['device'])

            model.set(loss = torch.nn.MSELoss(),
                      optimizer = Adam(model.parameters(), lr =0.01))

            # train the model
            model.train_model(train_loader, ml_params['n_epochs'], ml_params['device'])

            # eval the model
            model.eval_model(val_loader, ml_params['device'])

            # save the model
            name = 'model_L_' + str(prms['L']) + \
                '_V_' + str(int(potential*1e3)).zfill(4) + \
                '_beta_' + str(int(beta*1e3)).zfill(4) + \
                '_dt_' + str(int(prms['dt']*1e3)).zfill(4)

            torch.save(model.state_dict(), './data/trained_model/' + name)
