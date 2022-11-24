#!/usr/bin/env python
'''Main of the project
'''


default_parameters = {'model_dir': './data/', # folder where the metadata of the training are stored
                      'batch_size': 256,
                      'batches_per_epoch': 256,
                      'n_epochs': 20,
                      'device': 'cpu',
                      #'MLP_pr': {
                          'mlp_params': {
                              'data_dim': 15,
                              'layers': [],
                              'nonlin': 'id',
                              'output_nonlin': 'id'
                          },
                      #},
                  }




if __name__ == '__main__':
