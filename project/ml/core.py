'''Where the core operation for the machine learning part are stored.
'''
import os

from ml.utils import ensure_empty_dir

def train(prms, data_gen_params):
    '''Function to train the model
    '''
    # check if the model laready exists
    ensure_empty_dir(prms['model_dir'])
