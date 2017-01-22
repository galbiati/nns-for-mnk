# runs a fitting routine

# imports
import os
import sys
import numpy as np
import pandas as pd
import theano
import lasagne
import loading
from training import *
from network import *
from architectures import *

# aliases
L = lasagne.layers
nl = lasagne.nonlinearities
T = theano.tensor

# directories
headdir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games')
paramsdir = os.path.join(headdir, 'Analysis/0_hvh/Params/nnets/temp')
datadir = os.path.join(headdir, 'Data/model input')
resultsdir = os.path.join(headdir, 'Analysis/0_hvh/Loglik/nnets')

# loading data
data = loading.default_loader(os.path.join(datadir, '1-4 (no computer).csv'))
hvhdata = loading.default_loader(os.path.join(datadir, '0 (with groups).csv'))
Xs = np.concatenate(hvhdata[2])
ys = np.concatenate(hvhdata[3])
Ss = np.concatenate(hvhdata[4])

# fit function
def run_full_fit(arch, archname):
    """
    Runs the full fitting experiment,
    pretraining on later experiments and testing on first.

    Saves data as it goes to avoid eating memory.
    """

    # start training
    trainer = DefaultTrainer(stopthresh=75, print_interval=20)
    net_list = trainer.train_all(architecture=arch, data=data, seed=985227)

    # save params
    for i, n in enumerate(net_list):
        fname = '{} {} split agg fit exp 1-4'.format(archname, i)
        n.save_params(os.path.join(paramsdir, fname))

    tuner = FineTuner(stopthresh=20)
    for i, n in enumerate(net_list):
        for j in range(5):
            fname = '{} {} agg fit exp 1-4 {} tune fit exp 0'.format(archname, i, j)
            params = L.get_all_param_values(n.net)
            net = tuner.train_all(
                architecture=arch, data=hvhdata,
                split=j, startparams=params, freeze=True
            )
            net.save_params(os.path.join(paramsdir, fname))

    return None

if __name__ == '__main__':
    run_full_fit(multiconvX, 'multiconvX_lrg')
