# imports
import os
import sys
import yaml
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

def run_full_fit(architecture, data, hvhdata, paramsdir, tune=True, save=True):
    """
    Runs the full fitting experiment, pretraining on later experiments and testing on first.
    Saves data as it goes to avoid eating memory.
    (SORT OF; break this up into two functions)
    """

    import architectures as arches
    archname = architecture['name']
    archfunc = getattr(arches, architecture['type'])
    arch = lambda input_var=None: archfunc(input_var, **architecture['kwargs'])

    tunekws = {'freeze': True, 'exclude': [-5]}

    # start training
    trainer = DefaultTrainer(stopthresh=50, print_interval=25)
    net_list = trainer.train_all(architecture=arch, data=data, seed=985227)

    # save params
    if save:
        for i, n in enumerate(net_list):
            fname = '{} {} split agg fit exp 1-4'.format(archname, i)
            n.save_params(os.path.join(paramsdir, fname))

    if tune:
        tuner = FineTuner(stopthresh=10)

        for i, n in enumerate(net_list):
            for j in range(5):

                fname = '{} {} agg fit exp 1-4 {} tune fit exp 0'.format(archname, i, j)
                params = L.layers.get_all_param_values(n.net)
                net = tuner.train_all(architecture=arch, data=hvhdata, split=j, startparams=params, **tunekws )

                if save:
                    net.save_params(os.path.join(paramsdir, fname))

    return None

def main():
    paramsdir = os.path.abspath('/scratch/gvg218/params_archive')

    with open('arch_specs.yaml') as archfile:
        arch_dict = yaml.load(archfile)

    for name, architecture in arch_dict.items():
        run_full_fit(architecture, data=data, hvhdata=hvhdata, tune=True)


if __name__ == '__main__':
    main()
