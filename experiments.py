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
import architectures as arches

# aliases
L = lasagne.layers
nl = lasagne.nonlinearities
T = theano.tensor

# directories
headdir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games')
paramsdir_ = os.path.join(headdir, 'Analysis/0_hvh/Params/nnets/')
datadir = os.path.join(headdir, 'Data/model input')
resultsdir = os.path.join(headdir, 'Analysis/0_hvh/Loglik/nnets')

# loading data
data = loading.default_loader(os.path.join(datadir, '1-4 (no computer).csv'))
hvhdata = loading.default_loader(os.path.join(datadir, '0 (with groups).csv'))
Xs = np.concatenate(hvhdata[2])                                                     # hvhdata[2] is a list of 5 sets of Xs, per CV group provided in data files
ys = np.concatenate(hvhdata[3])
Ss = np.concatenate(hvhdata[4])

# load network specs
with open('arch_specs.yaml') as archfile:
    arch_dict = yaml.load(archfile)                                                 # load dictionary specifying architectures for easy reference

# experiment functions
def run_tuned_fit(architecture, data, hvhdata, paramsdir,
                    tune=True, save=True, freeze=True):
    """
    Runs the full fitting experiment, pretraining on later experiments and
    testing on first.

    Saves data as it goes to avoid eating memory.
    (SORT OF; break this up into two functions)

    Eventually, trainer train_all should be modified to take a network
    object instead of an architecture; this would save time by avoiding
    recompiling each iteration
    """

    import architectures as arches                                                  # need for use in notebook
    archname = architecture['name']
    archfunc = getattr(arches, architecture['type'])
    arch = lambda input_var=None: archfunc(input_var, **architecture['kwargs'])     # specifier function for network that freezes hyperparameters

    tunekws = {'freeze': freeze, 'exclude': [-5]}                                   # kwargs for FineTuning

    # start training
    trainer = DefaultTrainer(stopthresh=50, print_interval=25)                      # initialize trainer
    net_list = trainer.train_all(architecture=arch, data=data, seed=985227)         # returns a list of networks trained on every split in data

    # save params
    if save:
        for i, n in enumerate(net_list):
            fname = '{} {} split agg fit exp 1-4'.format(archname, i)
            n.save_params(os.path.join(paramsdir, fname))

    # fine tuning on hvhdata
    if tune:
        tuner = FineTuner(stopthresh=10)                                            # strict threshold to prevent overfitting

        for i, n in enumerate(net_list):                                            # for each pretrained network
            for j in range(5):

                fname = '{} {} agg fit exp 1-4 {} tune fit exp 0'.format(archname, i, j)
                params = L.layers.get_all_param_values(n.net)
                net = tuner.run_split(
                    architecture=arch, data=hvhdata, split=j,
                    startparams=params, **tunekws
                )

                if save:
                    net.save_params(os.path.join(paramsdir, fname))

    return None

def run_bulk_fit(archname):
    return None

def run_tuned_experiment(archnames):
    """
    Provided a list of architecture names (as in arch_specs.yaml),
    runs a full fit + fine tuning sequence with run_tuned_fit
    """

    for name in archnames:
        paramsdir = os.path.join(paramsdir_, name[:-1])
        architecture = arch_dict[name]

        af = getattr(arches, architecture['type'])
        net = af(**architecture['kwargs'])

        print('Param count:', L.count_params(net))
        print(architecture)

        run_tuned_fit(architecture, data, hvhdata, paramsdir=paramsdir, tune=True)

def run_bulk_experiment(archnames):
    return None

def main(argv):
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
