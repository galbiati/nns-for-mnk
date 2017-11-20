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

"""Reorganize this to put experiments together, rather than functiont types"""

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

fake_data = loading.default_loader(os.path.join(datadir, 'fake news (with groups).csv'))

# load network specs
with open('arch_specs.yaml') as archfile:
    arch_dict = yaml.load(archfile)                                                 # load dictionary specifying architectures for easy reference

# experiment functions
def run_tuned_fit(architecture, data, hvhdata, paramsdir,
                    tune=True, save=False, freeze=True):
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
                params = L.get_all_param_values(n.net)
                net = tuner.run_split(
                    architecture=arch, data=hvhdata, split=j,
                    startparams=params, **tunekws
                )

                if save:
                    net.save_params(os.path.join(paramsdir, fname))

    return None

def run_bulk_fit(architecture, paramsdir):
    """
    Runs a single pass through the "bulk" training routine, which appends
    supplementary data to each training split in hvhdata
    """
    import architectures as arches

    name = architecture['name']
    archfunc = getattr(arches, architecture['type'])
    arch = lambda input_var=None: archfunc(input_var, **architecture['kwargs'])

    trainer = DefaultTrainer(stopthresh=50, print_interval=50)

    test_errs_ = []                                                                 # storage for test errors to monitor performance
    filename_template = '{archname} bulk train {split_no} split.npz'

    for split in range(5):
        filename = filename_template.format(archname=name, split_no=split)

        print(name, split)
        train_idxs, val_idxs, test_idxs = trainer.get_split_idxs(5, split)

        X, y = [np.concatenate(np.array(hvhdata[i])[train_idxs]) for i in [2, 3]]
        X, y = [np.concatenate([Z, np.concatenate(np.array(data[i]))]) for Z, i in [(X, 2), (y, 3)]]    # add the extra data
        Xv, yv = [np.concatenate(np.array(hvhdata[i])[val_idxs]) for i in [2, 3]]
        Xt, yt = [np.concatenate(np.array(hvhdata[i])[test_idxs]) for i in [2, 3]]

        net = Network(arch)
        net.save_params(os.path.join(paramsdir, filename))
        trainer.train(net, (X, y), (Xv, yv))
        err, acc, bats = trainer.test(net, (Xt, yt))
        test_errs_.append(err / bats)

    return test_errs_

def run_fake_fit(architecture, paramsdir):

    # load archictecture
    import architectures as arches

    name = architecture['name']
    archfunc = getattr(arches, architecture['type'])
    arch = lambda input_var: archfunc(input_var, **architecture['kwargs'])

    # train networks on split data
    trainer = DefaultTrainer(stopthresh=50, print_interval=25)
    net_list = trainer.train_all(architecture=arch, data=fake_data, seed=985227)

    # testing
    idx_test_map = {0: 5, 1: 1, 2: 2, 3: 3, 4: 4}
    hD = hvhdata[0]     # data frame
    hX = hvhdata[2]
    hy = hvhdata[3]
    hP = pd.DataFrame(index=hD.index, columns=list(range(36)))      # predictions DF

    for i, net in enumerate(net_list):
        test_group = idx_test_map[i]
        test_loc = hD['group'] == test_group

        idx_ = test_group - 1
        nll = net.itemized_test_fn(hX[idx_], hy[3][idx_])
        pred = net.output_fn(hX[2][idx_])

        hD.loc[test_loc, 'nll'] = nll
        hP.loc[test_loc] = pred

        fname = '{} {} split fake data.npz'.format('bulk_' + name, i)
        net.save_params(os.path.join(paramsdir_, 'fake_' + name, fname))

    hD.to_csv(os.path.join(resultsdir, 'fake', 'nlls.csv'), index=False)
    hDmeans = hD.pivot_table(index='subject', values='nll')

    return hDmeans.mean()


def run_tuned_experiment(archnames):
    """
    Provided a list of architecture names (as in arch_specs.yaml),
    runs a full fit + fine tuning sequence with run_tuned_fit

    Needs to check if directory exists, as in bulk fit
    Consider making more pass-through parameters
    """

    for name in archnames:
        paramsdir = os.path.join(paramsdir_, name[:-1])
        os.makedirs(paramsdir, exist_ok=True)                                       # create output directory if it does not exist

        architecture = arch_dict[name]

        # print parameter count and architecture specification
        af = getattr(arches, architecture['type'])
        net = af(**architecture['kwargs'])

        print('Param count:', L.count_params(net))
        print(architecture)

        # run tuned fit
        run_tuned_fit(architecture, data, hvhdata, paramsdir=paramsdir, tune=True)

    return None

def run_bulk_experiment(archnames):
    """
    Provided a list of arch names, runs a full fit sequence combining
    all supplementary data with each split of hvhdata
    """
    test_errs = {}
    for name in archnames:
        paramsdir = os.path.join(paramsdir_, 'bulk_' + name[:-1])
        os.makedirs(paramsdir, exist_ok=True)

        architecture = arch_dict[name]

        test_errs[name] = run_bulk_fit(architecture, paramsdir)

    return test_errs

def run_fake_experiment(archnames):
    """
    Runs fit for all archnames by training on fake data
    """

    test_errs = {}
    for name in archnames:
        paramsdir = os.path.join(paramsdir_, 'fake_' + name[:-1])
        os.makedirs(paramsdir, exist_ok=True)

        architecture = arch_dict[name]

        test_errs[name] = run_fake_fit(architecture, paramsdir)

    return test_errs

def main(argv):
    """Pass network name and experiment type on command line"""
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
