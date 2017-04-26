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

def main():
    theano.gpuarray.use("cuda")
    paramsdir = os.path.abspath('/scratch/gvg218/params_archive')

    with open('arch_specs.yaml') as archfile:
        arch_dict = yaml.load(archfile)

    for name, architecture in arch_dict.items():
        print(architecture)
        run_full_fit(architecture, data=data, hvhdata=hvhdata, tune=True)


if __name__ == '__main__':
    main()
