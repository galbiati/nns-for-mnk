import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd
from util import *
from archs import *
from load_data import *
from network import *

# train NN and save params in output file
# TODO:
#   save params
#   data should OWN vs OPP, not BLACK vs WHITE
#   CV data split should be interleaved-random, not pure random
#   hard-code probability of zero for occupied squares
#
# FUTURE:
#   pretrain with reinforcement learning?

default_data_file = '../../Google Drive/Bas Zahy Gianni - Games/Data/0_hvh/Clean/_summaries/model_input.csv'
param_file = 'param_file.npz'

def CV_train(arch, datafile=default_data_file, splitsize=5, epochs=500, batchsize=500, thresh=25, everyn=100, custom_loaded=None):
    r = np.arange(splitsize)
    r = np.tile(r, [splitsize, 1])
    r = r + r.T
    split_idx = r % splitsize
    CV_nlls = []
    nets = []
    CV_traces = []

    for i in range(splitsize):
        net = MNKNet(arch, datafile)
        if custom_loaded:
            net.data, net.splits, net.Xsplit, net.ysplit, net.Ssplit, net.splitsize = custom_loaded
            print([x.shape for x in net.Xsplit])
        training = split_idx[i, :3]
        validation = split_idx[i, 3]
        test = split_idx[i, 4]

        Xtr = np.concatenate(np.array(net.Xsplit)[training])
        ytr = np.concatenate(np.array(net.ysplit)[training])
        Str = np.concatenate(np.array(net.Ssplit)[training])

        Xva = net.Xsplit[validation]
        yva = net.ysplit[validation]
        Sva = net.Ssplit[validation]

        Xte = net.Xsplit[test]
        yte = net.ysplit[test]
        Ste = net.Ssplit[test]

        nlls, traces = net.train((Xtr, ytr), (Xva, yva), batchsize=batchsize, epochs=epochs, thresh=thresh, everyn=everyn)
        test_nll, test_acc = net.test((Xte, yte), batchsize=batchsize)
        CV_nlls.append(test_nll)
        nets.append(net)
        CV_traces.append(traces)
        
    return CV_nlls, CV_traces, nets

def main():
    CV_train(cnn, splitsize=5)

if __name__ == '__main__':
    main()