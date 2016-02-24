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

# train NN and save params in output file
# TODO:
#   save params
#   data should OWN vs OPP, not BLACK vs WHITE
#   CV data split should be interleaved-random, not pure random
#   hard-code probability of zero for occupied squares
#
# FUTURE:
#   pretrain with reinforcement learning?

default_data_file = '../../Google Drive/Bas Zahy Gianni - Games/hvh_boards_for_neural_nets.txt'
param_file = 'param_file.npz'

class MNKNet():
    def __init__(self, arch):
        self.arch = arch
        self.build_net(self.arch)
        self.load_data()

    def build_net(self, arch):
        print("Compiling Theano expressions...")
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.network = self.arch(self.input_var)
        self.prediction = lasagne.layers.get_output(self.network)
        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var)
        self.loss = self.loss.mean()
        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target_var)
        self.test_loss = self.test_loss.mean()
        self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
                                dtype=theano.config.floatX)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(self.loss, self.params, 
                                                                learning_rate=.01, momentum=.9)
        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)
        self.val_fn = theano.function([self.input_var, self.target_var], [self.test_loss, self.test_acc])
        self.output_fn = theano.function([self.input_var], self.test_prediction) 
        return None

    def load_data(self, datafile=default_data_file):
        print("Loading data...")
        self.data = load_dataset(datafile)
        self.splits, self.subjects, self.raw = split_dataset(self.data, augment=True)
        self.Xtr, self.ytr, self.Xv, self.yv, self.Xt, self.yt = self.splits
        self.S, self.trainset, self.valset, self.testset = self.subjects
        self.X, self.y = self.raw
        return None

    def train(self, epochs=500, batchsize=500):
        print('Start training...')
        # values for performance plotting
        self.tr_nll = np.zeros(epochs)
        self.val_nll = np.zeros(epochs)
        self.val_acc = np.zeros(epochs)
        self.delta_val_nll = np.diff(self.val_nll)

        self.last_epoch = 0
        tr_start = time.time()

        for epoch in range(epochs):
            tr_err = 0
            tr_bats = 0
            epoch_start = time.time()

            for bat in iterate_minibatches(self.Xtr, self.ytr, batchsize=batchsize, shuffle=True):
                inputs, targets = bat
                tr_err += self.train_fn(inputs, targets)
                tr_bats += 1

            val_err = 0
            val_acc = 0
            val_bats = 0

            for bat in iterate_minibatches(self.Xv, self.yv, batchsize=batchsize, shuffle=False):
                inputs, targets = bat
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_bats += 1

            # early stopping
            self.tr_nll[epoch] = tr_err
            self.val_nll[epoch] = val_err
            self.val_acc[epoch] = val_acc
            self.delta_val_nll = np.diff(self.val_nll)
            self.last_epoch = epoch
            if epoch > 25:
                if self.delta_val_nll[epoch-25:epoch].mean() > 0:
                    print('Validation error stopped decreasing...')
                    print('Abandon ship!')
                    break

            print_network_status(epoch, epochs, epoch_start, tr_start,
                                tr_err, tr_bats, val_err, val_acc, val_bats, tr_start, everyn=50)

        return None

    def test(self, batchsize=500):
        self.test_nll = 0
        self.test_acc = 0
        test_bats = 0

        for bat in iterate_minibatches(self.Xt, self.yt, batchsize, shuffle=False):
            inputs, targets = bat
            err, acc = self.val_fn(inputs, targets)
            self.test_nll += err
            self.test_acc += acc
            test_bats += 1

        print("Final results:")
        print("  Stopped in epoch:\t\t\t{}".format(self.last_epoch+1))
        print("  test loss:\t\t\t{:.6f}".format(self.test_nll / test_bats))
        print("  test accuracy:\t\t{:.2f} %".format(self.test_acc / test_bats * 100))

    def save_params(self):
        self.param_vals = lasagne.layers.get_all_param_values(self.network)
        np.savez(param_file, *self.param_vals)
        return None

    def load_params(self):
        self.param_vals = np.load(param_file)
        lasagne.layers.set_all_param_values(self.network, [pars[arr] for arr in pars.files])
        return None


def main():
    net = MNKNet(cnn5)
    net.train(epochs=5000)
    net.test()

if __name__ == '__main__':
    main()