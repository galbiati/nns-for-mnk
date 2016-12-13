import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd
from util import *
from archs import *
from load_data import *

class MNKNet():
    def __init__(self, arch, datafile):
        self.datafile = datafile
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

    def load_data(self):
        print("Loading data...")
        self.data = load_dataset(self.datafile)
        self.splits, self.Xsplit, self.ysplit, self.Ssplit, self.splitsize = split_dataset(self.data)
        return None

    def train(self, TRAINING, VALIDATION, epochs=500, batchsize=500, augmented=True, everyn=150, thresh=25):
        if augmented:
            Xtr, ytr = augment(TRAINING)
            print('Ntrials =', len(ytr))
        else:
            Xtr, ytr = TRAINING
        Xva, yva = VALIDATION
        tr_nll_trace = np.zeros(epochs)
        val_nll_trace = np.zeros(epochs)
        val_acc_trace = np.zeros(epochs)
        traces = (tr_nll_trace, val_nll_trace, val_acc_trace)

        self.last_epoch = 0
        tr_start = time.time()

        for epoch in range(epochs):
            tr_err = 0
            tr_bats = 0
            epoch_start = time.time()

            for bat in iterate_minibatches(Xtr, ytr, batchsize=batchsize, shuffle=True):
                inputs, targets = bat
                tr_err += self.train_fn(inputs, targets)
                tr_bats += 1

            val_err = 0
            val_acc = 0
            val_bats = 0

            for bat in iterate_minibatches(Xva, yva, batchsize=batchsize, shuffle=False):
                inputs, targets = bat
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_bats += 1

            tr_nll_trace[epoch] = tr_err / tr_bats
            val_nll_trace[epoch] = val_err / val_bats
            val_acc_trace[epoch] = val_acc / val_bats * 100

            # early stop
            self.last_epoch = epoch
            delta_val_nll = np.diff(val_nll_trace)
            if epoch > 25:
                if delta_val_nll[epoch-thresh:epoch].mean() > 0:
                    print('Validation error stopped decreasing...')
                    print('Abandon ship!')
                    break
            if epoch % everyn == 0:
                print("Epoch {} of {} took {:.3f}s".format(
                        epoch+1, epochs, time.time() - epoch_start))
                print("  training loss:\t\t{:.4f}".format(tr_err / tr_bats))
                print("  validation loss:\t\t{:.4f}".format(val_err / val_bats))
                print("  validation accuracy:\t\t{:.2f}".format(val_acc / val_bats * 100))
                print("  total time elapsed:\t\t{:.3f}s".format(time.time() - tr_start))

        return (tr_err, val_err), (tr_nll_trace, val_nll_trace, val_acc_trace)

    def test(self, TEST, batchsize=500):
        Xte, yte = TEST
        self.test_nll = 0
        self.test_acc = 0
        test_bats = 0

        for bat in iterate_minibatches(Xte, yte, batchsize, shuffle=False):
            inputs, targets = bat
            err, acc = self.val_fn(inputs, targets)
            self.test_nll += err
            self.test_acc += acc
            test_bats += 1

        print("Final results:")
        print("  Stopped in epoch:\t\t\t{}".format(self.last_epoch+1))
        print("  test loss:\t\t\t{:.6f}".format(self.test_nll / test_bats))
        print("  test accuracy:\t\t{:.2f} %".format(self.test_acc / test_bats * 100))

        return self.test_nll/test_bats, self.test_acc / test_bats

    def save_params(self, param_file):
        self.param_vals = lasagne.layers.get_all_param_values(self.network)
        np.savez(param_file, *self.param_vals)
        return None

    def load_params(self, param_file):
        self.param_vals = np.load(param_file)
        lasagne.layers.set_all_param_values(self.network, [pars[arr] for arr in pars.files])
        return None
