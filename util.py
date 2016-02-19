import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs)-batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batchsize]
        else:
            excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]

def print_network_status(epoch, epochs, epoch_start, training_start,
                        tr_err, tr_bat, val_err, val_acc, val_bats, tr_start, everyn=10):
    if epoch % everyn == 0: 
        print("Epoch {} of {} took {:.3f}s".format(
                epoch+1, epochs, time.time() - epoch_start))
        print("  training loss:\t\t{:.4f}".format(tr_err / tr_bat))
        print("  validation loss:\t\t{:.4f}".format(val_err / val_bats))
        print("  validation accuracy:\t\t{:.2f}".format(val_acc / val_bats * 100))
        print("  total time elapsed:\t\t{:.3f}s".format(time.time() - training_start))