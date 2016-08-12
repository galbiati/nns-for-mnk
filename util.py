import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def iterate_minibatches(inputs, targets, batchsize=500, shuffle=False):
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

def save_appended_data(nets, filename):
    splitsize = 5
    r = np.arange(splitsize)
    r = np.tile(r, [splitsize, 1])
    r = r + r.T
    split_idx = r % splitsize

    L = []
    idx = nets[0].splits
    data = nets[1].data.copy()
    data.loc[:, 'cnn_nll'] = np.nan
    positions = list(map(lambda x: 'cnn_' + str(x), range(36)))
    for p in positions:
        data.loc[:, p] = np.nan

    for n, t in enumerate([4, 0, 1, 2, 3]):
        net = nets[n]
        X, y = net.Xsplit[t], net.ysplit[t]
        L.append([net.val_fn(X[np.newaxis, i], 
                             y[np.newaxis, i]) 
                  for i in range(X.shape[0])])
        data.loc[idx[t], 'cnn_nll'] = np.array(L[n])[:, 0]
        data.loc[idx[t], positions] = net.output_fn(X)

    piece_counter = lambda x: np.sum(list(map(int, list(x))))
    data.loc[:, 'n_pieces'] = data.bp.map(piece_counter) + data.wp.map(piece_counter)

    data = data.loc[:, ['subject', 'color', 'bp', 'wp', 
                        'response', 'rt', 'splitg', 'cnn_nll',
                        'n_pieces'] + positions]
    data.loc[:, 'prediction'] = np.argmax(data.loc[:, positions].values, axis=1)
    
    data.to_csv('./results/appended_data_' + filename+ '.csv', sep=',', index=False)
    return None