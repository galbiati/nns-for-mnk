import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

default_data_file = '../../Google Drive/Bas Zahy Gianni - Games/hvh_boards_for_neural_nets.txt' 

def load_dataset(data_file=default_data_file, subject=None):
    columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt']
    datatypes = [('subject', 'i4'), ('color', 'i4'), 
                 ('bp', 'S36'), ('wp', 'S36'), 
                 ('response', 'i4'), ('rt', 'i4')]
    data = np.loadtxt(data_file, delimiter=',', dtype=datatypes)
    data = pd.DataFrame.from_records(data)
    decoder = lambda x: x.decode('utf-8')
    data.loc[:, 'bp'] = data.loc[:, 'bp'].map(decoder)
    data.loc[:, 'wp'] = data.loc[:, 'wp'].map(decoder)
    data = data.reset_index(drop=True)
    if subject is not None:
        data = data.loc[data.subject==subject, :]
    return data

def split_dataset(data, augment=False):
    decoder = lambda x: np.array(list(x)).astype(int).reshape([4,9])
    bp = np.array(list(map(decoder, data.loc[:, 'bp'].values)))
    wp = np.array(list(map(decoder, data.loc[:, 'wp'].values)))
    X = np.zeros([bp.shape[0], 2, bp.shape[1], bp.shape[2]])
    X[:, 0, :, :] = bp
    X[:, 1, :, :] = wp
    y = data.loc[:, 'response'].values

    # get balanced by subject indices

    if augment:
        X = np.concatenate([X, X[:, :, :, ::-1], X[:, :, ::-1, :], X[:, :, ::-1, ::-1]])
        ytemp = np.zeros((y.shape[0], 36))
        ytemp[np.arange(ytemp.shape[0]), y] = 1
        ytemp = ytemp.reshape([ytemp.shape[0], 4, 9])
        y2 = np.concatenate([ytemp, ytemp[:, :, ::-1], ytemp[:, ::-1, :], ytemp[:, ::-1, ::-1]])
        y2 = y2.reshape([y2.shape[0], 36])
        y = np.where(y2==1)[1].astype(np.int32)

    N = X.shape[0]
    random_index = np.random.permutation(np.arange(N))
    X = X[random_index, :, :, :]
    y = y[random_index]
    print(str(N) + ' examples')

    cut = N // 8
    train = slice(0, N-cut*2)
    val = slice(N-cut*2, N-cut)
    test = slice(N-cut, N+1)

    Xtr, ytr = X[train, :, :, :], y[train]
    Xv, yv = X[val, :, :, :], y[val]
    Xt, yt = X[test, :, :, :], y[test]
    
    return Xtr, ytr, Xv, yv, Xt, yt