import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def load_dataset(data_file):
    columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt']
    datatypes = [('subject', 'i4'), ('color', 'i4'), 
                 ('bp', 'S36'), ('wp', 'S36'), 
                 ('response', 'i4'), ('rt', 'i4')]
    data = np.loadtxt(data_file, delimiter=',', dtype=datatypes)
    data = pd.DataFrame.from_records(data)
    decoder = lambda x: x.decode('utf-8')
    data.loc[:, 'bp'] = data.loc[:, 'bp'].map(decoder)
    data.loc[:, 'wp'] = data.loc[:, 'wp'].map(decoder)
    data.loc[data.color==1, ['bp', 'wp']] = data.loc[data.color==1, ['wp', 'bp']].values # swap so always own vs opp
    data = data.reset_index(drop=True)
    return data

def shape_dataset(data):
    decoder = lambda x: np.array(list(x)).astype(int).reshape([4,9])
    bp = np.array(list(map(decoder, data.loc[:, 'bp'].values)))
    wp = np.array(list(map(decoder, data.loc[:, 'wp'].values)))
    X = np.zeros([bp.shape[0], 2, bp.shape[1], bp.shape[2]])
    X[:, 0, :, :] = bp
    X[:, 1, :, :] = wp
    y = data.loc[:, 'response'].values
    S = data.loc[:, 'subject'].values
    return X, y, S

def split_dataset(data, splitsize=5):

    X, y, S = shape_dataset(data)
    splits = [[] for s in range(splitsize)]

    for subject in data.subject.unique():
        subject_idx = np.random.permutation(np.where(S==subject)[0])
        N_moves = subject_idx.shape[0]
        cut_size = N_moves//splitsize 

        for s in range(splitsize-1):
            splits[s].append(subject_idx[slice(s*cut_size, (s+1)*cut_size)])
        splits[-1].append(subject_idx[((splitsize-1)*cut_size):])
    splits = [np.concatenate(s) for s in splits]
    Xsplits = [X[s, :, :, :] for s in splits]
    ysplits = [y[s] for s in splits]
    Ssplits = [S[s] for s in splits]

    return splits, Xsplits, ysplits, Ssplits, splitsize

def augment(D):
    X, y = D
    X = np.concatenate([X, X[:, :, :, ::-1], X[:, :, ::-1, :], X[:, :, ::-1, ::-1]])
    # the below is probably wasteful. However, since augment isn't called that often, not a priority fix.
    ytemp = np.zeros((y.shape[0], 36))
    ytemp[np.arange(ytemp.shape[0]), y] = 1
    ytemp = ytemp.reshape([ytemp.shape[0], 4, 9])
    y2 = np.concatenate([ytemp, ytemp[:, :, ::-1], ytemp[:, ::-1, :], ytemp[:, ::-1, ::-1]])
    y2 = y2.reshape([y2.shape[0], 36])
    y = np.where(y2==1)[1].astype(np.int32)

    return X, y