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

def split_dataset(data, splitsize=8, augment=False):
    X, y, S = shape_dataset(data)    

    if augment:
        X = np.concatenate([X, X[:, :, :, ::-1], X[:, :, ::-1, :], X[:, :, ::-1, ::-1]])
        ytemp = np.zeros((y.shape[0], 36))
        ytemp[np.arange(ytemp.shape[0]), y] = 1
        ytemp = ytemp.reshape([ytemp.shape[0], 4, 9])
        y2 = np.concatenate([ytemp, ytemp[:, :, ::-1], ytemp[:, ::-1, :], ytemp[:, ::-1, ::-1]])
        y2 = y2.reshape([y2.shape[0], 36])
        y = np.where(y2==1)[1].astype(np.int32)
        S = np.concatenate([S, S, S, S])

    train = []
    val = []
    test = []
    for s in data.subject.unique():
        # for each subject, get training, validation, and test indices
        si = np.where(S==s)[0]
        Nsi = si.shape[0]
        cut = Nsi // splitsize
        r_sec = np.random.permutation(si)
        train.append(r_sec[slice(0, Nsi-cut*2)])        
        val.append(r_sec[slice(Nsi-cut*2, Nsi-cut)])
        test.append(r_sec[slice(Nsi-cut, Nsi+1)])


    _train = np.concatenate(train)
    _val = np.concatenate(val)
    _test = np.concatenate(test)

    Xtr, ytr = X[_train, :, :, :], y[_train]
    Xv, yv = X[_val, :, :, :], y[_val]
    Xt, yt = X[_test, :, :, :], y[_test]
    
    splits = (Xtr, ytr, Xv, yv, Xt, yt)
    subjects = (S, train, val, test)
    raw = (X, y)

    return splits, subjects, raw

# write new data loader:
# for each subject, divide data into train/test/validate
# then concatenate and store indices for later reference