import numpy as np
import pandas as pd

def read_datafile(filename):
    """Read a data file into a pandas dataframe"""
    
    columns = ['subject', 'color', 'bp', 'wp', 'zet', 'rt', 'group']
    data = pd.read_csv(filename, names=columns)
    c = data['color'] == 1
    data.loc[c, ['bp', 'wp']] = data.loc[c, ['wp', 'bp']].values
    return data

def X_decoder(x):
    """
    X_decoder transforms board bytestring representations into 4x9 arrays

    Expects an iterable
    Consider reimplementing for use with np.apply_along_axis or similar?
    """

    decoder = lambda bytestring: np.array(list(bytestring)).astype(int).reshape([4,9]) # do this to each element of x
    return np.array(list(map(decoder, x)))                                          # apply decoder

def unpack_data(df):
    """
    Convert dataframe into separate tensors and vectors for respective variables
    """

    bp = X_decoder(df['bp'].values)
    wp = X_decoder(df['wp'].values)
    X = np.zeros([bp.shape[0], 2, bp.shape[1], bp.shape[2]], dtype=np.float32)
    X[:, 0, :, :] = bp
    X[:, 1, :, :] = wp
    y = df['zet'].values.astype(np.int32)
    S = df['subject'].values
    G = df['group'].values - 1
    Np = X.sum(axis=(1, 2, 3))
    return X, y, S, G, Np

def random_split(data, splitsize=5):
    raise NotImplementedError

def augment(D):
    """
    Augment adds reflections to all boards and responses
    D is a tuple with X input tensor and y target vector
    """

    X, y = D
    n = y.shape[0]
    X = np.concatenate([X, X[:, :, :, ::-1], X[:, :, ::-1, :], X[:, :, ::-1, ::-1]])

    _y = np.zeros([n, 36])
    _y[np.arange(n), y] = 1 # convert to one-hot
    _y = _y.reshape([n, 4, 9])
    y = np.concatenate([_y, _y[:, :, ::-1], _y[:, ::-1, :], _y[:, ::-1, ::-1]])
    y = np.where(y.reshape([4*n, 36])==1)[1].astype(np.int32)

    return X, y

def default_loader(filename, subject=None):
    """
    Loads data into 5 cross-validation groups as listed in datafiles
    """

    F = read_datafile(filename)

    if subject is not None:
        F = F.loc[F['subject'] == subject, :]

    X, y, S, G, Np = unpack_data(F)

    groups = [np.where(G==g)[0] for g in np.arange(5)]
    Xsplits = [X[g, :, :, :] for g in groups]
    ysplits = [y[g] for g in groups]
    Ssplits = [S[g] for g in groups]
    Npsplits = [Np[g] for g in groups]

    return F, groups, Xsplits, ysplits, Ssplits
