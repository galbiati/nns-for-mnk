import os
import yaml
import lasagne
import pandas as pd
import numpy as np

from network import Network
import architectures as arches

L = lasagne.layers

# loads data with names according to autoload_data.py
from autoload_data import *

# load specs for all networks
with open('arch_specs.yaml') as archfile:
    arch_dict = yaml.load(archfile)

### Compiling results from saved parameters ###

def compute_pretrained_results(net, archname, idx, test_data, fake=False):
    """
    Compute pre-tuning results for a given arch/network on appropriate test data
    """
    Xt, yt = test_data

    if fake:
        fname = '{} {} split fake data.npz'
        fname = fname.format('fake_' + archname, idx)
        paramsdir = os.path.join(paramsdir_, 'fake_' + archname)

    else:
        fname = '{} {} split agg fit exp 1-4.npz'
        fname = fname.format(archname.replace('_', ' '), idx)
        paramsdir = os.path.join(paramsdir_, archname[:-1])

    results_df = pd.DataFrame(index=np.arange(Xt.shape[0]), columns=[idx])

    net.load_params(os.path.join(paramsdir, fname))

    nlls = net.itemized_test_fn(Xt, yt)
    predictions = net.output_fn(Xt)
    results_df[idx] = nlls

    return results_df, predictions


def compute_tuned_results(net, archname, idx, test_idx, test_data, df):
    """
    Compute post-tuning results for a given architecture/network on appropriate
        test data
    """
    Xt, yt = test_data
    group_idx = (test_idx - 1) % 5 # fix eventually to take df/groupidx/selection passed independently?
    selection = df.loc[df['group']==(group_idx+1)].index.values
    results_df = pd.DataFrame(index=np.arange(Xt.shape[0]), columns=[idx])
    predictions_df = pd.DataFrame(index=selection, columns=np.arange(36))

    fname = '{} {} agg fit exp 1-4 {} tune fit exp 0.npz'
    fname = fname.format(archname.replace('_', ' '), idx, test_idx)

    net.load_params(os.path.join(paramsdir_, archname[:-1], fname))

    nlls = net.itemized_test_fn(Xt[selection, :, :, :], yt[selection])
    predictions = net.output_fn(Xt[selection, :, :, :])
    predictions_df.loc[selection, :] = predictions
    results_df.loc[selection, idx] = nlls
    return results_df, predictions_df


def compute_net_results(net, archname, test_data, df):
    """
    For a given network, test on appropriate test data and return dataframes
    with results and predictions (named obviously)
    """
    pretrain_results = []
    pretrain_predictions = []
    tune_results = []
    tune_predictions = []

    for idx in range(5):
        results_df, predictions_df = compute_pretrained_results(net, archname, idx, test_data)
        pretrain_results.append(results_df)
        pretrain_predictions.append(predictions_df)

    pretrain_results = pd.concat(pretrain_results, axis=1)

    for idx in range(5):
        for test_idx in range(5):
            results_df, predictions_df  = compute_tuned_results(net, archname, idx, test_idx, test_data, df)
            tune_results.append(results_df)
            tune_predictions.append(predictions_df)

    tune_results = pd.concat(tune_results, axis=1, join='inner').stack().unstack()

    return pretrain_results, pretrain_predictions, tune_results, tune_predictions


def rehydrate(verbose=False):
    """
    Recompile networks, load params, and run on appropriate test data

    outputs dictionaries with keys = names of architectures as in arch_specs.yaml

    outputs:
    PTx is pretrained on bulk but untuned
    Tx is tuned
    xR is results (nlls)
    xP is predictions (distributions)

    param_counts is what it sounds like.
    """
    Xt, yt, _, _, _ = loading.unpack_data(df)       # get Xs and ys

    PTR = {}                # results and predictions holders
    TR = {}
    PTP = {}
    TP = {}
    param_counts = {}       # counter of parameters per net

    for archname in arch_dict.keys():
        # for each network
        arch_dir = archname[:-1]        # get directory

        if arch_dir not in os.listdir(paramsdir_):
            # if it doesn't exist
            print("{} not started".format(archname[:-1]))       # alert us
            continue

        files = os.listdir(os.path.join(paramsdir_, arch_dir))
        if not any(archname.replace('_', ' ') in f for f in files):
            # if a network doesn't have a full set of parameter fits
            print("{} not completed".format(archname))      # let us know
            continue

        if verbose:
            print(archname)
        arch = arch_dict[archname]
        af = getattr(arches, arch['type'])
        arch_func = lambda input_var: af(input_var, **arch['kwargs'])
        net = Network(arch_func)        # compile network from specs in arch_specs.yaml

        param_counts[archname] = L.count_params(net.net)        # count the params
        pretrain_R, pretrain_P, tune_R, tune_P = compute_net_results(net, archname, (Xt, yt), df)

        PTR[archname] = pretrain_R      # insert results into respective holders
        TR[archname] = tune_R
        PTP[archname] = pretrain_P
        TP[archname] = tune_P

        pretrain_R.to_csv(os.path.join(resultsdir, 'pretrain {}.csv'.format(archname)))     # save results into respective directories
        tune_R.to_csv(os.path.join(resultsdir, 'train {}.csv'.format(archname)))

    return PTR, TR, PTP, TP, param_counts


### Statistics and summaries ###

def entropy_zets(zets):
    """Shannon entropy"""
    z = np.histogram(zets, bins=np.arange(37), normed=True)[0]
    z = z[z > 0]
    return -(z * np.log2(z)).sum()


def count_pieces(row):
    """Counts pieces in a given binstring state representation"""
    bp, wp = row[['bp', 'wp']]
    n_bp = np.array(list(bp)).astype(int).sum()
    n_wp = np.array(list(wp)).astype(int).sum()

    return n_bp + n_wp

def aggregate_results(PTR, TR, param_counts):
    """
    Args:
    PTR, TR, are dictionaries of results dataframes with archnames as keys,
        archnames as in arch_specs.yaml and PTR/TR as produced by rehydrate()
    param_counts also keyed by archnames and as produced by rehydrate; contains
        parameter counts

    outputs:
    F is a dataframe containing individual networks in rows with mean
        performances in columns

    """
    pc_series = pd.Series(param_counts)
    pc_series.to_csv(os.path.join(resultsdir, 'params per net.csv'))
    F = pd.DataFrame(index=np.arange(len(pc_series.index)), columns=['net name'])
    F['net name'] = pc_series.index
    F['num params'] = pc_series.values

    for k, v in PTR.items():
        idx = F['net name'] == k
        v['subject'] = Ss
        v['mean'] = v.mean()
        F.loc[idx, 'pretrained'] = v['mean'].mean()
        pvt = v.pivot_table(index='subject', values='mean', aggfunc=np.mean).mean().values
        F.loc[idx, 'pretrained subject'] = pvt

    for k, v in TR.items():
        idx = F['net name'] == k
        v['subject'] = Ss
        v['mean'] = v.mean()
        F.loc[idx, 'tuned'] = v['mean'].mean()
        pvt = v.pivot_table(index='subject', values='mean', aggfunc=np.mean).mean().values
        F.loc[idx, 'tuned subject'] = pvt

    F['tuning improvement'] = -(F['tuned'] - F['pretrained'])

    return F

def error_per_piece(archname, resultdict, datadf):
    """
    Args:
    archname is name of architecture as in arch_specs.yaml
    resultdict should be a dictionary of predictions where keys = archnames
    datadf is the original data containing positions, subjects, moves, etc

    Outputs:
    preds is a list of dataframes containing predictions for reach observation
        in datadf and training split for archname
    per_pieces is a df with the mean chance-relative nll for each subject and
        number of pieces in position

    """

    chancenll = lambda x: -np.log(1/(36-x))
    cross_entropy = lambda row: -np.log(row[int(row['zet'])])
    preds = [pd.concat(resultdict[archname][i:i+5]).sort_index() for i in range(5)]

    for pred in preds:
        pred['subject'] = df['subject']
        pred['zet'] = df['zet']
        pred['num pieces'] = df['np']
        pred['rt'] = df['rt']
        pred['chance nll'] = chancenll(df['np'].values)
        pred['error'] = pred.apply(cross_entropy, axis=1)
        pred['relative error'] = pred['chance nll'] - pred['error']

    per_pieces = pd.concat([
        p.pivot_table(index='subject', values='relative error', columns='num pieces').mean(axis=0)
        for p in preds
    ], axis=1)

    return preds, per_pieces
