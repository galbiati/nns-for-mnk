import os
import yaml
import lasagne
import pandas as pd
import numpy as np

from network import Network
import architectures as arches

L = lasagne.layers

headdir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games')
paramsdir_ = os.path.join(headdir, 'Analysis/0_hvh/Params/nnets/')

with open('arch_specs.yaml') as archfile:
    arch_dict = yaml.load(archfile)

def unfreeze(net):
    # move this function onto Network class!
    for layer in L.get_all_layers(net.net):
        for param in layer.params:
            layer.params[param].add('trainable')
    net.params = L.get_all_params(net.net, trainable=True)

    return None

def compute_pretrained_results(archname, idx, test_data, fake=False):
    Xt, yt = test_data
    arch = arch_dict[archname]
    af = getattr(arches, arch['type'])
    arch_func = lambda input_var: af(input_var, **arch['kwargs'])

    if fake:
        fname = '{} {} split fake data.npz'
        fname = fname.format('fake_' + archname, idx)
        paramsdir = os.path.join(paramsdir_, 'fake_' + archname)

    else:
        fname = '{} {} split agg fit exp 1-4.npz'
        fname = fname.format(archname.replace('_', ' '), idx)
        paramsdir = os.path.join(paramsdir_, archname[:-1])

    results_df = pd.DataFrame(index=np.arange(Xt.shape[0]), columns=[idx])

    net = Network(arch_func)

    net.load_params(os.path.join(paramsdir, fname))

    nlls = net.itemized_test_fn(Xt, yt)
    predictions = net.output_fn(Xt)
    results_df[idx] = nlls

    n_params = L.count_params(net.net)
    return results_df, predictions, n_params

def compute_tuned_results(archname, idx, test_idx, test_data, df):
    Xt, yt = test_data
    group_idx = (test_idx - 1) % 5 # fix eventually to take df/groupidx/selection passed independently?
    selection = df.loc[df['group']==(group_idx+1)].index.values
    results_df = pd.DataFrame(index=np.arange(Xt.shape[0]), columns=[idx])
    predictions_df = pd.DataFrame(index=selection, columns=np.arange(36))

    arch = arch_dict[archname]
    af = getattr(arches, arch['type'])
    arch_func = lambda input_var: af(input_var, **arch['kwargs'])
    fname = '{} {} agg fit exp 1-4 {} tune fit exp 0.npz'
    fname = fname.format(archname.replace('_', ' '), idx, test_idx)

    net = Network(arch_func)
    net.load_params(os.path.join(paramsdir_, archname[:-1], fname))

    nlls = net.itemized_test_fn(Xt[selection, :, :, :], yt[selection])
    predictions = net.output_fn(Xt[selection, :, :, :])
    predictions_df.loc[selection, :] = predictions
    results_df.loc[selection, idx] = nlls
    return results_df, predictions_df
