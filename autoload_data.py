import os
import yaml
import numpy as np
import loading

headdir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games')
paramsdir_ = os.path.join(headdir, 'Analysis/0_hvh/Params/nnets/')
datadir = os.path.join(headdir, 'Data/model input')
resultsdir = os.path.join(headdir, 'Analysis/0_hvh/Loglik/nnets')

data = loading.default_loader(os.path.join(datadir, '1-4 (no computer).csv'))
fake_data = loading.default_loader(os.path.join(datadir, 'fake news (with groups).csv'))
hvhdata = loading.default_loader(os.path.join(datadir, '0 (with groups).csv'))
df = hvhdata[0]
Xs = np.concatenate(hvhdata[2])
ys = np.concatenate(hvhdata[3])
Ss = np.concatenate(hvhdata[4])

defmod = np.loadtxt(os.path.expanduser('~/Downloads/loglik_hvh_final.txt')).reshape([40, 5])

with open('arch_specs.yaml') as archfile:
    arch_dict = yaml.load(archfile)
