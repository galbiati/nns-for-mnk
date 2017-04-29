import os
import numpy as np
import theano
import lasagne as L
import time
from scipy.stats import bayes_mvs as bmvs
from loading import augment
from network import Network

T = theano.tensor

class Trainer(object):
    """
    Base for subclassing optimizers
    Includes:
        - a function for iterating minibatches
        - a training function that trains a given network on provided training
        and validation data as X, y tuples
        - a test function that tests a given network on provided test data as
        an X, y tuple
    """
    def __init__(self, batchsize=128, stopthresh=100, print_interval=50,
                updates=L.updates.adam, update_args={}, seed=None):
        """
        ToDos:
        - More options?

        Arguments:
        - batchsize: number of examples in each minibatch
        - stopthresh: early stopping threshold. training stops when mean
        gradient of validation error becomes positive over last <stopthresh>
        epochs
        - print_interval: print a small report every <print_interval> epochs
        - updates: reference to updates algorithm, either from lasagne.updates
        or implemented similarly
        - update_args: dictionary of arguments for update algorithm (eg learning
        rate, momentum, etc)
        - seed: random seed for repeating experiment
        """
        self.updates = updates
        self.bs = batchsize
        self.epoch = 0
        self.max_epoch = 5000                   # default: really high
        self.stopthresh = stopthresh
        self.print_interval = print_interval
        self.update_args = update_args
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def train(self, network, training_data, validation_data):
        """
        Training and validation
        It might be better to abstract the training and validation loops into
        their own functions, but not a priority for now
        """
        network.updates = self.updates(network.loss, network.params, **self.update_args)
        X, y = training_data
        Xv, yv = validation_data
        self.train_start = time.time()

        for epoch in range(self.max_epoch):
            train_err = 0
            train_bats = 0
            val_err = 0
            val_acc = 0
            val_bats = 0
            epoch_start = time.time()

            for batch in self.iterate_minibatches(X, y, shuffle=True):
                inputs, targets = batch
                train_err += network.train_fn(inputs, targets)
                train_bats += 1

            epoch_dur = time.time() - epoch_start

            for batch in self.iterate_minibatches(Xv, yv, shuffle=False):
                inputs, targets = batch
                error, accuracy = network.test_fn(inputs, targets)
                val_err += error
                val_acc += accuracy
                val_bats += 1

            network.train_err = train_err / train_bats
            network.val_err = val_err / val_bats
            network.update_traces()

            self.epoch = epoch
            del_val_err = np.diff(network.val_trace)
            if epoch > self.stopthresh:
                if del_val_err[epoch-self.stopthresh:epoch].mean() > 0:
                    print("Abandon ship!")
                    break

            if epoch % self.print_interval == 0:
                print("Epoch {} took {:.3f}s".format(epoch, epoch_dur))
                print("\ttraining loss:\t\t\t{:.4f}".format(train_err/train_bats))
                print("\tvalidation loss:\t\t{:.4f}".format(val_err/val_bats))
                print("\tvalidation accuracy:\t\t{:.2f}%".format(100*val_acc/val_bats))
                print("\ttotal time elapsed:\t\t{:.3f}s".format(time.time() - self.train_start))

        return train_err, val_err

    def test(self, network, testing_data):
        X, y = testing_data
        test_err = 0
        test_acc = 0
        test_bats = 0

        for batch in self.iterate_minibatches(X, y, shuffle=False):
            inputs, targets = batch
            error, accuracy = network.test_fn(inputs, targets)
            test_err += error
            test_acc += accuracy
            test_bats += 1

        network.test_err = test_err/test_bats
        print("\nTEST PERFORMANCE")
        print("\tStopped in epoch:\t\t{}".format(self.epoch))
        print("\tTest loss:\t\t\t{:.4f}".format(test_err/test_bats))
        print("\tTest accuracy:\t\t\t{:.2f}%\n".format(100*test_acc/test_bats))

        return test_err, test_acc, test_bats

    def iterate_minibatches(self, inputs, targets, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            idxs = np.arange(len(inputs))
            np.random.shuffle(idxs)

        for idx in range(0, len(inputs)-self.bs+1, self.bs):
            if shuffle:
                excerpt = idxs[idx:idx+self.bs]
            else:
                excerpt = slice(idx, idx+self.bs)
            yield inputs[excerpt], targets[excerpt]

class DefaultTrainer(Trainer):
    """
    Implements an additional function that does training for all 5 default
    cross-validation splits

    This is meant as a standalone, not for subclassing. But I should consider
    implementing a basic train_all function that does random cv splits rather
    than premade...

    self.train_all may be further decomposable -
    eg separate "unpack data" function...

    Automatic parameter saving to be implemented later...
    """

    def run_split(self, architecture, data, split, augment_fn):
        print("\nSplit Number {}".format(split))
        D, groups, Xs, ys, Ss = data
        num_splits = len(Xs)
        r = np.tile(np.arange(num_splits), [num_splits, 1])
        r = (r + r.T) % num_splits

        train_idxs = r[split, :3] # you've hard-coded 5 splits here accidentally; revisit and fix at some point
        val_idxs = r[split, 3:4]
        test_idxs = r[split, 4:]

        X, y, S = [np.concatenate(np.array(Z)[train_idxs]) for Z in [Xs, ys, Ss]]
        Xv, yv, Sv = [np.concatenate(np.array(Z)[val_idxs]) for Z in [Xs, ys, Ss]]
        Xt, yt, St = [np.concatenate(np.array(Z)[test_idxs]) for Z in [Xs, ys, Ss]]
        X, y = augment_fn((X, y))
        S = np.concatenate([S, S, S, S])
        print(Xt.shape)

        net = Network(architecture)
        self.train(net, training_data=(X, y), validation_data=(Xv, yv))
        self.test(net, testing_data=(Xt, yt))

        return net

    def train_all(self, architecture, data, seed=None, save_params=False, augment_fn=augment):
        net_list = []
        if seed:
            np.random.seed(seed)

        starttime = time.time()
        num_splits = len(data[2])
        for split in range(num_splits):
            net = self.run_split(architecture, data, split, augment_fn)
            net_list.append(net)

        mvs = bmvs([n.test_err for n in net_list ], alpha=.95)
        time_elapsed = time.time() - starttime
        print("\n\nOVERALL RESULTS")
        print("\tAverage NLL:\t\t{:.3f}".format(mvs[0][0]))
        print("\tCred. Interval:\t\t[{:.3f}, {:.3f}]".format(mvs[0][1][0], mvs[0][1][1]))
        print("\tTotal time:\t\t{:.2f}".format(time_elapsed))

        return net_list

class FineTuner(DefaultTrainer):
    """
    Trainer to fine tune networks to individual subjects

    Consider moving freeze, param set functions properly into Network object
    Abstracting split functions and augment in DefaultTrainer would be good too
    """

    def train_all(self, architecture, data, split, seed=None,
                    startparams=None, freeze=True, exclude=[-4]):
        if seed:
            np.random.seed(seed)

        D, groups, Xs, ys, Ss = data
        num_splits = len(Xs)
        r = np.tile(np.arange(num_splits), [num_splits, 1])
        r = (r + r.T) % num_splits

        starttime = time.time()
        net = Network(architecture)
        if startparams:
            _layers = L.layers.get_all_layers(net.net)
            L.layers.set_all_param_values(_layers, startparams)
            if freeze:
                net.freeze_params(exclude=exclude)

        train_idxs = r[split, :3]
        val_idxs = r[split, 3:4]
        test_idxs = r[split, 4:]

        X, y, S = [np.concatenate(np.array(Z)[train_idxs]) for Z in [Xs, ys, Ss]]
        Xv, yv, Sv = [np.concatenate(np.array(Z)[val_idxs]) for Z in [Xs, ys, Ss]]
        Xt, yt, St = [np.concatenate(np.array(Z)[test_idxs]) for Z in [Xs, ys, Ss]]
        X, y = augment((X, y))
        S = np.concatenate([S, S, S, S])
        self.train(net, training_data=(X, y), validation_data=(Xv, yv))
        self.test(net, testing_data=(Xt, yt))
        time_elapsed = time.time() - starttime

        return net

def run_full_fit(architecture, data, hvhdata, paramsdir, tune=True, save=True):
    """
    Runs the full fitting experiment, pretraining on later experiments and testing on first.
    Saves data as it goes to avoid eating memory.
    (SORT OF; break this up into two functions)
    """

    import architectures as arches
    archname = architecture['name']
    archfunc = getattr(arches, architecture['type'])
    arch = lambda input_var=None: archfunc(input_var, **architecture['kwargs'])

    tunekws = {'freeze': True, 'exclude': [-5]}

    # start training
    trainer = DefaultTrainer(stopthresh=50, print_interval=25)
    net_list = trainer.train_all(architecture=arch, data=data, seed=985227)

    # save params
    if save:
        for i, n in enumerate(net_list):
            fname = '{} {} split agg fit exp 1-4'.format(archname, i)
            n.save_params(os.path.join(paramsdir, fname))

    if tune:
        tuner = FineTuner(stopthresh=10)

        for i, n in enumerate(net_list):
            for j in range(5):

                fname = '{} {} agg fit exp 1-4 {} tune fit exp 0'.format(archname, i, j)
                params = L.layers.get_all_param_values(n.net)
                net = tuner.train_all(architecture=arch, data=hvhdata, split=j, startparams=params, **tunekws )

                if save:
                    net.save_params(os.path.join(paramsdir, fname))

    return None
