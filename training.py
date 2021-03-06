import os
import numpy as np
import theano
import lasagne
import time
from scipy.stats import bayes_mvs
from loading import augment
from network import Network, Autoencoder

L = lasagne.layers
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
                updates=lasagne.updates.adam, update_args={}, seed=None):
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
        self.val_trace = []
        self.train_trace = []
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

            mean_train_err = train_err / train_bats
            mean_val_err = val_err / val_bats
            self.val_trace.append(mean_val_err)

            self.epoch = epoch
            del_val_err = np.diff(self.val_trace)

            if epoch > self.stopthresh:
                if del_val_err[epoch-self.stopthresh:].mean() > 0:
                    print("Abandon ship!")
                    break

            if epoch % self.print_interval == 0:
                print("Epoch {} took {:.3f}s".format(epoch, epoch_dur))
                print("\ttraining loss:\t\t\t{:.4f}".format(mean_train_err))
                print("\tvalidation loss:\t\t{:.4f}".format(mean_val_err))
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

    self.train_all may be further decomposable
    (eg separate "unpack data" function...)
    """

    def get_split_idxs(self, num_splits, split):
        """
        Generates an array for of split indices for training, validation,
        and test sets, then returns training, validation, and test set indices
        for input split.
        """

        split_array = np.tile(np.arange(num_splits), [num_splits, 1])               # stack [0 ... num_splits] x num_splits
        split_array = (split_array + split_array.T) % num_splits                    # add transpose and modulo to rotate each row forward 1

        train_idxs = split_array[split, :-2]                                        # train set idxs in row split, columns until last 2
        val_idxs = split_array[split, -2:-1]                                        # second to last col is validation index
        test_idxs = split_array[split, -1:]                                         # final col is test idx

        return train_idxs, val_idxs, test_idxs


    def run_split(self, architecture, data, split, augment_fn):
        """
        Trains an architecture on a single training/validation/test split
        Data is a tuple such as that returned by loading.default_loader()

        Augmentation can be ignored by passing a a pass-through function as
        augment_fn
        """

        print("\nSplit Number {}".format(split))

        D, groups, Xs, ys, Ss = data                                                # unpack data tuple
        num_splits = len(Xs)                                                        # Xs is a list with number of members = number of groups in data file
        train_idxs, val_idxs, test_idxs = self.get_split_idxs(num_splits, split)    # get indices of splits in each group

        X, y, S = [np.concatenate(np.array(Z)[train_idxs]) for Z in [Xs, ys, Ss]]   # compile testing, validation, and training splits into single arrays
        Xv, yv, Sv = [np.concatenate(np.array(Z)[val_idxs]) for Z in [Xs, ys, Ss]]
        Xt, yt, St = [np.concatenate(np.array(Z)[test_idxs]) for Z in [Xs, ys, Ss]]

        X, y = augment_fn((X, y))                                                   # augment training data
        S = np.concatenate([S, S, S, S])                                            # subjects too

        net = Network(architecture)                                                 # compile network
        self.train(net, training_data=(X, y), validation_data=(Xv, yv))             # train network
        self.test(net, testing_data=(Xt, yt))                                       # test network

        return net


    def train_all(self, architecture, data,
                    seed=None, save_params=False, augment_fn=augment):
        """
        Runs all training splits for a given architecture and caches trained
        networks in a list.
        """

        net_list = []                                                               # initialize list

        if seed:
            np.random.seed(seed)                                                    # set random seed if provided

        starttime = time.time()                                                     # set start time

        num_splits = len(data[2])
        for split in range(num_splits):
            net = self.run_split(architecture, data, split, augment_fn)
            net_list.append(net)

        mvs = bayes_mvs([n.test_err for n in net_list ], alpha=.95)                      # get mean test performance after all splits complete
        time_elapsed = time.time() - starttime                                      # check total elapsed time

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

    def run_split(self, architecture, data, split, seed=None,
                    startparams=None, freeze=True, exclude=[-4]):
        """
        Fine tunes an architecture given an existing set of (trained) weights
        Should be renamed "run_split" to be consistent with above
        """

        if seed:
            np.random.seed(seed)

        D, groups, Xs, ys, Ss = data
        num_splits = len(Xs)
        train_idxs, val_idxs, test_idxs = self.get_split_idxs(num_splits, split)

        X, y, S = [np.concatenate(np.array(Z)[train_idxs]) for Z in [Xs, ys, Ss]]
        Xv, yv, Sv = [np.concatenate(np.array(Z)[val_idxs]) for Z in [Xs, ys, Ss]]
        Xt, yt, St = [np.concatenate(np.array(Z)[test_idxs]) for Z in [Xs, ys, Ss]]
        X, y = augment((X, y))
        S = np.concatenate([S, S, S, S])

        net = Network(architecture)
        if startparams:
            _layers = L.get_all_layers(net.net)
            L.set_all_param_values(_layers, startparams)
            if freeze:
                net.freeze_params(exclude=exclude)

        starttime = time.time()

        self.train(net, training_data=(X, y), validation_data=(Xv, yv))
        self.test(net, testing_data=(Xt, yt))
        time_elapsed = time.time() - starttime

        return net


class AutoencoderTrainer(DefaultTrainer):
    def train_autoencoder(self, net, X):
        X = np.concatenate([X, X[:, :, ::-1, :], X[:, :, :, ::-1], X[:, :, ::-1, ::-1]]) # augment
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)

        Xv = X[idxs[:X.shape[0]//10], :, :, :]
        Xtr = X[idxs[X.shape[0]//10]:, :, :, :]

        training_start = time.time()

        validation_trace = []

        for epoch in range(self.max_epoch):
            training_error = 0
            training_batches = 0
            validation_error = 0
            validation_batches = 0
            epoch_start = time.time()

            for batch in self.iterate_minibatches(Xtr, Xtr, shuffle=True):
                inputs, targets = batch
                training_error += net.ae_train_fn(inputs, targets)
                training_batches += 1

            epoch_duration = time.time() - epoch_start

            for batch in self.iterate_minibatches(Xv, Xv, shuffle=False):
                inputs, targets = batch
                validation_error += net.ae_test_fn(inputs, targets)
                validation_batches += 1

            net.ae_training_error = training_error / training_batches
            net.ae_validation_error = validation_error / validation_batches
            validation_trace.append(net.ae_validation_error)

            if epoch > self.stopthresh:
                if np.mean(validation_trace[epoch-self.stopthresh:]) > 0:
                    print('Abandon ship!')
                    break

            if epoch % self.print_interval == 0:
                print("Epoch {} took {:.3f}s".format(epoch, epoch_duration))
                print("\ttraining loss:\t\t\t{:.4f}".format(net.ae_training_error))
                print("\tvalidation loss:\t\t{:.4f}".format(net.ae_validation_error))
                print("\ttotal time elapsed:\t\t{:.3f}s".format(time.time() - training_start))

        return None

    def run_split(self, net, data, split, augment_fn):
        print('\nSplit Number {}'.format(split))
        D, groups, Xs, ys, Ss = data
        num_splits = len(Xs)
        train_idxs, val_idxs, test_idxs = self.get_split_idxs(num_splits, split)

        X, y, S = [np.concatenate(np.array(Z)[train_idxs]) for Z in [Xs, ys, Ss]]   # compile testing, validation, and training splits into single arrays
        Xv, yv, Sv = [np.concatenate(np.array(Z)[val_idxs]) for Z in [Xs, ys, Ss]]
        Xt, yt, St = [np.concatenate(np.array(Z)[test_idxs]) for Z in [Xs, ys, Ss]]

        X, y = augment_fn((X, y))                                                   # augment training data
        S = np.concatenate([S, S, S, S])

        self.train(net, training_data=(X, y), validation_data=(Xv, yv))
        self.test(net, testing_data=(Xt, yt))

        return net


    def train_all(self, net, data, seed=None, save_params=False, augment_fn=augment):
        net_list = []

        if seed:
            np.random.seed(seed)

        Xs = np.concatenate(data[2])
        start_params = L.get_all_param_values(net.net)            # cache params to avoid recompiling

        num_splits = len(data[2])
        for split in range(num_splits):
            L.set_all_param_values(net.net, start_params)
            net = self.run_split(net, data, split, augment_fn)
            net_list.append(L.get_all_param_values(net.net))      # save params instead of full network to list

        return net_list
