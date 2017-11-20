import numpy as np
import theano
import lasagne

## ALIASES ##
L = lasagne.layers
T = theano.tensor
get_output = L.get_output
get_all_params = L.get_all_params
cross_entropy = lasagne.objectives.categorical_crossentropy
get_layers = L.get_all_layers


class Network(object):
    """
    Wrapper for neural networks for MNK that automates network compilation and
    provides some conveninece functions for freezing, saving, and loading params

    Things to consider doing:
        mod save/load to use named layers
        add self.reinitialize(layers)
    """

    def __init__(self, architecture):
        self.architecture = architecture
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.update_algo = lasagne.updates.adam # just a default
        self.build()
        self.objectives()
        self.compile_functions()
        self.val_trace = np.zeros(500)
        self.train_trace = np.zeros(500)
        self.trace_loc = 0

    def build(self):
        """
        Generates network graph, grabs params and output symbols
        """
        self.net = self.architecture(self.input_var)
        self.prediction = get_output(self.net)
        self.test_prediction = get_output(self.net, deterministic=True)
        self.params = get_all_params(self.net, trainable=True)
        self.filters = [p for p in self.params if 'conv.W' in p.name]
        self.val_weights = [p for p in self.params if 'dense.W' in p.name]
        self.wsum_weights = [p for p in self.params if 'wsum.W' in p.name]
        self.value_layer = get_layers(self.net)[-4]
        self.value_prediction = get_output(self.value_layer)
        return None

    def objectives(self):
        """
        Adds loss and accuracy nodes
        """
        self.loss = cross_entropy(self.prediction, self.target_var)
        self.loss = self.loss.mean()
        # regularizers
        l2 = lambda x: T.sum([T.sum(T.pow(f, 2)) for f in x])
        l1 = lambda x: T.sum([T.sum(T.abs_(w)) for w in x])


        # conv weights should be close to either 0 or 1, so use modified l2
        self.conv_weights_l2 = .5 * l2(self.filters) + .5 * l2([f - 1 for f in self.filters])
        # filters should be sparse-ish
        self.conv_weights_size_l2 = T.sum(T.pow([T.sum(f) for f in self.filters], 2))
        # value weights should be sparse
        self.val_weights_l1 = l1(self.val_weights)
        # sum weights should be smallish
        self.wsum_weights_l2 = T.sum(T.pow(self.wsum_weights, 2))

        self.reg_terms = self.conv_weights_l2 + .1 * self.conv_weights_size_l2
        self.reg_terms += self.val_weights_l1 + .1 * self.wsum_weights_l2

        self.regularized_loss = self.loss + self.reg_terms
        self.itemized_loss = cross_entropy(self.test_prediction, self.target_var)
        self.test_loss = self.itemized_loss.mean()
        self.test_acc = T.mean(
            T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
            dtype=theano.config.floatX
        )
        self.updates = self.update_algo(self.loss, self.params)

        return None

    def compile_functions(self):
        """
        Compiles theano functions for computing output, losses, etc
        """
        self.output_fn = theano.function([self.input_var], self.test_prediction)
        self.value_fn = theano.function([self.input_var], self.value_prediction)
        self.train_fn = theano.function(
            [self.input_var, self.target_var], self.regularized_loss,
            updates=self.updates
        )
        self.test_fn = theano.function(
            [self.input_var, self.target_var],
            [self.test_loss, self.test_acc]
        )
        self.itemized_test_fn = theano.function(
            [self.input_var, self.target_var],
            self.itemized_loss
        )

        return None

    def update_traces(self):
        """
        Saves traces for plotting
        """
        self.val_trace[self.trace_loc] = self.val_err
        self.train_trace[self.trace_loc] = self.train_err
        self.trace_loc += 1 # so hacky
        return None

    def freeze_params(self, net=None, exclude=None):
        """
        Sets params to be untrainable
        Excludes layers in optional arg exclude (tuple or list)
        """
        if net is None:
            net = self.net

        layers = get_layers(net)
        num_layers = len(layers)
        exclude = [i if i >= 0 else num_layers + i for i in exclude]

        if exclude is not None:
            layers = [layer for l, layer in enumerate(layers) if not (l in exclude)]

        for layer in layers:
            for param in layer.params:
                layer.params[param].remove('trainable')

        self.params = get_all_params(net, trainable=True)       # CAUTION: needs rewritten to not throw errors as autoencoders develop

        return None

    def unfreeze_params(self):
        """
        Sets all parameters back to trainable
        """
        for layer in L.get_all_layers(self.net):
            for param in layer.params:
                layer.params[param].add('trainable')
        self.params = L.get_all_params(self.net, trainable=True)
        return None

    def save_params(self, param_file):
        """
        Save parameters for reuse later
        """
        all_params = L.get_all_param_values(self.net)
        np.savez(param_file, *all_params)
        return None

    def load_params(self, paramsfile):
        """
        Loads parameters from npz files
        """
        with np.load(paramsfile) as loaded:
            params_list = [(i[0], i[1]) for i in loaded.items()]
            params_order = np.array([i[0][4:6] for i in params_list]).astype(int)
            params_list = [params_list[i] for i in params_order.argsort()]
            L.set_all_param_values(self.net, [i[1] for i in params_list])

        return None

class Autoencoder(Network):
    """
    Wrapper for training and testing transfer learning with an autoencoder.
    Almost as cool as it sounds.

    Later, use super() to cut down bloat inside functions
    """

    def __init__(self, architecture):
        self.architecture = architecture
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.ae_target_var = T.tensor4('ae inputs')
        self.update_algo = lasagne.updates.adam
        self.val_trace = []
        self.train_trace = []
        self.build()
        self.objectives()
        self.compile_functions()

    def build(self):
        """Generates graph, caches params, output symbols"""
        self.autoencoder, self.value_layer, self.net = self.architecture(self.input_var)
        self.prediction = get_output(self.net)
        self.test_prediction = get_output(self.net, deterministic=True)
        self.value_prediction = get_output(self.value_layer)

        self.image = get_output(self.autoencoder)
        self.test_image = get_output(self.autoencoder, deterministic=True)
        self.params = get_all_params(self.net)
        self.ae_params = get_all_params(self.autoencoder)
        return None

    def objectives(self):
        """Loss functions, etc"""
        self.loss = cross_entropy(self.prediction, self.target_var).mean()
        self.itemized_test_loss = cross_entropy(self.test_prediction, self.target_var)
        self.test_loss = self.itemized_test_loss.mean()
        self.test_acc = T.mean(
            T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
            dtype=theano.config.floatX
        )

        self.updates = self.update_algo(self.loss, self.params)

        self.ae_loss = T.mean((self.ae_target_var - self.image)**2, dtype=theano.config.floatX)
        self.ae_test_loss = T.mean((self.ae_target_var - self.test_image)**2, dtype=theano.config.floatX)
        self.ae_updates = self.update_algo(self.ae_loss, self.ae_params)

        return None

    def compile_functions(self):
        """Compile theano functions"""
        self.output_fn = theano.function([self.input_var], self.test_prediction)
        self.value_fn = theano.function([self.input_var], self.value_prediction)
        self.train_fn = theano.function(
            [self.input_var, self.target_var],
            self.loss,
            updates = self.updates
        )

        self.test_fn = theano.function(
            [self.input_var, self.target_var],
            [self.test_loss, self.test_acc]
        )

        self.itemized_test_fn = theano.function(
            [self.input_var, self.target_var],
            self.itemized_test_loss
        )

        self.ae_output_fn = theano.function([self.input_var], self.test_image)
        self.ae_train_fn = theano.function(
            [self.input_var, self.ae_target_var],
            self.ae_loss,
            updates=self.ae_updates
        )

        self.ae_test_fn = theano.function(
            [self.input_var, self.ae_target_var],
            self.ae_test_loss
        )

        return None
