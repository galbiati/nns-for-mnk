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
        add self.unfreeze(params)
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
        self.value_layer = get_layers(self.net)[-4]
        self.value_prediction = get_output(self.value_layer)
        return None

    def objectives(self):
        """
        Adds loss and accuracy nodes
        """
        self.loss = cross_entropy(self.prediction, self.target_var)
        self.loss = self.loss.mean()
        self.itemized_loss = cross_entropy(self.test_prediction, self.target_var)
        self.test_loss = self.itemized_loss.mean()
        self.test_acc = T.mean(
            T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
            dtype=theano.config.floatX
        )

        return None

    def compile_functions(self):
        """
        Compiles theano functions for computing output, losses, etc
        """
        self.updates = self.update_algo(self.loss, self.params)
        self.output_fn = theano.function([self.input_var], self.test_prediction)
        self.value_fn = theano.function([self.input_var], self.value_prediction)
        self.train_fn = theano.function(
            [self.input_var, self.target_var], self.loss,
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

    def freeze_params(self, exclude=None):
        """
        Sets params to be untrainable
        Excludes layers in optional arg exclude (tuple or list)
        """
        layers = get_layers(self.net)
        num_layers = len(layers)
        exclude = [i if i >= 0 else num_layers + i for i in exclude]

        if exclude is not None:
            layers = [layer for l, layer in enumerate(layers) if not (l in exclude)]

        for layer in layers:
            for param in layer.params:
                layer.params[param].remove('trainable')

        self.params = get_all_params(self.net, trainable=True)

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
