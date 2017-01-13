import numpy as np
import theano
import lasagne as L

T = theano.tensor
get_output = L.layers.get_output
get_all_params = L.layers.get_all_params
cross_entropy = L.objectives.categorical_crossentropy
get_layers = L.layers.get_all_layers

class Network(object):
    """
    Base for subclassing networks for MNK
    """
    def __init__(self, architecture):
        self.architecture = architecture
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.update_algo = L.updates.adam # just a default
        self.build()
        self.objectives()
        self.compile_functions()
        self.val_trace = np.zeros(5000)
        self.train_trace = np.zeros(5000)
        self.trace_loc = 0

    def build(self):
        self.net = self.architecture(self.input_var)
        self.prediction = get_output(self.net)
        self.test_prediction = get_output(self.net, deterministic=True)
        self.params = get_all_params(self.net, trainable=True)
        self.value_layer = get_layers(self.net)[-4]
        self.value_prediction = get_output(self.value_layer)
        return None

    def objectives(self):
        self.loss = cross_entropy(self.prediction, self.target_var)
        self.loss = self.loss.mean()
        self.test_loss = cross_entropy(self.test_prediction, self.target_var)
        self.test_loss = self.test_loss.mean()
        self.test_acc = T.mean(
            T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
            dtype=theano.config.floatX
        )

        return None

    def compile_functions(self):
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

        return None

    def update_traces(self):
        """
        Saves traces for plotting
        """
        self.val_trace[self.trace_loc] = self.val_err
        self.train_trace[self.trace_loc] = self.train_err
        self.trace_loc += 1
        return None

    def freeze_params(self, layer):
        """
        Sets params of layer to be untrainable
        May want to make layer a list or flexible type
        Todo!
        """
        return None

    def save_params(self, param_file):
        """
        Save parameters for reuse later
        """
        all_params = L.layers.get_all_param_values(self.net)
        np.savez(param_file, *all_params)
        return None

    def load_params(self, param_file):
        """
        Load params from saved file
        """
        with np.load(param_file) as loaded:
            L.layers.set_all_param_values(self.net, [i[1] for i in loaded.items()])
        return None
