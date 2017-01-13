import theano
import lasagne

T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities

def make_FixLayer(input_var):
    class FixLayer(L.Layer):
        def get_output_for(self, input, **kwargs):
            corrector = (1 - input_var.sum(axis=1))
            corrector = corrector.reshape((input_var.shape[0], 36))
            numer = input * corrector
            return numer

    return FixLayer

class ReNormLayer(L.Layer):
    def get_output_for(self, input, **kwargs):
        return input / input.sum(axis=1).dimshuffle((0, 'x'))

def default_network(nfil=32, input_var=None):
    """Theano graph for basic convnet WITH legal move filter"""
    input_shape=(None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)

    network = L.Conv2DLayer(
        input_layer,
        num_filters=nfil, filter_size=(4,4), pad='full',
        nonlinearity=nl.rectify,
        W=lasagne.init.GlorotUniform()
    )

    network = L.DropoutLayer(network, p=.75)

    network = L.DenseLayer(
        network,
        num_units=36,
        nonlinearity=nl.identity
    )

    network = L.NonlinearityLayer(
        network,
        nonlinearity=nl.softmax
    )

    network = FixLayer(network)

    return network

def default_autoencoder():
    pass
