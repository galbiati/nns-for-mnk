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

def default_convnet(input_var=None):
    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    network = L.Conv2DLayer(
        input_layer, num_filters=32, filter_size=(4,4), pad='full',
        nonlinearity=nl.identity
    )

    network = L.ParametricRectifierLayer(network, shared_axes='auto') # default: auto
    network = L.FeaturePoolLayer(network, pool_function=T.sum, pool_size=4) # default: T.sum, 2
    network = L.DropoutLayer(network, p=.75) # default: .75
    network = L.DenseLayer(
        network, num_units=36,
        nonlinearity=nl.very_leaky_rectify, W=lasagne.init.HeUniform(gain='relu')
    )
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)

    return network

def default_autoencoder():
    pass
