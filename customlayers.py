import theano
import lasagne

### ALIASES ###
T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities


### NONLINEARITIES ETC###
def binarize(input_tensor):
    """
    Convert input to el {0, 1}

    Might be better to achieve this by
    forcing boolean shared on ConvLayer subclass
    """
    return 2 * T.cast(input_tensor >= .5, theano.config.floatX) - 1


def sum_count_conv(input, W, input_shape, W_shape,
                    subsample=None, border_mode=None, filter_flip=None):
    """
    After convolving, checks each filter map (per channel) to determine if
    feature activation equals weight sum

    Meant for use with binarized filters,
    resulting in an on/off "feature detector"

    Not sure what happens without binarized features, but likely not what you
    wanted!

    Does not currently work due to dim mismatch error; debugging soon
    """
    W_sum = W.sum(axis=-1).sum(axis=-1)
    conved = T.nnet.conv2d(
        input, W, input_shape, W_shape,
        subsample=subsample, border_mode=border_mode, filter_flip=filter_flip
    )

    comparand = T.tile(
        W_sum,
        (conved.shape[0], conved.shape[2], conved.shape[3], 1)
    )

    comparand = comparand.dimshuffle(0, 3, 2, 1)

    return T.eq(conved, comparand)


### LAYERS ###


def make_FixLayer(input_var):
    """
    Creates a layer that "fixes" a value distribution for a given input size

    This is used to enforce (m, n, k) game rules - it is not legal to move
    at an occupied location, so in a value/choice distribution, occupied locations
    should be set to 0.

    Should return to this to make a proper class using super to pass input_shape
    (low priority)
    """

    class FixLayer(L.Layer):
        def get_output_for(self, input, **kwargs):
            corrector = (1 - input_var.sum(axis=1))
            reshape = (input_var.shape[0], 36)
            corrector = corrector.reshape(reshape)
            numer = input * corrector
            return numer

    return FixLayer


class BinConvLayer(L.Conv2DLayer):
    """
    Binarizes weights before computing convolution

    This does not work very well; gradient is 0 at all locations, so backprop
    does not work correctly. Consider using tanh or sigmoid activation function
    instead.
    """

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        binarized = binarize(self.W)
        conved = self.convolution(input, binarized,
            self.input_shape, self.get_W_shape(),
            subsample=self.stride,
            border_mode=border_mode,
            filter_flip=self.flip_filters
        )

        return conved


class WeightedSumLayer(L.ElemwiseMergeLayer):
    """
    Weighted sum layer computes a linear function of input layers

    It has a number of weights equivalent to the number of input layers

    self.get_output_for should probably compute a dot product directly
    after stacking instead (low priority)
    """

    def __init__(self, incoming, W=lasagne.init.Constant(1.), **kwargs):
        super(L.ElemwiseMergeLayer, self).__init__(incoming, T.add, **kwargs)
        num_inputs = len(incoming)
        self.W = self.add_param(W, (num_inputs,), name='W')

    def get_output_shape_for(self, input_shapes):
        return (None, 36) # hardcode because I'm an ignorant pig

    def get_output_for(self, inputs, **kwargs):
        outputs = T.stack(inputs, axis=2) * self.W
        return T.sum(outputs, axis=2)


class ReNormLayer(L.Layer):
    """
    Renormalizes a value distribution (after illegal moves are zeroed out)
    """
    def get_output_for(self, input, **kwargs):
        return input / input.sum(axis=1).dimshuffle((0, 'x'))


### COMPOSITE LAYERS ###


def subnet(network, input_var,
            num_filters=4, filter_size=(4, 4), pad='valid'):
    """
    Subnet produces a "column" with a given convolution shape and filter count

    Return to this to make kwarg passing more legible (low priority)
    """

    FixLayer = make_FixLayer(input_var)

    net = L.Conv2DLayer(
        network,
        num_filters=num_filters, filter_size=filter_size, pad=pad,
        nonlinearity=nl.identity
    )

    net = L.ParametricRectifierLayer(net, shared_axes='auto')
    net = L.DropoutLayer(net, p=.125, shared_axes=(2, 3))
    net = L.FeaturePoolLayer(net, pool_function=T.sum, pool_size=num_filters)
    net = L.DenseLayer(
        net, num_units=36,
        nonlinearity=nl.very_leaky_rectify, W=lasagne.init.HeUniform(gain='relu')
    )
    net = L.DropoutLayer(net, p=.125)
    net = FixLayer(net)

    return net


def subnet2(network, input_var,
                num_filters=4, filter_size=(4, 4), pad='valid',
                densekws={'nonlinearity': nl.tanh}):
    """
    Sigmoid subnet uses sigmoid activation on value layer to make values more
    interpretable
    """

    FixLayer = make_FixLayer(input_var)
    net = L.Conv2DLayer(
        network,
        num_filters=num_filters, filter_size=filter_size, pad=pad,
        nonlinearity=nl.very_leaky_rectify
    )

    net = L.DropoutLayer(net, p=.125, shared_axes=(2, 3))
    net = L.FeaturePoolLayer(net, pool_function=T.sum, pool_size=num_filters)
    net = L.DenseLayer(
        net,
        num_units=36, W=lasagne.init.GlorotUniform(gain=1.0), **denskws
    )

    net = L.DropoutLayer(net, p=.125)
    return net


def make_subnets(network, input_var, subnet_func=subnet, subnet_specs=None):
    """
    Generates a list of "columns" according to subnet_specs

    subnet_specs is a list of kwargs to be passed to subnet_func
    """
    if subnet_specs is None:
        # set default
        subnet_specs = [
            (4, (1, 4)), (4, (4, 1)), (4, (4, 4)),
            (4, (1, 3)), (4, (3, 1)), (4, (3, 3)),
            (4, (1, 2)), (4, (2, 1)), (4, (2, 2))
        ]
    FixLayer = make_FixLayer(input_var)
    input_shape = (None, 2, 4, 9)
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    subnets = [
        subnet_func(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]
    return subnets


def output_layers(incoming, FixLayer, prefilter=False):
    """Applies softmax and filters illegal moves"""

    if prefilter:
        network = FixLayer(incoming)
        network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    else:
        network = L.NonlinearityLayer(incoming, nonlinearity=nl.softmax)

    network = FixLayer(network)
    network = ReNormLayer(network)

    return network
