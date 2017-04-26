import theano
import lasagne

T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities

### LAYERS ###
def make_FixLayer(input_var):
    class FixLayer(L.Layer):
        def get_output_for(self, input, **kwargs):
            corrector = (1 - input_var.sum(axis=1))
            reshape = (input_var.shape[0], 36)
            corrector = corrector.reshape(reshape)
            numer = input * corrector
            return numer

    return FixLayer


class WeightedSumLayer(L.ElemwiseMergeLayer):
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
    def get_output_for(self, input, **kwargs):
        return input / input.sum(axis=1).dimshuffle((0, 'x'))


### NETWORKS ###
# to do ideas:
    # force binary features
    # hand-engineer heuristic function imitator layers


def subnet(network, input_var, num_filters=4, filter_size=(4, 4), pad='valid'):
    FixLayer = make_FixLayer(input_var)
    net = L.Conv2DLayer(
        network, num_filters=num_filters,
        filter_size=filter_size, pad=pad,
        nonlinearity=nl.identity
    )
    net = L.ParametricRectifierLayer(net, shared_axes='auto')
    net = L.DropoutLayer(net, p=.125, shared_axes=(2, 3))
    net = L.FeaturePoolLayer(net, pool_function=T.sum, pool_size=num_filters)
    # net = L.DropoutLayer(net, p=.25)
    net = L.DenseLayer(
        net, num_units=36,
        nonlinearity=nl.very_leaky_rectify, W=lasagne.init.HeUniform(gain='relu')
    )
    net = L.DropoutLayer(net, p=.125)
    net = FixLayer(net)
    return net


def make_subnets(network, input_var, subnet_specs=None):
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
        subnet(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]
    return subnets


def multiconvX_ws(input_var=None, subnet_specs=None):
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
        subnet(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]

    network = WeightedSumLayer(subnets)
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)
    return network


def multiconvX(input_var=None, subnet_specs=None):
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
        subnet(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]

    network = L.ElemwiseMergeLayer(subnets, merge_function=T.add)
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)
    return network

def archX(input_var=None, num_filters=32, pool_size=2, filter_size=(4,4), pool=True, pad='full'):
    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    network = L.Conv2DLayer(
        input_layer, num_filters=num_filters,
        filter_size=filter_size, pad=pad,
        nonlinearity=nl.identity
    )
    network = L.ParametricRectifierLayer(network, shared_axes='auto')
    network = L.DropoutLayer(network, p=.125, shared_axes=(2, 3))

    if pool:
        network = L.FeaturePoolLayer(network, pool_function=T.sum, pool_size=pool_size)
    network = L.DropoutLayer(network, p=.75)
    network = L.DenseLayer(
        network, num_units=36,
        nonlinearity=nl.very_leaky_rectify, W=lasagne.init.HeUniform(gain='relu')
    )
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)

    return network

def archX_deep(input_var=None,
                conv1kws={'num_filters': 4, 'filter_size': (2, 2)},
                conv2kws={'num_filters': 32, 'filter_size': (2, 2)},
                num_latent=32, pool=False
                ):
    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)

    network = L.Conv2DLayer(
        input_layer, pad='full', nonlinearity=nl.identity, **conv1kws
    )

    network = L.ParametricRectifierLayer(network, shared_axes='auto')
    network = L.DropoutLayer(network, p=.0625, shared_axes=(2, 3))

    network = L.Conv2DLayer(
        network, pad='full', nonlinearity=nl.identity,
        **conv2kws
    )

    network = L.ParametricRectifierLayer(network, shared_axes='auto')
    network = L.DropoutLayer(network, p=.0625, shared_axes=(2, 3))

    if pool:
        network = L.FeaturePoolLayer(
            network, pool_function=T.sum, pool_size=conv2kws['num_filters']
        )


    network = L.DenseLayer(
        network, num_units=num_latent, nonlinearity=nl.leaky_rectify,
        W=lasagne.init.HeUniform(gain='relu')
    )
    network = L.DropoutLayer(network, p=.25)

    network = L.DenseLayer(
        network, num_units=36, nonlinearity=nl.identity,
        W=lasagne.init.HeUniform(gain='relu')
    )

    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)

    return network
