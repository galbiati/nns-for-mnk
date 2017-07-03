import theano
import lasagne

from customlayers import *

def multiconvX_ws(input_var=None, subnet_specs=None):
    """Columnar architecture WITH feature value weighting"""
    if subnet_specs is None:
        # set default: 3 horizontal, 3 vertical, 3 diagonal/complex
        subnet_specs = [
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4)
        ]

    FixLayer = make_FixLayer(input_var)
    input_shape = (None, 2, 4, 9)
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    subnets = [
        subnet(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]

    network = WeightedSumLayer(subnets)
    network = output_layers(network, FixLayer, prefilter=True)

    return network

def multiconvX_ws_fullpad(input_var=None, subnet_specs=None):
    """Columnar architecture WITH feature value weighting"""
    if subnet_specs is None:
        # set default: 3 horizontal, 3 vertical, 3 diagonal/complex
        subnet_specs = [
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4)
        ]

    FixLayer = make_FixLayer(input_var)
    input_shape = (None, 2, 4, 9)
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    subnets = [
        subnet3(input_layer, input_var, num_filters=nf, filter_size=fs)
        for nf, fs in subnet_specs
    ]

    network = WeightedSumLayer(subnets)
    network = output_layers(network, FixLayer, prefilter=True)

    return network


def multiconvX_ws_tanh(input_var=None, subnet_specs=None):
    """Columnar architecture with feature value weighting and tanh activation"""
    if subnet_specs is None:
        subnet_specs = [
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4),
            (1, (1, 4)), (1, (4, 1)), (1, 4, 4)
        ]

    FixLayer = make_FixLayer(input_var)
    input_shape = (None, 2, 4, 9)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)

    subnets = make_subnets(input_layer, input_var, subnet_specs=subnet_specs)

    network = WeightedSumLayer(subnets)
    network = L.NonlinearityLayer(network, nonlinearity=nl.tanh)
    network = output_layers(network, FixLayer, prefilter=True)

    return network


def multiconvX(input_var=None, subnet_specs=None):
    """Columnar architecture with NO feature value weighting"""
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
    network = output_layers(network, FixLayer, prefilter=True)
    return network


def archX(input_var=None, num_filters=32, pool_size=2, filter_size=(4,4), pool=True, pad='full'):
    """
    More typical architecture with a single convolutional layer,
    activation map sum-pooling
    """

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

    network = output_layers(network, FixLayer, prefilter=True)

    return network


def archX_binconv(input_var=None,
                    num_filters=32, filter_size=(4, 4), pad='full',
                    pool_size=2, pool=True):

    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    network = BinConvLayer(
        input_layer,
        num_filters=num_filters, filter_size=filter_size, pad=pad,
        # convolution=sum_count_conv,
        nonlinearity=nl.identity
    )

    if pool:
        network = L.FeaturePoolLayer(network, pool_function=T.sum, pool_size=pool_size)

    network = L.DropoutLayer(network, p=.125, shared_axes=(2, 3))
    network = L.DenseLayer(
        network,
        num_units=128,
        W=lasagne.init.HeUniform(gain='relu'), nonlinearity=nl.very_leaky_rectify
    )
    network = L.DropoutLayer(network, p=.5)
    network = L.DenseLayer(
        network,
        num_units=36,
        W=lasagne.init.HeUniform(gain='relu'), nonlinearity=nl.very_leaky_rectify
    )

    network = output_layers(network, FixLayer, prefilter=True)

    return network

def archX_deep(input_var=None,
                conv1kws={'num_filters': 4, 'filter_size': (2, 2)},
                conv2kws={'num_filters': 32, 'filter_size': (2, 2)},
                num_latent=32, pool=False
                ):

    """
    Stack archX two deep
    """

    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)

    network = L.Conv2DLayer(
        input_layer, pad='full', nonlinearity=nl.identity, **conv1kws
    )

    network = L.ParametricRectifierLayer(network, shared_axes='auto')
    network = L.DropoutLayer(network, p=.0625, shared_axes=(2, 3))

    network = L.Conv2DLayer(
        network, pad='full', nonlinearity=nl.identity, **conv2kws
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

    network = output_layers(network, FixLayer, prefilter=True)

    return network
