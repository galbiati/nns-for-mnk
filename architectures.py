import theano
import lasagne

T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities
# in general, try to make abstract constructors for net motifs

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
    def __init__(self, incoming, W=lasagne.init.Constant(0.), **kwargs):
        super(L.ElemwiseMergeLayer, self).__init__(incoming, T.add, **kwargs)
        num_inputs = len(incoming)
        self.W = self.add_param(W, (num_inputs,), name='W')

    def get_output_shape_for(self, input_shapes):
        """Cribbed from parent class source"""
        return (None, 36) # hardcode because I'm an ignorant pig

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs * self.W)

class ReNormLayer(L.Layer):
    def get_output_for(self, input, **kwargs):
        return input / input.sum(axis=1).dimshuffle((0, 'x'))

class BinaryConvLayer(L.Conv2DLayer):
    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        transW = self.W >= .5
        conved = self.convolution(
            input, transW, self.input_shape, self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=self.flip_filters
        )
        return conved

### NETWORKS ###
# to do ideas:
    # force binary features
    # weighted dense layer sum for subnet outputs
        # let the weights be finetuned by themselves

def binary_subnet(network, input_var, num_filters=4, filter_size=(4, 4)):
    FixLayer = make_FixLayer(input_var)
    net = BinaryConvLayer(
        network, num_filters=num_filters,
        filter_size=filter_size, pad='full',
        nonlinearity=nl.identity, W=lasagne.init.Constant(.5)
    )
    # net = L.ParametricRectifierLayer(net, shared_axes='auto')

    net = L.FeaturePoolLayer(net, pool_function=T.sum, pool_size=num_filters)
    net = L.DropoutLayer(net, p=.5)
    net = L.DenseLayer(
        net, num_units=36,
        nonlinearity=nl.very_leaky_rectify, W=lasagne.init.HeUniform(gain='relu')
    )
    network = FixLayer(net)

def subnet(network, input_var, num_filters=4, filter_size=(4, 4)):
    FixLayer = make_FixLayer(input_var)
    net = L.Conv2DLayer(
        network, num_filters=num_filters,
        filter_size=filter_size, pad='full',
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
    subnets = [subnet(input_layer, input_var, num_filters=nf, filter_size=fs) for nf, fs in subnet_specs]

    network = L.ElemwiseMergeLayer(subnets, merge_function=T.add)
    # network = WeightedSumLayer(subnets)
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)
    return network

def multiconvX_medlrg(input_var=None):
    FixLayer = make_FixLayer(input_var)

    input_shape = (None, 2, 4, 9)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    subnet1 = subnet(input_layer, input_var, num_filters=2, filter_size=(1,4))
    subnet2 = subnet(input_layer, input_var, num_filters=2, filter_size=(4,1))
    subnet3 = subnet(input_layer, input_var, num_filters=4, filter_size=(4,4))

    subnet4 = subnet(input_layer, input_var, num_filters=2, filter_size=(1,3))
    subnet5 = subnet(input_layer, input_var, num_filters=2, filter_size=(3,1))
    subnet6 = subnet(input_layer, input_var, num_filters=4, filter_size=(3,3))

    subnet7 = subnet(input_layer, input_var, num_filters=2, filter_size=(1,2))
    subnet8 = subnet(input_layer, input_var, num_filters=2, filter_size=(2,1))
    subnet9 = subnet(input_layer, input_var, num_filters=4, filter_size=(2,2))
    subnets = [subnet1, subnet2, subnet3, subnet4, subnet5, subnet6, subnet7, subnet8, subnet9]

    network = L.ElemwiseMergeLayer(subnets, merge_function=T.add)
    # network = WeightedSumLayer(subnets)
    network = FixLayer(network)
    network = L.NonlinearityLayer(network, nonlinearity=nl.softmax)
    network = FixLayer(network)
    network = ReNormLayer(network)
    return network

def multiconvX1(input_var=None):
    FixLayer = make_FixLayer(input_var)

    input_shape = (None, 2, 4, 9)

    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    subnet1 = subnet(input_layer, input_var, num_filters=4, filter_size=(1,4))
    subnet2 = subnet(input_layer, input_var, num_filters=4, filter_size=(4,1))
    subnet3 = subnet(input_layer, input_var, num_filters=8, filter_size=(4,4))

    subnet4 = subnet(input_layer, input_var, num_filters=4, filter_size=(1,3))
    subnet5 = subnet(input_layer, input_var, num_filters=4, filter_size=(3,1))
    subnet6 = subnet(input_layer, input_var, num_filters=8, filter_size=(3,3))

    subnet7 = subnet(input_layer, input_var, num_filters=4, filter_size=(1,2))
    subnet8 = subnet(input_layer, input_var, num_filters=4, filter_size=(2,1))
    subnet9 = subnet(input_layer, input_var, num_filters=8, filter_size=(2,2))
    subnets = [subnet1, subnet2, subnet3, subnet4, subnet5, subnet6, subnet7, subnet8, subnet9]

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

def default_convnet(input_var=None):
    return archX(input_var, num_filters=32, pool_size=2, filter_size=(4, 4))
