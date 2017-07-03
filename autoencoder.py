import theano
import lasagne

from customlayers import output_layers, make_FixLayer

### ALIASES ###
T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities

def autoencoder(input_var=None,
                    num_filters=16, filter_size=(4, 4), pad='full',
                    latent_size=20):

    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    # encode
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    conv1 = L.Conv2DLayer(
        input_layer,
        num_filters=num_filters, filter_size=filter_size, pad=pad,
        nonlinearity=nl.very_leaky_rectify
    )

    conv1_d = L.DropoutLayer(conv1, p=.0625, shared_axes=(2, 3))

    conv2 = L.Conv2DLayer(
        conv1,
        num_filters=num_filters*2, filter_size=filter_size, pad=pad,
        nonlinearity=nl.very_leaky_rectify
    )

    conv2_d = L.DropoutLayer(conv2, p=.0625, shared_axes=(2, 3))

    latent = L.DenseLayer(conv2_d, num_units=latent_size, nonlinearity=nl.very_leaky_rectify)
    latent_d = L.DropoutLayer(latent, p=.25)

    # decode
    latent_decode = L.InverseLayer(latent_d, latent)
    unconv2 = L.InverseLayer(latent_decode, conv2)
    decoded = L.InverseLayer(unconv2, conv1)

    # predict
    values = L.DenseLayer(latent_d, num_units=36, nonlinearity=nl.very_leaky_rectify)
    prediction = output_layers(values, FixLayer, prefilter=False)

    return decoded, values, prediction
