import theano
import lasagne

from customlayers import output_layers, make_FixLayer

### ALIASES ###
T = theano.tensor
L = lasagne.layers
nl = lasagne.nonlinearities

def autoencoder(input_var=None,
                    num_filters=32, filter_size=(3, 3), pad='same',
                    latent_size=128):

    input_shape = (None, 2, 4, 9)
    FixLayer = make_FixLayer(input_var)

    # encode
    input_layer = L.InputLayer(shape=input_shape, input_var=input_var)
    conv1 = L.Conv2DLayer(
        input_layer,
        num_filters=num_filters, filter_size=filter_size, pad=pad,
        nonlinearity=nl.tanh
    )

    latent = L.DenseLayer(conv1, num_units=num_units, nonlinearity=nl.rectify)

    # decode
    latent_decode = L.InverseLayer(latent, latent)
    unconv1 = L.InverseLayer(latent_decode, conv1)

    # predict
    values = L.DenseLayer(latent, num_units=36, nonlinearity=nl.rectify)
    prediction = output_layers(values, FixLayer, prefilter=False)

    return unconv1, prediction
