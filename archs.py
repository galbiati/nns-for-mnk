import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def ffn(input_var=None):
    input_shape = (None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.DropoutLayer(network, p=.2)
    network = lasagne.layers.DenseLayer(network, num_units=800,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=800,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=36, nonlinearity=lasagne.nonlinearities.softmax)
    return network

def cnn(input_var=None):
    # best: 256 filters; 64 filters almost as good, so keeping that
    input_shape=(None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(4,4), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.DenseLayer(network, num_units=36, 
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network