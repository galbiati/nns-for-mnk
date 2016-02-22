import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def mlp(input_var=None):
    input_shape = (None, 2, 4, 9)
    l_in = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=.2)
    l_hidden_1 = lasagne.layers.DenseLayer(l_in_drop, num_units=100, 
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform())
    l_h1_drop = lasagne.layers.DropoutLayer(l_hidden_1, p=.5)
    l_hidden_2 = lasagne.layers.DenseLayer(l_h1_drop, num_units=100,
                                           nonlinearity=lasagne.nonlinearities.rectify)
    l_h2_drop = lasagne.layers.DropoutLayer(l_hidden_2, p=.5)
    l_out = lasagne.layers.DenseLayer(l_h2_drop, num_units=36,
                                      nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def cnn2(input_var=None):
    input_shape=(None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(2,2), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=512,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=36, 
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network

def cnn3(input_var=None):
    input_shape=(None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(2,2), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=256,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=36, 
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network

def cnn4(input_var=None):
    input_shape=(None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(2,2), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(4,4), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.DenseLayer(network, num_units=256,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=36, 
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network

def cnn5(input_var=None):
    # best: 256 filters; 64 filters almost as good, so keeping that
    input_shape=(None, 2, 4, 9)
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(4,4), pad='full',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform())
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.DenseLayer(network, num_units=36, 
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network