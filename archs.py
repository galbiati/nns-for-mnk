import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def naive_agent(nfil=32, input_var=None):
    input_shape=(None, 2, 4, 9)
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(
        input_layer, 
        num_filters=nfil, filter_size=(4,4), pad='full',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    
    network = lasagne.layers.DropoutLayer(network, p=.75)
    
    network = lasagne.layers.DenseLayer(
        network, 
        num_units=36, 
        nonlinearity=lasagne.nonlinearities.softmax
    )
    
    return network

def smart_agent(nfil=32, input_var=None):
    
    class FixLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            corrector = (1 - input_var.sum(axis=1)).reshape((input_var.shape[0], 36))
            numer = input * corrector
            return numer / numer.sum(axis=1).dimshuffle((0, 'x'))
        
    input_shape=(None, 2, 4, 9)
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(
        input_layer, 
        num_filters=nfil, filter_size=(4,4), pad='full',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    
    network = lasagne.layers.DropoutLayer(network, p=.75)

    network = lasagne.layers.DenseLayer(
        network, 
        num_units=36, 
        nonlinearity=lasagne.nonlinearities.identity
    )
    
    network = lasagne.layers.NonlinearityLayer(
        network, 
        nonlinearity=lasagne.nonlinearities.softmax
    )
    
    network = FixLayer(network)
    
    return network

def softmax_agent(nfil=None, input_var=None):
    
    class FixLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            corrector = (1 - input_var.sum(axis=1)).reshape((input_var.shape[0], 36))
            numer = input * corrector
            return numer / numer.sum(axis=1).dimshuffle((0, 'x'))
        
    input_shape=(None, 2, 4, 9)
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    
    network = lasagne.layers.DenseLayer(
        input_layer, 
        num_units=36, 
        nonlinearity=lasagne.nonlinearities.identity
    )
    
    network = lasagne.layers.NonlinearityLayer(
        network, 
        nonlinearity=lasagne.nonlinearities.softmax
    )
    
    network = FixLayer(network)
    
    return network