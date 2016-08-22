import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

def naive_agent(nfil=32, input_var=None):
    """Theano graph for basic convnet WITHOUT legal move filter"""
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
    """Theano graph for basic convnet WITH legal move filter"""
    
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
    """Just an input layer and a dense softmax output layer"""
    # TODO: in load_data.py, add a subject-by-subject loader for alternate training scheme
    
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

def binconv2d(
    input, filters, image_shape=None, filter_shape=None,
    border_mode='valid', subsample=(1, 1), **kargs
):
    filters = T.round_half_away_from_zero(T.nnet.sigmoid(filters))
    return T.nnet.conv2d(
        input, filters, image_shape=image_shape, filter_shape=filter_shape,
        border_mode=border_mode, subsample=subsample, **kargs
    )
    

def hard_filter_agent(nfil=32, input_var=None):
    """Creates a convnet with a special convolution that truncates weights to 0/1, 
    analogous to our heuristic function for search models"""
    
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
        W=lasagne.init.GlorotUniform(),
        convolution=binconv2d
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