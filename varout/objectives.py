#!/usr/bin/env python

import lasagne.layers
import theano.tensor as T
from varout.layers import VariationalDropout
from lasagne.utils import floatX

# first part is the KL divergence of the priors
def priorKL(output_layer):
    """
    Has the same interface as L2 regularisation in lasagne.
    Input:
        * output_layer - final layer in your network, used to pull the 
        weights out of every other layer
    Output:
        * Theano expression for the KL divergence on the priors:
        - D_{KL}( q_{\phi}(w) || p(w) )
    """
    # gather up all the alphas
    params = lasagne.layers.get_all_params(output_layer)
    alphas = [T.nnet.sigmoid(p) for p in params if p.name == "logitalpha"]

    # I hope all these decimal places are important
    c1 = 1.161451241083230
    c2 = -1.502041176441722
    c3 = 0.586299206427007

    # will get taken apart again in the autodiff
    return sum([0.5*T.sum(T.log(alpha)) + c1*T.sum(alpha) + c2*T.sum(T.pow(alpha,2))
                                 + c3*T.sum(T.pow(alpha,3)) for alpha in alphas])

def sparsityKL(output_layer):
    """
    Based on the paper "Variational Dropout Sparsifies Deep Neural Networks" by 
    Dmitry Molchanov, Arsenii Ashukha and Dmitry Vetrov, https://arxiv.org/abs/1701.05369.
    Modification so that we don't need to constrain alpha to be below 1. Then,
    the network is free to drop units that are not useful.
    Input:
        * output_layer - final layer in a Lasagne network, so we can pull  the 
        alpha values out of all the layers
    """
    # gather up all the alphas
    params = lasagne.layers.get_all_params(output_layer)
    alphas = [T.exp(p) for p in params if p.name == "logalpha"]

    return sum([0.64*T.nnet.sigmoid(1.5*(1.3*T.log(alpha)))-0.5*T.log(1+T.pow(alpha,-1)) for alpha in alphas])
    
def mclog_likelihood(N=None, 
        base_likelihood=lasagne.objectives.categorical_crossentropy):
    return lambda predictions, targets: N*base_likelihood(predictions, targets)

class LowerBound(object):
    def __init__(self, base_objective, output_layer, dataset_size=50000):
        self.base_objective = base_objective
        self.N = dataset_size
        self.DKL = priorKL(output_layer)

    def __call__(self, predictions, targets):
        return T.mean(self.base_objective(predictions, targets)) +\
               self.DKL/self.N

class LowerBound(object):
    def __init__(self, base_objective, output_layer, dataset_size=50000):
        self.base_objective = base_objective
        self.N = dataset_size
        self.DKL = sparsityKL(output_layer)

    def __call__(self, predictions, targets):
        return T.mean(self.base_objective(predictions, targets)) +\
               self.DKL/self.N


