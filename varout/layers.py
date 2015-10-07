#!/usr/bin/env python

import lasagne.layers
import theano
import theano.tensor as T
import numpy as np
import warnings

from theano.sandbox.rng_mrg import MRG_RandomStreams
_srng = MRG_RandomStreams(42)

def _logit(x):
    """
    Logit function in Theano. Useful for parameterizing alpha.
    """
    return np.log(x/(1. - x))

def _check_p(p):
    """
    Thanks to our logit parameterisation we can't accept p of greater than or
    equal to 0.5 (or we get inf logitalphas). So we'll just warn the user and
    scale it down slightly.
    """
    if p == 0.5:
        warnings.warn("Cannot set p to exactly 0.5, limits are: 0 < p < 0.5."
                " Setting to 0.4999", RuntimeWarning)
        return 0.4999
    elif p > 0.5:
        warnings.warn("Cannot set p to greater than 0.5, limits are: "
                "0 < p < 0.5. Setting to 0.4999", RuntimeWarning)
        return 0.4999
    elif p <= 0.0:
        warnings.warn("Cannot set p to less than or equal to 0.0, limits are: "
                "0 < p < 0.5. Setting to 0.0001", RuntimeWarning)
        return 0.0001
    else:
        return p

class VariationalDropout(lasagne.layers.Layer):
    """
    Base class for variational dropout layers, because the noise sampling
    and initialisation can be shared between type A and B.
    Inits:
        * p - initialisation of the parameters sampled for the noise 
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the 
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't 
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, p=0.5, adaptive=None, nonlinearity=None, 
                 **kwargs):
        lasagne.layers.Layer.__init__(self, incoming, **kwargs)
        self.adaptive = adaptive
        p = _check_p(p)
        # init based on adaptive options:
        if self.adaptive == None:
            # initialise scalar param, but don't register it
            self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )           
        elif self.adaptive == "layerwise":
            # initialise scalar param, allow updates
            self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )
            self.add_param(self.logitalpha, ())
        elif self.adaptive == "elementwise":
            # initialise param for each activation passed
            self.logitalpha = theano.shared(
                value=np.array(
                    np.ones(self.input_shape[1])*_logit(np.sqrt(p/(1.-p)))
                    ).astype(theano.config.floatX),
                name='logitalpha'
                )           
            self.add_param(self.logitalpha, (self.input_shape[1]))
        elif self.adaptive == "weightwise":
            # not implemented yet
            raise NotImplementedError("Not implemented yet, will have to "
                    "use DenseLayer inheritance.")
        # if we get no nonlinearity, just put a non-function there
        if nonlinearity == None:
            self.nonlinearity = lambda x: x
        else:
            self.nonlinearity = nonlinearity

class WangGaussianDropout(lasagne.layers.Layer):
    """
    Replication of the Gaussian dropout of Wang and Manning 2012.
    To use this right, similarly to the above, this has to be applied
    to the activations of the network _before the nonlinearity_. This means
    that the prior layer must have _no nonlinearity_, and then you can 
    either apply a nonlinearity in this layer or afterwards yourself.

    Uses some of the code and comments from the Lasagne GaussianNoiseLayer:
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or tensor scalar, effective dropout probability
    nonlinearity : a nonlinearity to apply after the noising process
    """
    def __init__(self, incoming, p=0.5, nonlinearity=None, **kwargs):
        lasagne.layers.Layer.__init__(self, incoming, **kwargs)
        p = _check_p(p)
        self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )
        # if we get no nonlinearity, just put a non-function there
        if nonlinearity == None:
            self.nonlinearity = lambda x: x
        else:
            self.nonlinearity = nonlinearity

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        if deterministic or self.logitalpha.get_value() <= -20:
            return self.nonlinearity(input)
        else:
            # sample from the Gaussian that dropout would produce:
            self.alpha = T.exp(self.logitalpha)
            mu_z = input
            sigma_z = T.sqrt(T.pow(self.alpha,2)*T.pow(input,2))
            randn = _srng.normal(input.shape, avg=1.0, std=1.)
            return self.nonlinearity(mu_z + sigma_z*randn)

class SrivastavaGaussianDropout(lasagne.layers.Layer):
    """
    Replication of the Gaussian dropout of Srivastava et al. 2014 (section
    10). Applies noise to the activations prior to the weight matrix
    according to equation 11 in the Variational Dropout paper; to match the
    adaptive dropout implementation.

    Uses some of the code and comments from the Lasagne GaussianNoiseLayer:
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or tensor scalar, effective dropout probability
    """
    def __init__(self, incoming, p=0.5, **kwargs):
        super(SrivastavaGaussianDropout, self).__init__(incoming, **kwargs)
        p = _check_p(p)
        self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        self.alpha = T.nnet.sigmoid(self.logitalpha)
        if deterministic or T.mean(self.alpha).eval() == 0:
            return input
        else:
            return input + \
                input*self.alpha*_srng.normal(input.shape, 
                                                      avg=0.0, std=1.)

class VariationalDropoutA(VariationalDropout, SrivastavaGaussianDropout):
    """
    Variational dropout layer, implementing correlated weight noise over the 
    output of a layer. Adaptive version of Srivastava's Gaussian dropout.

    Inits:
        * p - initialisation of the parameters sampled for the noise 
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the 
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't 
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, p=0.5, adaptive=None, nonlinearity=None, 
                 **kwargs):
        VariationalDropout.__init__(self, incoming, p=p, adaptive=adaptive, 
                nonlinearity=nonlinearity, **kwargs)

class VariationalDropoutB(VariationalDropout, WangGaussianDropout):
    """
    Variational dropout layer, implementing independent weight noise. Adaptive
    version of Wang's Gaussian dropout.

    Inits:
        * p - initialisation of the parameters sampled for the noise 
    distribution.
        * adaptive - one of:
            * None - will not allow updates to the dropout rate
            * "layerwise" - allow updates to a single parameter controlling the 
            updates
            * "elementwise" - allow updates to a parameter for each hidden layer
            * "weightwise" - allow updates to a parameter for each weight (don't 
            think this is actually necessary to replicate)
    """
    def __init__(self, incoming, p=0.5, adaptive=None, nonlinearity=None, 
                 **kwargs):
        VariationalDropout.__init__(self, incoming, p=p, adaptive=adaptive, 
                nonlinearity=nonlinearity, **kwargs)

