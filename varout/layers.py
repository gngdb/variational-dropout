#!/usr/bin/env python

import lasagne.layers
import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams
_srng = MRG_RandomStreams(42)

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
    def __init__(self, incoming, p=0.5, adaptive=None):
        super(VariationalDropout, self).__init__(incoming, **kwargs)
        self.adaptive = adaptive
        # init based on adaptive options:
        if self.adaptive == None:
            # initialise scalar param, disallow updates through _get_params
            self.alpha = theano.shared(
                value=np.array(np.sqrt(p/(1.-p))).astype(theano.config.floatX),
                name='alpha'
                )           
        elif self.adaptive == "layerwise":
            # initialise scalar param, allow updates
            self.alpha = theano.shared(
                value=np.array(np.sqrt(p/(1.-p))).astype(theano.config.floatX),
                name='alpha'
                )           
        elif self.adaptive == "elementwise":
            # initialise param for each activation passed
            self.alpha = theano.shared(
                value=np.array(
                    np.ones(self.input_shape[1])*np.sqrt(p/(1.-p))
                    ).astype(theano.config.floatX),
                name='alpha'
                )           
        elif self.adaptive == "weightwise":
            # not implemented yet
            raise NotImplementedError("Not implemented yet, will have to "
                    "use DenseLayer inheritance.")

    def get_params(self):
        """
        returns parameters, if allowed
        """
        if self.adaptive != None:
            return self.alpha

    def _sample_noise(self):
        """
        sample a noise matrix using the current alpha: N(1,alpha) 
        aka N(vector of ones, diag(alpha))
        """
        noise = _srng.normal(self.input_shape, avg=0.0, std=1.0)
        # vectors will multiply row-wise, and scalar will distribute
        # (check it out, the reparameterization trick:)
        noise = 1.0+self.alpha*noise
        return noise

class VariationalDropoutA(VariationalDropout):
    """
    Variational dropout layer, implementing correlated weight noise over the 
    output of a layer. 

    Inits:
        * p - initialisation of the parameters sampled for the noise 
    distribution.
        * 
    """
    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        return None

class VariationalDropoutB(VariationalDropout):
    """
    Variational dropout layer, implementing independent weight noise.
    """
    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        return None


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
        # sigma is called alpha in the paper
        self.sigma = theano.shared(
                value=np.array(np.sqrt(p/(1.-p))).astype(theano.config.floatX),
                name='alpha'
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
        if deterministic or self.sigma.get_value() == 0:
            return input
        else:
            return input + \
                input*_srng.normal(input.shape, avg=0.0, std=1.)*self.sigma

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
        super(WangGaussianDropout, self).__init__(incoming, **kwargs)
        # interpretation is inclusion probabilities so we have to reverse the 
        # Lasagne convention
        self.alpha = theano.shared(
                value=np.array(1.-p).astype(theano.config.floatX),
                name='alpha'
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
        if deterministic or self.p.get_value() == 0:
            return self.nonlinearity(input)
        else:
            # sample from the Gaussian that dropout would produce:
            mu_z = input
            sigma_z = T.sqrt((self.alpha/(1.-self.alpha))*T.pow(input,2))
            randn = _srng.normal(input.shape, avg=1.0, std=1.)
            return self.nonlinearity(mu_z + sigma_z*randn)
