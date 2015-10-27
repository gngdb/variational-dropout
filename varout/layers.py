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
    Logit function in Numpy. Useful for parameterizing alpha.
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
        self.init_adaptive(adaptive, p=p)

    def init_adaptive(self, adaptive, p):
        """
        Initialises adaptive parameters.
        """
        if not hasattr(self, 'input_shape'):
            self.input_shape = self.input_shapes[0]
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
            self.add_param(self.logitalpha, (self.input_shape[1],))
        elif self.adaptive == "weightwise":
            # this will only work in the case of dropout type B
            thetashape = (self.input_shapes[1][1],self.input_shapes[0][1])
            self.logitalpha = theano.shared(
                value=np.array(
                    np.ones(thetashape)*_logit(np.sqrt(p/(1.-p)))
                    ).astype(theano.config.floatX),
                name='logitalpha'
                )                      
            self.add_param(self.logitalpha, thetashape)


class WangGaussianDropout(lasagne.layers.MergeLayer):
    """
    Replication of the Gaussian dropout of Wang and Manning 2012.
    This layer will only work after a dense layer, but can probably be extended
    to work with convolutional layers. Internally, this pulls out the weights
    from the previous dense layer and applies them again itself, throwing away 
    the expression passed from the dense layer. This is necessary because we
    need the expression before the nonlinearity is applied and because we need
    to calculate the sigma. This idiosyncratic method was chosen because it 
    keeps the dropout architecture descriptions easy to read.

    Uses some of the code and comments from the Lasagne GaussianNoiseLayer:
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or tensor scalar, effective dropout probability
    nonlinearity : a nonlinearity to apply after the noising process
    """
    def __init__(self, incoming, p=0.5, **kwargs):
        incoming_input = lasagne.layers.get_all_layers(incoming)[-2] 
        lasagne.layers.MergeLayer.__init__(self, [incoming, incoming_input], 
                **kwargs)
        # store p in logit space
        p = _check_p(p)
        self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )
        # and store the parameters of the previous layer
        self.num_units = incoming.num_units
        self.theta = incoming.W
        self.b = incoming.b
        self.nonlinearity = incoming.nonlinearity

    def get_output_shape_for(self, input_shapes):
        """
        Output shape will always be equal the shape coming out of the dense
        layer previous.
        """
        return (input_shapes[1][0], self.num_units)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        # repeat check from DenseLayer
        if inputs[1].ndim > 2:
            # flatten if we have more than 2 dims
            inputs[1].ndim = inputs[1].flatten(2)
        self.alpha = T.nnet.sigmoid(self.logitalpha)
        mu_z = T.dot(inputs[1], self.theta) + self.b.dimshuffle('x', 0)
        if deterministic or T.mean(self.alpha).eval() == 0:
            return self.nonlinearity(mu_z)
        else:
            # sample from the Gaussian that dropout would produce
            sigma_z = T.sqrt(T.dot(T.square(inputs[1]), 
                                   self.alpha*T.square(self.theta)))
            randn = _srng.normal(size=inputs[0].shape, avg=0.0, std=1.)
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

class VariationalDropoutB(WangGaussianDropout, VariationalDropout):
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
    def __init__(self, incoming, p=0.5, adaptive=None, **kwargs):
        WangGaussianDropout.__init__(self, incoming, p=p, **kwargs)
        self.init_adaptive(adaptive, p=p)

class SingleWeightSample(lasagne.layers.DenseLayer):
    """
    MC on the uncertainty of the weights by taking a single sample of the 
    weight matrix and propagating forwards.
    """
    def __init__(self, incoming, num_units, p=0.5, **kwargs):
        super(SingleWeightSample, self).__init__(incoming, num_units, **kwargs)
        # then initialise the noise terms for each weight
        p = _check_p(p)
        self.logitalpha = theano.shared(
                value=np.array(_logit(np.sqrt(p/(1.-p)))).astype(theano.config.floatX),
                name='logitalpha'
                )
        self.alpha = T.nnet.sigmoid(self.logitalpha)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        self.W_noised = self.W*(1. + _srng.normal(self.W.shape, avg=0.0, std=1.0)*T.sqrt(self.alpha))
        activation = T.dot(input, self.W_noised)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class SeparateWeightSamples(SingleWeightSample):
    """
    MC on the uncertainty of the weights by taking a separate sample of the
    weight matrix for each example in the input matrix. Extremely slow.
    """
    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
        output from the previous layer
        deterministic : bool
        If true noise is disabled, see notes
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        self.W_noised = self.W*(1 + _srng.normal((self.input_shape[0], 
                        self.W.shape[0],
                        self.W.shape[1]), avg=0.0, std=1.0)*T.sqrt(self.alpha))
        # then just extract from each independent weight matrix
        activation,_ = theano.scan(lambda i,w: T.dot(i,w), 
                sequences=[input, self.W_noised])
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
