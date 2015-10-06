#!/usr/bin/env python

# Going by the architecture used in Srivastava's paper, detailed here:
# https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/papers/dropout/mnist_valid.yaml
# 
# Unsure how to deal with input dropout; we're assuming (in the case of Wang)
# that input dropout is propagated through to be noise on the pre-nonlinearity
# activations. But then, the final noise is going to end up directly damaging
# the predictions, which makes no sense. Probably better just to ignore reason
# in this case and just write the architecture the way it's probably supposed
# to be.

import layers
import lasagne.layers
import lasagne.nonlinearities
import urllib2
import imp

def wangDropoutArchitecture(batch_size=128, input_dim=784, output_dim=10,
                            DropoutLayer=layers.WangGaussianDropout,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2)
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden, 
            nonlinearity=lambda x: x)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lambda x: x)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def srivastavaDropoutArchitecture(batch_size=128, input_dim=784, output_dim=10,
                            DropoutLayer=layers.WangGaussianDropout,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2)
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5)
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def earlystopping(loop, delta=0.01, max_N=1000, verbose=False):
    """
    Stops the expriment once the loss stops improving by delta per epoch.
    With a max_N of epochs to avoid infinite experiments.
    """
    
    return loop

def load_data():
    """
    Standardising data loading; all using MNIST in the usual way:
        * train: 50000
        * valid: 10000
        * test: separate 10000
    """
    # is this the laziest way to load mnist?
    mnist = imp.new_module()
    exec urllib2.urlopen("https://raw.githubusercontent.com/Lasagne/Lasagne"
            "/master/examples/mnist.py").read() in mnist.__dict__
    dataset = mnist.load_dataset()
    return dict(X_train=dataset[0].reshape(-1, 784),
                y_train=dataset[1],
                X_valid=dataset[2].reshape(-1, 784),
                y_valid=dataset[3],
                X_test=dataset[4].reshape(-1, 784),
                y_test=dataset[5])
