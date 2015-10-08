
__Status__: not able to replicate all of the results in the paper yet, see
below.

This is a replication of the pre-print paper [Variational Dropout and the
Local Reparameterization Trick][arxiv] by [Diederik Kingma][kingma], 
[Tim Salimans][salimans] and [Max Welling][welling].

The code is written using [Theano][] and [Lasagne][], following Lasagne
layer conventions so that it should be modular enough to use elsewhere.
Instructions for how to replicate results are below.

Installation
============

The requirements listed in the `requirements.txt` are only what's required
to install this package so you can use it as a module. They aren't
sufficient to actually run all of the scripts and notebooks. In addition,
you will need:

```
ipython[notebook]
holoviews
holo-nets
pandas
seaborn
```

There is a Dockerfile accompanying this repository, which can be pulled
from [Docker Hub][dockerhub]. It's based on the [IPython
Scipyserver][scipyserver] image. To run the image with a self-signed
certificate you can do the following; first pull the image:

```
docker pull gngdb/variational-dropout
```

Then clone this repository so it can be mounted in this container and `cd`
into it.

```
git clone https://github.com/gngdb/variational-dropout.git
cd variational-dropout
```

Then run the container with the following command (choosing a password):

```
docker run -d -p 443:8888 -e PASSWORD=<CHOOSE A PASSWORD> -v $PWD:/variational-dropout gngdb/variational-dropout
```

Now you can just navigate to https://localhost to use your notebook.
Unfortunately, this has no support for CUDA or GPUs (although [it is
possible to do this inside a container][kaixhin]) so any of the experiment
scripts will take a very long time to run. They're not completely
unworkable on a reasonable desktop though.

Finally, in order to run scripts or use most of the notebooks you must
install the package in develop mode. Open a terminal on the Jupyter server
(or otherwise get a shell inside the container):

```
python setup.py develop
```

Replicating Results
===================

There are practically just two parts of the paper we'd like to be able to
reproduce:

* Table 1 - showing empirical variance estimates of the method versus other
methods.
* Figure 1 - showing performance in terms of percentage error on the test
set for the following:
    * No dropout
    * Regular binary dropout
    * Gaussian dropout A ([Srivastava et al][sriva])
    * Variational dropout A
    * Variational dropout A2
    * Gaussian dropout B ([Wang et al][wang])
    * Variational dropout B

Once this is done, we'd like to look at the adaptive gradients in a bit
more detail (there doesn't appear to have been space in the paper to
discuss them more) and see what kind of properties they have.

So far, the results comparing performance on the validation set (should be
run on the __test set__, updated results are pending) are as follows:

![figure1a]

![figure1b]

These graphs are produced in the notebook called "Opening Results" and the
results are by running the scripts in the `experiments` directory.

Table of results comparing the gradient variances pending.

[arxiv]: http://arxiv.org/abs/1506.02557
[kingma]: http://dpkingma.com/
[salimans]: http://timsalimans.com/
[welling]: https://www.ics.uci.edu/~welling/
[theano]: http://deeplearning.net/software/theano/
[lasagne]: https://lasagne.readthedocs.org/en/latest/
[dockerhub]: https://hub.docker.com/r/gngdb/variational-dropout/
[scipyserver]: https://github.com/ipython/docker-notebook/tree/master/scipyserver
[kaixhin]: https://github.com/Kaixhin/dockerfiles
[sriva]: http://jmlr.org/papers/v15/srivastava14a.html
[wang]: http://machinelearning.wustl.edu/mlpapers/papers/wang13a
[figure1a]: images/figure1a.png
[figure1b]: images/figure1b.png
