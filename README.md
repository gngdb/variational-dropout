
__FAQ__:

* This is the old algorithm, what about the newer one that induces sparsity in a nice way? The author's published their own [Lasagne code for that][sparse] (I wouldn't recommend using the implementation for that hidden in this repo).
* Lasagne, who still uses that? Here's a [tensorflow replication of the sparsifying variational dropout][tfsparse].

[sparse]: https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn
[tfsparse]: https://github.com/BayesWatch/tf-variational-dropout

__Status__: can replicate the ordering of the variances, but the numbers 
don't quite match yet.

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

The following graphs are attempting to reproduce Figure 1, and we can see a 
similar progression for Variational Dropout A and A2, getting better 
performance. In this case A has performed better than A2, which is not what
we see in the paper.

![figure1a]

![figure1b]

These graphs are produced in the notebook called [Opening Results][opening] 
and the results are by running the scripts in the `experiments` directory.

The following are the results reproduced for Table 1 in the paper. The 
ordering of the variances is approximately correct, but the variances 
_increase_ after training to 100 epochs, which is likely a bug. Also, the 
difference between the estimators is not as great as in the paper:

 stochastic gradient estimator      | top 10 | bottom 10 | top 100 | bottom 100 
------------------------------------|--------|-----------|---------|------------
local reparameterization            | 2.4e+04 | 6.1e+02 | 2.5e+05 | 3e+03
separate weight samples             | 4.8e+04 | 1.2e+03 | 4.9e+05 | 8.2e+03
single weight sample                | 5.8e+04 | 1.5e+03 | 4.7e+05 | 6.8e+03
no dropout                          | 1.5e+04 | 5.5e+02 | 1.4e+05 | 2.7e+03

These are produced in the notebook [Comparing Empirical Variance][compare].

Finally, there is the notebook [Investigating Adaptive Properties][investigating], 
which includes the following image showing the alpha parameters (noise standard
deviations) over the MNIST dataset. It's nice to see that it learns to ignore
the edges:

![ignore]

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
[figure1a]: presentation/images/replfigure1a.png
[figure1b]: presentation/images/replfigure1b.png
[ignore]: presentation/images/ignore.png
[compare]: https://github.com/gngdb/variational-dropout/blob/master/notebooks/Comparing%20Empirical%20Variance.ipynb
[opening]: https://github.com/gngdb/variational-dropout/blob/master/notebooks/Opening%20Results.ipynb
[investigating]: https://github.com/gngdb/variational-dropout/blob/master/notebooks/Investigating%20Adaptive%20Properties.ipynb
