FROM ipython/scipyserver

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip2 install -r /tmp/requirements.txt

# install some unnecessary requirements to run notebooks and scripts
ADD extra_requirements.txt /tmp/extra_requirements.txt
RUN pip2 install -r /tmp/extra_requirements.txt

RUN mkdir /variational-dropout
WORKDIR /variational-dropout
