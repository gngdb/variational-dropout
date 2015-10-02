FROM ipython/scipyserver

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip2 install -r /tmp/requirements.txt

RUN mkdir /variational-dropout
WORKDIR /variational-dropout
