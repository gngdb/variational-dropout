FROM ipython/scipyserver

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
