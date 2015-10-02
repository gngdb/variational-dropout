#!/usr/bin/env python

from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

setup(name='variational-dropout',
      version='0.0.1',
      description='Initial Variational Dropout replication.',
      author='Gavin Gray',
      author_email='g.d.b.gray@sms.ed.ac.uk',
      packages=['varout'],
      tests_require=['pytest'],
      # install reqs pending
      cmdclass={'test': PyTest},
)
