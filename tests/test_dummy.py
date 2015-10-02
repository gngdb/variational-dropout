#!/usr/bin/env python

########################################
# Tests for variational dropout layers #
########################################

import unittest
import pytest

class TestDummy(unittest.TestCase):
    """
    To test the switching dropout functionality of the switching dropout
    layer. 
    """
    @classmethod
    def setUpClass(self):
        self.something = "something"
    def test_get_output_for(self):
        assert self.something == "something"

def main():
    unittest.main()

if __name__ == "__main__":
    main()
