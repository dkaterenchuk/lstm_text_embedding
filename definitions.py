#! /usr/bin/env python

"""
Defines global project variables.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(ROOT, "data/tokenizer.pickle")
WIKI_DATA = os.path.join(ROOT, "data/wiki/")
