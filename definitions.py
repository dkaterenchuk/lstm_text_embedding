#! /usr/bin/env python

"""
Defines global project variables.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
WIKI_DATA = os.path.join(ROOT, "data/wiki/")
WORD2VEC_PATH = os.path.join(ROOT, "data/word_embeddings/wiki_dump_128dim_word2vec.model")
FASTTEXT_PATH = os.path.join(ROOT, "data/word_embeddings/wiki_dump_128dim_fasttext.model")
