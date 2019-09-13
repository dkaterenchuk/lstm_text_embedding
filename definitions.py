#! /usr/bin/env python

"""
Defines global project paths and configurations.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
WIKI_DATA = os.path.join(ROOT, "data/wiki/")
WORD2VEC_PATH = os.path.join(ROOT, "data/word_embeddings/wiki_dump_128dim_word2vec.model")
FASTTEXT_PATH = os.path.join(ROOT, "data/word_embeddings/wiki_dump_128dim_fasttext.model")
FASTTEXT_PATH = "/scratch/denys/fasttext/wiki-news-300d-1M-subword.vec"

PATHS = {
    "root": os.path.dirname(os.path.abspath(__file__)),
    "fasttext": FASTTEXT_PATH
}

# Defining LSTM hyper-parameters
HYPER_PARAM = {
    "epochs": 1, # how many times to go over your dataset
    # (batch_size * steps_per_epoch) = whole dataset for the generator
    "batch_size": 16,  # 1024, # batch size - in generator this is a single step
    "steps_per_epoch": 12000,  # number of steps defines your entire dataset in generator
    "sequence_length": 64,  # length of each text - in this case a sentence
    "sentence_embedding_dim": 512,  # size of the latent space
    "workers": 4,  # cores
    "use_multiprocessing": True,  # multiprocessing
    "verbose": True  # verbose
}
