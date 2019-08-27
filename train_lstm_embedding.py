#! /usr/bin/env python

"""
Trains an LSTM embedding model on Wiki articles.

The data is freelly available here: https://www.wikidata.org/wiki/Wikidata:Database_download

NOTE: the data is in WIKI xml format and in order to extract text I recommend to use wikiextractor
(https://github.com/attardi/wikiextractor) This project adds "--json" param to have the data in json 
format.

A sample of the data is in "data/wiki/*" folder.

Run: python train_lstm_embedding.py <data_dir> <output_file>

data_dir - is the wiki data processed with "wikiextractor"
ouput_file - is trained model
"""

import os
import sys
import logging
import pickle
import numpy as np
from definitions import TOKENIZER_PATH
from code import data_processing
from code import lstm_embedding_model
from keras.preprocessing.text import Tokenizer

logging.basicConfig(level=logging.DEBUG)


def main(data_path, model_path):
    """
    Main driver function.

    args:
    data_path: str - path to wiki data
    model_path: str - path to save trained model
    """
    # Defining hyper-parameters
    sequence_length = 64
    word_dim = 64
    sentence_dim = 256



    logging.info("Preparing data generator.")
    tokenizer = data_processing.get_tokenizer(data_path, TOKENIZER_PATH)

    sequence_data = data_processing.get_sequence_generator(data_path, tokenizer)

    for i, sent in enumerate(sequence_data):
        print(sent)
        if i > 1:
            break

    num_words = len(tokenizer.word_index) + 1
    logging.debug("Vocabulary size is: %s", num_words)
    logging.info("Compiling the model.")
    lstm_autoencoder, lstm_embedding = lstm_embedding_model.get_models(vocab_size=num_words,
                                                                       input_length=sequence_length,
                                                                       word_dim=word_dim,
                                                                       sentence_dim=sentence_dim)


    temp_X_data = []

    for i, sent in enumerate(sequence_data):
        temp_X_data.append(sent)
        if i == 99:
            break

    logging.info("Training the model.")
    lstm_embedding_model.train_model(lstm_autoencoder, lstm_embedding, np.asarray(temp_X_data))

    # logging.info("Saving trained model to: %s", model_path)
    #
    # logging.info("Loading the model to test embeddings.")




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
    else:
        main(sys.argv[1], sys.argv[2])
