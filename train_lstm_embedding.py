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

import sys
import logging
import numpy as np
from code import data_processing
from code import lstm_embedding_model
from keras.models import load_model
from definitions import FASTTEXT_PATH, WORD2VEC_PATH

logging.basicConfig(level=logging.INFO)


def main(data_path, model_path):
    """
    Main driver function.

    args:
    data_path: str - path to wiki data
    model_path: str - path to save trained model
    """

    # Defining hyper-parameters
    batch_size = 128
    steps_per_epoch = 1000  # how many steps of batch_size to make during training
    epochs = 10
    verbose = True
    sequence_length = 64
    sentence_embedding_dim = 256

    # Steps to train an LSTM model
    logging.info("Preparing data generator.")
    w2v_model = data_processing.get_word_embedding_model(FASTTEXT_PATH)
    sequence_data_generator = data_processing.get_sequence_generator(data_path, w2v_model,
                                                                     sequence_len=sequence_length)

    logging.info("Intitializing LSTM model")
    sample_sentence = next(sequence_data_generator)  # used for model initialization
    lstm_autoencoder = lstm_embedding_model.get_autoencoder_model(data_sample=sample_sentence,
                                                                  lstm_embedding_dim=sentence_embedding_dim)

    logging.info("Training the model.")
    # NOTE: # Train with in-memory data if possible.
    # If the dataset can fit into RAM, load it first and then do the training.
    # limit = 1000
    # temp_x_data = np.asarray([sent for i, sent in enumerate(sequence_data_generator) if i == limit])
    # lstm_autoencoder = lstm_embedding_model.train_model(lstm_autoencoder, temp_x_data,
    #                                                     batch_size=batch_size,
    #                                                     epochs=epochs,
    #                                                     verbose=verbose)

    # Train with data generator (when the data does not fit into memory)
    data_generator = data_processing.get_autoencoder_sequence_generator(data_path, w2v_model,
                                                                        batch_size=batch_size,
                                                                        sequence_len=sequence_length)

    lstm_autoencoder = lstm_embedding_model.train_model_on_generator(lstm_autoencoder,
                                                                     data_generator,
                                                                     steps_per_epoch=steps_per_epoch,
                                                                     epochs=epochs,
                                                                     verbose=verbose)

    # Splitting autoencoder into encoder and decoder parts
    encoder, decoder = lstm_embedding_model.split_autoencoder(lstm_autoencoder)

    logging.info("Testing embedding.")
    logging.info("Initial sentence: %s",
                 data_processing.vector_sequence_to_words(sample_sentence, w2v_model))

    logging.info("Original sentence dimensions: %s", str(sample_sentence.shape))
    encoded_sent = encoder.predict(np.asarray([sample_sentence]))
    logging.info("Embedded sentence dimensions: %s", str(encoded_sent.shape))

    decoded_sent = decoder.predict(encoded_sent)
    logging.info("Reconstructed sentence: %s",
                 data_processing.vector_sequence_to_words(decoded_sent[0], w2v_model))

    logging.info("Saving trained model to: %s", model_path)
    lstm_autoencoder.save(model_path)

    logging.info("Loading the model to test embeddings.")
    loaded_lstm_autoencoder_model = load_model(model_path)
    loaded_lstm_autoencoder_model.summary()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
    else:
        main(sys.argv[1], sys.argv[2])
