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

import tensorflow as tf
import keras.backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(level=logging.INFO)


def main(data_path, model_path):
    """
    Main driver function.

    args:
    data_path: str - path to wiki data
    model_path: str - path to save trained model
    """

    # Defining hyper-parameters
    epochs = 1000  # how many times to go over your dataset
    # (batch_size * steps_per_epoch) = whole dataset for the generator
    batch_size = 1024  # batch size - in generator this is a single step
    steps_per_epoch = 12000  # number of steps defines your entire dataset in generator
    sequence_length = 64  # length of each text - in this case a sentence
    sentence_embedding_dim = 512  # size of the latent space
    workers = 4  # cores
    use_multiprocessing = True  # multiprocessing
    verbose = True  # verbose

    training_sentences = 10**6 # approximmate number of data
    
    # Steps to train an LSTM model
    logging.info("Preparing data generator.")
    w2v_model = data_processing.get_word_embedding_model(FASTTEXT_PATH)
    sequence_data_generator = data_processing.get_sequence_generator(data_path, w2v_model,
                                                                     sequence_length=sequence_length)

    logging.info("Initializing LSTM model")
    sample_sentence = next(sequence_data_generator)  # used for model initialization
    lstm_autoencoder = lstm_embedding_model.get_models(data_sample=sample_sentence,
                                                       lstm_embedding_dim=sentence_embedding_dim)

    logging.info("Loading the data.")
    
    train_data = []
    for i, sent in enumerate(sequence_data_generator):
        train_data.append(sent)
        if i == training_sentences:
            break
    
    #test_data = np.asarray(train_data[-100:])
    train_data = np.asarray(train_data)

    logging.info("Training the model.")
    # Example of how to fit a small data set (loading into ram)
    lstm_autoencoder = lstm_embedding_model.train_model(lstm_autoencoder,
                                                        train_data,
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        verbose=verbose)

    # # Train using a generator (when data cannot fit into ram)
    # data_generator = data_processing.get_batch_sequence_generator(data_path, w2v_model,
    #                                                               sequence_length=sequence_length,
    #                                                               batch_size=batch_size)

    # lstm_autoencoder = lstm_embedding_model.train_model_on_generator(lstm_autoencoder,
    #                                                                  data_generator,
    #                                                                  model_path=model_path,
    #                                                                  validation_data=(test_data, test_data),
    #                                                                  steps_per_epoch=steps_per_epoch,
    #                                                                  epochs=epochs,
    #                                                                  workers=workers,
    #                                                                  use_multiprocessing=use_multiprocessing,
    #                                                                  verbose=verbose)

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
