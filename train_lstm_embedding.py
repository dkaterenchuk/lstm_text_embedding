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
from definitions import PATHS, HYPER_PARAM

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
#logging.basicConfig(level=logging.DEBUG)


def main(data_path, model_path):
    """
    Main driver function.

    args:
    data_path: str - path to wiki data
    model_path: str - path to save trained model
    """
    # Defining hyper-parameters
    epochs = HYPER_PARAM["epochs"]  # how many times to go over your dataset
    # (batch_size * steps_per_epoch) = whole dataset for the generator
    batch_size = HYPER_PARAM["batch_size"]  # batch size - in generator this is a single step
    sequence_length = HYPER_PARAM["sequence_length"]  # length of each text - in this case a sentence
    sentence_embedding_dim = HYPER_PARAM["sentence_embedding_dim"]  # size of the latent space
    workers = HYPER_PARAM["workers"]  # cores
    use_multiprocessing = HYPER_PARAM["use_multiprocessing"]  # multiprocessing
    verbose = HYPER_PARAM["verbose"]  # verbose
    training_sentences = 20 # * 10**6  # approximate number of sentences in wiki_small 
    steps_per_epoch = HYPER_PARAM["steps_per_epoch"]  # number of steps defines your entire dataset in generator    
    
    # Steps to train an LSTM model
    logging.info("Preparing data.")
    w2v_model = data_processing.get_word_embedding_model(PATHS["fasttext"])

    with open(data_path, "r") as f_reader:
        text_data = f_reader.readlines()

    # This cleans the data on the fly. Time consuming - advised to pre-process the data before-hand.
    # An example of pre-processing is defined in "code/wiki_preprocessing.py"
    #
    # sequence_data_generator = data_processing.get_sequence_generator(data_path, w2v_model,
    #                                                                  sequence_length=sequence_length)
    #
    # logging.info("Loading the data.")
    # train_data = []
    # for i, sent in enumerate(sequence_data_generator):
    #     train_data.append(sent)
    #     if i == training_sentences:
    #         break
    # train_data = np.asarray(train_data)

    # train_data = data_processing.get_preprocessed_data(data_path, w2v_model, sent_length=sequence_length)
    # test_data = train_data[:100]
    # train_data = train_data[:100]
    # print(train_data.shape)
    #
    # logging.info("Compiling LSTM model.")
    # sample_sentence = train_data[0]  # used for model initialization
    # lstm_autoencoder = lstm_embedding_model.get_models(data_sample=sample_sentence,
    #                                                    lstm_embedding_dim=sentence_embedding_dim)
    #
    # logging.info("Training the model.")
    # # Example of how to fit a small data set (loading into ram and train)
    # lstm_autoencoder = lstm_embedding_model.train_model(lstm_autoencoder,
    #                                                     train_data,
    #                                                     validation_data=test_data,
    #                                                     model_path=model_path,
    #                                                     batch_size=batch_size,
    #                                                     epochs=epochs,
    #                                                     verbose=verbose)

    # data_generator = data_processing.get_batch_preprocessed_data_generator(data_path,
    #                                                                        w2v_model,
    #                                                                        sequence_length=sequence_length,
    #                                                                        batch_size=batch_size)

    logging.info("Compiling a network.")
    sent_generator = data_processing.get_preprocessed_data(text_data, w2v_model, sent_length=sequence_length)
    sample_sentence = next(sent_generator)   # used for model initialization

    # TODO: remove break
    # sys.exit()
    

    logging.debug("Sample shape: %s", sample_sentence.shape)
    lstm_autoencoder = lstm_embedding_model.get_models(data_sample=sample_sentence,
                                                       lstm_embedding_dim=sentence_embedding_dim)


    logging.debug("Creating a data generator")
    # Train using a generator (when data cannot fit into ram)
    data_generator = data_processing.get_batch_preprocessed_data_generator(text_data,
                                                                           w2v_model,
                                                                           sequence_length=sequence_length,
                                                                           batch_size=batch_size)

    logging.info("Loading test data.")
    test_data = []
    for i, sent in enumerate(sent_generator):
        test_data.append(sent)
        if i == 100:
            break
        
    test_data = np.asarray(test_data)

    # train_data = []
    # for i, sent_batch in enumerate(data_generator):
    #     train_data.extend(sent_batch)
    #     if i == 100:
    #         break
        
    # train_data = np.asarray(train_data)
    # print("trai_data.shape", train_data.shape)

    
    # TODO: remove break
    # sys.exit()
    
    logging.info("Training the model")
    lstm_autoencoder = lstm_embedding_model.train_model_on_generator(lstm_autoencoder,
                                                                     data_generator,
                                                                     model_path=model_path,
                                                                     # validation_data=(test_data, test_data),
                                                                     steps_per_epoch=steps_per_epoch,
                                                                     epochs=epochs,
                                                                     workers=workers,
                                                                     use_multiprocessing=use_multiprocessing,
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
