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
from definitions import FASTTEXT_PATH, WORD2VEC_PATH

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
    sentence_embedding_dim = 256

    logging.info("Preparing data generator.")
    w2v_model = data_processing.get_word_embedding_model(FASTTEXT_PATH)
    sequence_data_generator = data_processing.get_sequence_generator(data_path, w2v_model,
                                                                     sequence_len=sequence_length)

    logging.info("Intitializing LSTM model")
    sample_sentence = next(sequence_data_generator)  # used for model initialization
    lstm_autoencoder, lstm_encoder, lstm_decoder = lstm_embedding_model.get_models(
        data_sample=sample_sentence, lstm_embedding_dim=sentence_embedding_dim)


    temp_X_data = []

    for i, sent in enumerate(sequence_data_generator):
        temp_X_data.append(sent)
        if i == 999:
            break

    logging.info("Training the model.")
    lstm_embedding_model.train_model(lstm_autoencoder, np.asarray(temp_X_data))

    logging.info("Testing embedding.")
    # for vect in sample_sentence:
    #     print(vect)
    #     word = w2v_model.most_similar(positive=[vect], topn=1)
    #     print(word)

    original_sent = " " .join([w2v_model.most_similar(positive=[vect], topn=1)[0][0] for vect in sample_sentence])
    logging.info("Initial sentence: %s", original_sent)

    logging.info("Original sentence dimensions are: %s", str(sample_sentence.shape))
    encoded_sent = lstm_encoder.predict(np.asarray([sample_sentence]))
    logging.info("Embedded dimensions: %s", str(encoded_sent.shape))

    decoded_sent = lstm_decoder.predict(encoded_sent)
    logging.info("Reconstructed sentence: %s",
                 " ".join([w2v_model.most_similar(positive=[vect], topn=1)[0][0] for vect in decoded_sent[0]]))


    logging.info("Saving trained model to: %s", model_path)

    logging.info("Loading the model to test embeddings.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
    else:
        main(sys.argv[1], sys.argv[2])
