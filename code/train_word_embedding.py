#! /usr/bin/env python

"""
Trains word embedding model on the data. This is often beneficial to the model.

Usage:
    python train_word_embedding.py <data_path> <output_model>

Args:
    data_path - path to wiki data
    output_model - path to save trained model
"""

import sys
import logging
import data_processing
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText


def main(data_path, output_path, training_algorithm="fasttext"):
    """
    Main function to execute the training.

    :param data_path: str - path to the data
    :param output_path: str - output path
    :param training_algorithm: str - fasttext or word2vec (default) training algorithm
    :return: None
    """
    logging.info("Crating data generator.")
    text_data = [x for x in data_processing.get_word_sequences_generator(data_path)]

    logging.info("Training the model")
    if training_algorithm == "fasttext":
        logging.info("Using FastText algorithm")
        word_embedding_model = FastText(size=128, window=5, min_count=2, iter=5, workers=4, sg=0)
    else:
        word_embedding_model = Word2Vec(size=128, window=5, min_count=2, iter=5, workers=4, sg=0)

    word_embedding_model.build_vocab(text_data)
    word_embedding_model.train(text_data, total_examples=word_embedding_model.corpus_count, epochs=10)

    logging.info("Saving the model to: %s", output_path)
    word_embedding_model.save(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
