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


logging.basicConfig(level=logging.INFO)


def main(data_path, output_path, training_algorithm="word2vec"):
    """
    Main function to execute the training.

    :param data_path: str - path to the data
    :param output_path: str - output path
    :param training_algorithm: str - training algorithm (word2vec/fastText)
    :return: None
    """
    sentence_limit = 64
    size = 128
    window = 5
    min_count = 10
    iterations = 5
    workers = 8
    sg = 0

    logging.info("Loading data.")
    text_data = [x for x in data_processing.get_text_generator(data_path, sentence_tags=True, pad=sentence_limit)]

    logging.info("Training the model")
    if training_algorithm == "fasttext":
        logging.info("Using FastText algorithm")
        word_embedding_model = FastText(size=size, window=window, min_count=min_count, iter=iterations, workers=workers, sg=sg)
    else:
        word_embedding_model = Word2Vec(size=size, window=window, min_count=min_count, iter=iterations, workers=workers, sg=sg)

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
