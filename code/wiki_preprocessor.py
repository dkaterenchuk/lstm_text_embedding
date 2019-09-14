#! /usr/bin/env python

"""
Pre-processed wiki articles and saves clean padded sentences into a txt file

Usage:
    python wiki_preprocessor.py <data_path> <output_model>

Args:
    data_path - path to wiki data
    output_model - path to save trained model
"""

import sys
import logging
import data_processing

logging.basicConfig(level=logging.INFO)


def main(data_path, output_path, training_algorithm="word2vec"):
    """
    Main function to execute the training.

    :param data_path: str - path to the data
    :param output_path: str - output path
    :return: None
    """
    sentence_limit = 64
    logging.info("Loading data.")

    with open(output_path, "a") as f_writer:
        for sent in data_processing.get_text_generator(data_path,sentence_tags=True, pad=sentence_limit):
            f_writer.write(" ".join(sent) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
