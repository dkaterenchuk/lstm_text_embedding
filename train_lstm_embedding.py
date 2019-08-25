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
from code import data_processing

logging.basicConfig(level=logging.DEBUG)


def main(data_path, model_path):
    """
    Main driver function.

    args:
    data_path: str - path to wiki data
    model_path: str - path to save trained model
    """
    logging.info("Preparring data generator.")
    data_generator = data_processing.get_generator(data_path)
    
    logging.info("Compiling the model.")

    logging.info("Training the model.")

    logging.info("Saving trained model to: %s", model_path)
    


if __name__ == "__maing__":
    if len(sys.argv) != 3:
        print(__doc__)
    else:
        main(sys.argv[1], sys.argv[2]))
