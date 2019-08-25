#! /usr/bin/env python3

"""
Loads and preprocesses Wikipedia data.

This script is meant to be run as a module.
"""

import os
import sys


# Converts the unicode file to ascii
def unicode_to_ascii(sentence):
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", sentence)
    sentence = sentence.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    sentence = '<start> ' + sentence + ' <end>'

    return sentence


def main(data_path):
    """
    Data processing test function.

    args data_path: path to wiki data 
    """

    



if __name__ == "__main__":
    if len(sys.argv) != 2:
        pritn(__doc__)
    else:
        main(sys.argv[1])
