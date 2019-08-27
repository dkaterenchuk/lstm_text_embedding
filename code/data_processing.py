#! /usr/bin/env python3

"""
Loads and preprocesses Wikipedia data.

This script is meant to be run as a module. However, it contains a main function that is created
to test the logic of processing textual data. This script designed to work with Wikipedia data.
In case the text comes from a different source, "get_generator" function needs to be updated.

Run: python data_processing.py <data_dir>

Args:
    data_dir - str: a path to wiki directory ("data/wiki/")
"""

import os
import re
import sys
import json
import spacy
import pickle
import logging
import unicodedata
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


spacy_nlp = spacy.load("en_core_web_sm")


def unicode_to_ascii(sentence):
    """
    Converts the unicode file to ascii
    :param sentence: str/unicode
    :return: str
    """
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence, sentence_tags=False):
    """
    Cleans a text by remove punctuation keeping ",.!?()", words, and digits.

    :param sentence: str - a sentence
    :param sentence_tags: adds <start> and <end> tags to define sentence boundaries
    :return: sentence - clean sentence
    """
    sentence = unicode_to_ascii(sentence.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sentence = re.sub(r"([?.!,\(\)])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "(", ")")
    sentence = re.sub(r"[^a-zA-Z0-9?.!,\(\)]+", " ", sentence)
    sentence = sentence.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    if sentence_tags:
        sentence = '<start> ' + sentence + ' <end>'

    return sentence


def process_document(document_text):
    """
    Splits a wiki article into sentences and cleans each sentence.
    :param document_text: string
    :return: sentence_list
    """
    sentence_list = []

    doc = spacy_nlp(document_text)

    for sent in doc.sents:
        clean_sentence = preprocess_sentence(sent.text)
        sentence_list.append(clean_sentence)

    return sentence_list


def get_tokenizer(data_path, tokenizer_path):
    """

    :param data_path:
    :return:
    """

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as file_reader:
            tokenizer = pickle.load(file_reader)
    else:
        logging.info("Training a tokenizer on the data")
        data_generator = get_text_generator(data_path)
        tokenizer = Tokenizer(filters="", oov_token="<oov>")
        tokenizer.fit_on_texts(data_generator)

        logging.info("Saving the tokenizer to: %s", tokenizer_path)
        with open(tokenizer_path, "wb") as file_writer:
            pickle.dump(tokenizer, file_writer, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


def get_text_generator(data_path):
    """
    Reads Wiki data dump, cleans the data and returns a generator of sentences

    :param data_path: str -  path to wiki corpus
    :return: data_generator
    """
    for folder in os.listdir(data_path):
        logging.debug("Data path at folder: %s", os.path.join(data_path, folder))
        for doc in os.listdir(os.path.join(data_path, folder)):
            logging.debug("A file path: %s", os.path.join(data_path, folder, doc))
            with open(os.path.join(data_path, folder, doc), "r") as f_reader:
                wiki_doc = f_reader.readlines()

            for line in wiki_doc:
                wiki_text = json.loads(line)["text"]

                for sent in process_document(wiki_text):
                    yield sent


def get_word_sequences_generator(data_path):
    """
    Returns word lists - splits the words.
    Wrapper for "get_text_generator".

    :param data_path: path to the data
    :return: lists of words
    """
    for sent in get_text_generator(data_path):
        yield sent.split(" ")


def get_sequence_generator(data_path, tokenizer, batch_size=64, sequence_len=64):
    """
    Wrapper for "get_text_generator" that adds work to int mapping.

    :param data_path: str - path to wiki corpus
    :param tokenizer: obj - trained tokenizer
    :param sequence_len: int - max sequence length
    :return: generator obj - sequences of word integers
    """
    data_matrix = []
    for sent in pad_sequences(
            tokenizer.texts_to_sequences(
                get_text_generator(data_path)), maxlen=sequence_len, padding="pre"):
        yield sent

        # data_matrix.append((sent, sent))
        # if len(data_matrix) == batch_size:
        #     batch = np.asarray(data_matrix)
        #     data_matrix = []
        #
        #     yield batch



def main(data_path):
    """
    Data processing test function.

    :params data_path: path to wiki data
    """
    logging.debug("Loading sentences")
    data_generator = get_text_generator(data_path)

    logging.debug(data_generator)

    for i, sentence in enumerate(data_generator):
        logging.debug(i, sentence)
        logging.debug("%n: sentence is: %s:", i, sentence)
        if i == 1000:
            break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
    else:
        main(sys.argv[1])
