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
import logging
import unicodedata
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors


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
    Cleans a text by removing some punctuation keeping ",.!?()", words, and digits.
    NOTE: punctuation can be relevant to some tasks (Named Entity Recognition).
    Modify this function that suits your needs.

    :param sentence: str - a sentence
    :param sentence_tags: adds <start> and <end> tags to define sentence boundaries
    :return: sentence - clean sentence
    """
    sentence = unicode_to_ascii(sentence.lower().strip())

    sentence = re.sub(r"([?.!,\(\)])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    sentence = re.sub(r"[^a-zA-Z0-9?.!,\(\)]+", " ", sentence)
    sentence = sentence.rstrip().strip()

    if sentence_tags:
        sentence = '<start> ' + sentence + ' <end>'

    return sentence


def process_document(document_text, sentence_tags=False):
    """
    Splits a wiki article into sentences and cleans each sentence.
    :param document_text: string
    :return: sentence_list
    """
    doc = spacy_nlp(document_text)

    for sent in doc.sents:
        clean_sentence = preprocess_sentence(sent.text, sentence_tags=sentence_tags)
        yield clean_sentence


def get_pos_sentences(document_text):
    """
    Splits a wiki article into sentences and cleans each sentence.
    :param document_text: string
    :return: sentence_list
    """
    doc = spacy_nlp(document_text)

    for sent in doc.sents:
        yield [word.pos_ for word in sent]


def get_word_embedding_model(word_embedding_path):
    """
    Loads a word embedding model. Chooses a correct algorithm based on the file name:
    "*word2vec.model" vs "*fasttext.model",

    NOTE: The model can be trained with the auxiliary script "train_word_embedding.py".

    :param word_embedding_path: str - path to a trained model
    :return: obj - word embedding model
    """
    if "fasttext.model" in word_embedding_path:
        w2v_model = FastText.load(word_embedding_path)
    elif "wiki-news-300d-1M-subword.vec" in word_embedding_path:
        w2v_model = KeyedVectors.load_word2vec_format(word_embedding_path)
    else:
        w2v_model = Word2Vec.load(word_embedding_path)

    return w2v_model


def pad_text(word_list, pad=None):
    """
    Pads word list with "<pad>" tags at the beginning.
    NOTE: We add at most 5 <pad>'s to save the space

    :param word_list: list - words
    :param pad: ing - max length of a sentence
    """
    logging.debug("Pad text size: %s", " ".join(word_list))

    if len(word_list) > pad:
        word_list = word_list[:pad]
    else:
        delta = pad - len(word_list)
        word_list = ["<pad>"] * min(delta, 5) + word_list  # padding
        #word_list = ["<pad>"] * delta + word_list  # padding

    return word_list


def get_text_generator(data_path, sentence_tags=False, pad=False):
    """
    Reads Wiki data dump, cleans the data and returns a generator of sentences

    :param data_path: str -  path to wiki corpus
    :return: data_generator
    """
    for folder in os.listdir(data_path):
        logging.debug("Data path at folder: %s", os.path.join(data_path, folder))
        print(folder)
        for doc in os.listdir(os.path.join(data_path, folder)):
            logging.debug("A file path: %s", os.path.join(data_path, folder, doc))
            print(doc)
            with open(os.path.join(data_path, folder, doc), "r") as f_reader:
                wiki_doc = f_reader.readlines()
            for line in wiki_doc:
                wiki_text = json.loads(line)["text"]
                for sent in process_document(wiki_text, sentence_tags=sentence_tags):
                    logging.debug("From get_text_generator: %s", sent)
                    sent = sent.split(" ")
                    if pad:
                        sent = pad_text(sent, pad=pad)

                    yield sent


def pad_sequence_with_zeros(sequence_list, length_limit):
    """
    Padding a sequence (a text in w2v representation of any length) with zero vectors.
    This make the input to have the same length (different document length is standardized).

    :param sequence_list: - np array of np arrays
    :param length_limit: int - max length of a sequence
    :return sequence_list - padded sequences to the same length
    """
    n_dim = len(sequence_list.shape)
    logging.debug("Sentence dimension: %s", sequence_list.shape)

    if sequence_list.shape[0] > length_limit:
        sequence_list = sequence_list[:length_limit]
    elif sequence_list.shape[0] < length_limit:
        pad = length_limit - sequence_list.shape[0]
        if n_dim == 2:
            sequence_list = np.pad(sequence_list, [(pad, 0), (0, 0)], "constant")
        else:
            sequence_list = np.pad(sequence_list, [(pad), (0)], "constant")

    return np.asarray(sequence_list)


def get_sequence_generator(data_path, w2v_model, sequence_length=64):
    """
    Wrapper for "get_text_generator" that adds work to int mapping.

    :param data_path: str - path to wiki corpus
    :param w2v_model: obj - trained word embedding model
    :param sequence_length: int - max sequence length
    :return: generator obj - sequences of word integers
    """
    while True:
        for sent in get_text_generator(data_path):
            sent_sequence = [w2v_model[w] for w in sent if w in w2v_model]
            if len(sent_sequence) > 0:
                yield pad_sequence_with_zeros(np.asarray(sent_sequence), length_limit=sequence_length)


def get_batch_sequence_generator(data_path, w2v_model, sequence_length=64, batch_size=32):
    """
    Wrapper for "get_sequence_generator" that adds batches of data.

    :param data_path: str - path to wiki corpus
    :param w2v_model: obj - trained word embedding model
    :param sequence_length: int - max sequence length
    :param batch_size: int - number of instances per batch
    :return: generator obj - sequences of word integers
    """
    batch = []
    for sent in get_sequence_generator(data_path, w2v_model, sequence_length=sequence_length):
        batch.append(sent)
        if len(batch) == batch_size:
            complete_batch = np.asarray(batch)
            batch = []
            yield complete_batch, complete_batch


def pad_sequence(vector_sequence, pad_vector, sent_length=64):
    """
    Padding a sequence with pad vectors.

    :param vector_sequence: np matrix
    :param pad_vector: np vector
    :param sent_length: int - length of the sequence
    :return:
    """
    delta = sent_length - vector_sequence.shape[0]

    return np.insert(vector_sequence, [0] * delta, pad_vector, axis=0)


# def get_preprocessed_data(data_path, w2v_model, sent_length=64):
#     """
#     Reads pre-processed wiki corpus file by line and returns sentence vector representations.
#
#     :param data_path: str - path to the data
#     :param w2v_model: w2v_model
#     :param sent_length: int - length of the sequence
#     :return:
#     """
#     print(data_path)
#     while True:
#         with open(data_path, "r") as f_reader:
#             for line in f_reader:
#                 line = line.strip("\n").split()
#
#                 v = [w2v_model.wv[w] for w in line if w in w2v_model.wv]
#                 vector_sequence = np.asarray(v)
#                 if vector_sequence.shape[0]:  # empty sentence
#                     vector_sequence = pad_sequence(vector_sequence, w2v_model.wv["<pad>"], sent_length=sent_length)
#                     yield vector_sequence

def get_preprocessed_data(text_data, w2v_model, sent_length=64):
    """
    Reads pre-processed wiki corpus file by line and returns sentence vector representations.

    :param data_path: str - path to the data
    :param w2v_model: w2v_model
    :param sent_length: int - length of the sequence
    :return:
    """
    while True:
        for line in text_data:
            line = line.strip("\n").split()

            v = [w2v_model.wv[w] for w in line if w in w2v_model.wv]
            vector_sequence = np.asarray(v)
            if vector_sequence.shape[0]:  # empty sentence
                vector_sequence = pad_sequence(vector_sequence, w2v_model.wv["<pad>"], sent_length=sent_length)
                yield vector_sequence


def get_batch_preprocessed_data_generator(text_data, w2v_model, sequence_length=64, batch_size=32):
    """
    Wrapper for "get_sequence_generator" that adds batches of data.

    :param data_path: str - path to wiki corpus
    :param w2v_model: obj - trained word embedding model
    :param sequence_length: int - max sequence length
    :param batch_size: int - number of instances per batch
    :return: generator obj - sequences of word integers
    """
    batch = []
    for sent in get_preprocessed_data(text_data, w2v_model, sent_length=sequence_length):
        batch.append(sent)
        if len(batch) == batch_size:
            complete_batch = np.asarray(batch)
            batch = []
            yield complete_batch, complete_batch 


def vector_sequence_to_words(sequence, w2v_model):
    """
    Reconstructs a sentence from an array of word vectors.

    :param sequence: np array of word vectors (matrix)
    :param w2v_model: word embedding model
    :return: str
    """

    return " ".join([w2v_model.most_similar(positive=[vect], topn=1)[0][0] for vect in sequence])


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
