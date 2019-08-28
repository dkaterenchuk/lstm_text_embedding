#! /usr/bin/env python

"""
Defines an LSTM model for training sentence autoencoder and encoder
"""
import os
import logging
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.layers import RepeatVector, Bidirectional, TimeDistributed, Dense, Input, Flatten

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_models(data_sample, lstm_embedding_dim=256):
    """
    Defines model architecture and hyper-parameters.

    :return: lstm_autoencoder, lstm_encoder
    """
    print(data_sample.shape)
    sequence_length, word_embedding_dim = data_sample.shape

    # encoder
    input_layer = Input(shape=(sequence_length, word_embedding_dim))
    lstm_encoder_layer = LSTM(lstm_embedding_dim, return_sequences=False, activation="tanh")(input_layer)

    # decoder
    repeat_layer = RepeatVector(sequence_length)(lstm_encoder_layer)
    lstm_decoder_layer = LSTM(lstm_embedding_dim, return_sequences=True, activation="tanh")(repeat_layer)
    dense_layer = TimeDistributed(Dense(word_embedding_dim, activation="tanh"))(lstm_decoder_layer)

    # models
    lstm_autoencoder = Model(input_layer, dense_layer)
    lstm_encoder = Model(input_layer, lstm_encoder_layer)

    decoder_input = Input(shape=(lstm_embedding_dim,))
    decoder_layer = lstm_autoencoder.layers[-3](decoder_input)
    decoder_layer = lstm_autoencoder.layers[-2](decoder_layer)
    decoder_layer = lstm_autoencoder.layers[-1](decoder_layer)
    lstm_decoder = Model(decoder_input, decoder_layer)


    lstm_autoencoder.compile(optimizer='adam', loss='mse')

    print(lstm_autoencoder.summary())

    return lstm_autoencoder, lstm_encoder, lstm_decoder



def train_model(lstm_autoencoder, sequence_generator):
    """

    :param lstm_autoencoder: keras model
    :param lstm_encoder: keras model
    :param sequence_generator: data generator
    :return: trained models
    """
    print(sequence_generator.shape)
    lstm_autoencoder.fit(sequence_generator, sequence_generator, batch_size=16, epochs=100, verbose=True)


    #lstm_autoencoder.fit_generator(sequence_generator, steps_per_epoch=8, epochs=1, workers=4, use_multiprocessing=True)


def main():
    """
    Test and fun with LSTMs
    :return:
    """

    docs = ['Well done!',
            'Good work',
            'Great (good) effort!',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor-work',
            'Could have done better.']

    # # define class labels
    # labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    #
    # vocab_size = 50
    # max_length = 4
    # word_dim = 8
    #
    # tokenizer = Tokenizer(num_words=vocab_size, filters="")
    # tokenizer.fit_on_texts(docs)
    #
    # tokenized_docs = pad_sequences(tokenizer.texts_to_sequences(docs), maxlen=max_length, padding="pre")
    #
    # print(tokenizer.sequences_to_texts(tokenized_docs))
    #
    # # define the model
    # input_layer = Input(shape=(max_length,))
    # embedding_layer = Embedding(vocab_size, word_dim, input_length=max_length)(input_layer)
    # lstm_layer = LSTM(32, return_sequences=False)(embedding_layer)
    #
    # repeat_layer = RepeatVector(max_length)(lstm_layer)
    #
    # decoder_layer = LSTM(16, return_sequences=True)(repeat_layer)
    #
    # output_layer = TimeDistributed(Dense(4, activation='tanh'))(decoder_layer)
    #
    # model = Model(input_layer, output_layer)
    #
    # # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # # summarize the model
    # print(model.summary())
    #
    # # fit the model
    # model.fit(tokenized_docs, labels, batch_size=2, epochs=5, verbose=1)
    # model.fit_generator(epochs=1,workers=1, use_multiprocessing=True)
    # # evaluate the model
    # loss, accuracy = model.evaluate(tokenized_docs, labels, verbose=0)
    # print('Accuracy: %f' % (accuracy * 100))


if __name__ == "__main__":
    main()
