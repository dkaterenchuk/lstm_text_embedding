#! /usr/bin/env python

"""
Defines an LSTM model for training sentence autoencoder and encoder
"""
import os
import logging
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers import RepeatVector, Bidirectional, TimeDistributed, Dense, Input

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_autoencoder_model(data_sample, lstm_embedding_dim=256):
    """
    Defines model architecture and hyper-parameters.

    :return: lstm_autoencoder, lstm_encoder
    """
    sequence_length, word_embedding_dim = data_sample.shape

    # encoder
    input_layer = Input(shape=(sequence_length, word_embedding_dim))
    lstm_encoder_layer = LSTM(lstm_embedding_dim, return_sequences=False, activation="tanh")(input_layer)

    # decoder
    repeat_layer = RepeatVector(sequence_length)(lstm_encoder_layer)
    lstm_decoder_layer = LSTM(lstm_embedding_dim, return_sequences=True, activation="tanh")(repeat_layer)
    dense_layer = TimeDistributed(Dense(word_embedding_dim, activation="tanh"))(lstm_decoder_layer)

    # model
    lstm_autoencoder = Model(input_layer, dense_layer)
    lstm_autoencoder.compile(optimizer='adam', loss='mse')

    return lstm_autoencoder


def split_autoencoder(autoencoder):
    """

    :param autoencoder: LSTM autonecoder
    :return: encoder model
    :return: decoder model
    """
    encoder_input = Input(shape=(autoencoder.layers[0].input_shape[1:]))
    encoder_layer = autoencoder.layers[1](encoder_input)
    encoder_model = Model(encoder_input, encoder_layer)

    logging.debug("Encoder last layer output: %s", str(autoencoder.layers[1].output_shape))
    decoder_input = Input(shape=(autoencoder.layers[1].output_shape[1],))
    decoder_layer = autoencoder.layers[2](decoder_input)
    decoder_layer = autoencoder.layers[3](decoder_layer)
    decoder_layer = autoencoder.layers[4](decoder_layer)
    decoder_model = Model(decoder_input, decoder_layer)

    return encoder_model, decoder_model


def train_model(lstm_autoencoder, data, batch_size=16, epochs=100, verbose=True):
    """

    :param lstm_autoencoder: keras model
    :param sequence_generator: data generator
    :return: trained models
    """
    lstm_autoencoder.fit(data, data,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose)

    return lstm_autoencoder


def train_model_on_generator(lstm_autoencoder, data_generator, steps_per_epoch,
                             epochs=100, use_multiprocessing=True, verbose=True):
    """

    :param lstm_autoencoder: obj - encoder model defined in get_autoencoder_model
    :param data_generator: generator - dataset
    :param steps_per_epoch: int - number of steps
    :param epochs: int - number of epochs
    :param use_multiprocessing: bool - use multiprocessing
    :param verbose: bool - verbose or not
    :return: lstm_autoencoder - trained model
    """
    lstm_autoencoder.fit_generator(data_generator,
                                   steps_per_epoch=2,
                                   epochs=2,
                                   workers=-1,
                                   use_multiprocessing=True,
                                   verbose=True)

    # lstm_autoencoder.fit_generator(data_generator,
    #                                steps_per_epoch=steps_per_epoch,
    #                                epochs=epochs,
    #                                workers=-1,
    #                                use_multiprocessing=use_multiprocessing,
    #                                verbose=verbose)

    return lstm_autoencoder
