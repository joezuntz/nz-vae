import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .vae import save_model, Sampling

# Model with padding

def make_model_1_padding(verbose=False):
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(6, 79))
    x = tf.keras.layers.Reshape((6, 79, 1))(encoder_inputs)
    x = tf.keras.layers.ZeroPadding2D(padding=(3,0), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(64, (3, 21), activation='relu', padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3, 21), activation='relu', padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(12 * 79, activation="relu")(latent_inputs)
    x = layers.Reshape((12, 79, 1))(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(1, 3, activation="relu", strides=1, padding="same")(x)
    x = tf.keras.layers.CenterCrop(6, 79)(x)
    x = tf.keras.layers.Reshape((6, 79))(x)
    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    save_model('models/conv1-padding', encoder, decoder, latent_dim)

    if verbose:
        encoder.summary()
        decoder.summary()
    
# Model with padding
def make_model_1_no_padding(verbose=False):
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(6, 79))
    x = tf.keras.layers.Reshape((6, 79, 1))(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, (3, 21), activation='relu', padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3, 21), activation='relu', padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(6 * 79, activation="relu")(latent_inputs)
    x = layers.Reshape((6, 79, 1))(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(1, 3, activation="relu", strides=1, padding="same")(x)
    x = tf.keras.layers.Reshape((6, 79))(x)
    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    save_model('models/conv1-no-padding', encoder, decoder, latent_dim)
    if verbose:
        encoder.summary()
        decoder.summary()

def make_dense_model_1(verbose=False):
    encoder_inputs = keras.Input(shape=(6, 79))
    x = tf.keras.layers.Flatten()(encoder_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(6*79, activation='relu')(x)
    x = tf.keras.layers.Reshape((6,79))(x)
    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    save_model('models/dense-1', encoder, decoder, latent_dim)
    if verbose:
        encoder.summary()
        decoder.summary()

def make_dense_model_2(latent_dim = 2, verbose=False):
    encoder_inputs = keras.Input(shape=(6, 79))
    x = tf.keras.layers.Flatten()(encoder_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(6*79, activation='relu')(x)
    x = tf.keras.layers.Reshape((6,79))(x)
    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    save_model('models/dense-2', encoder, decoder, latent_dim)
    if verbose:
        encoder.summary()
        decoder.summary()

def make_dense_model_3(latent_dim, verbose=False):
    encoder_inputs = keras.Input(shape=(6, 79))
    x = tf.keras.layers.Flatten()(encoder_inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(6*79, activation='relu')(x)
    x = tf.keras.layers.Reshape((6,79))(x)
    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    save_model('models/dense-3', encoder, decoder, latent_dim)
    if verbose:
        encoder.summary()
        decoder.summary()

    
    
    
def make_all_models(verbose=False):
    make_model_1_no_padding(verbose=verbose)
    make_model_1_padded(verbose=verbose)
