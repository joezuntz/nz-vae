import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import matplotlib.pyplot as plt
from .vae import save_model, Sampling

def make_conv_model(nbin, nz, ndata, latent_dim, verbose=False):
    encoder_inputs = keras.Input(shape=(nbin, nz))
    x = tf.keras.layers.Reshape((nbin, nz, 1))(encoder_inputs)
    x = layers.Conv2D(32, 3, strides=1, padding="same")(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64)(decoder_inputs)
    x = layers.Reshape((64, 1))(x)
    x = layers.Conv1DTranspose(64, 3, padding="same")(x)
    x = layers.Conv1DTranspose(32, 3, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(ndata)(x)
    decoder_outputs = x    

    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")


    if verbose:
        encoder.summary()
        decoder.summary()

    return encoder, decoder


class LAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim,  kl_weight=1, **kwargs):
        super(LAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @staticmethod
    def normalize_nz(nz):
        return tf.linalg.normalize(nz, axis=(2))[0]

    def train_step(self, data):
        nzs, data_vectors = data
        with tf.GradientTape() as tape:
            # encode and decode again
            z_mean, z_log_var, z = self.encoder(nzs)
            predicted_data_vectors = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data_vectors - predicted_data_vectors), axis=1))

            # mean of total square errors
            # loss from how different the z values are from a normal distribution
            kl_loss1 = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss1, axis=1))  * self.kl_weight
            
            # Total is the sum of these two
            total_loss = reconstruction_loss + kl_loss

        # do the training step update
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    
    def fit_to_nz_data(self, nz_data, likelihoods, epochs, batch_size=2400):
        self.compile(optimizer=keras.optimizers.Adam())
        history = self.fit((nz_data, likelihoods), epochs=epochs, batch_size=batch_size, verbose=0)        
        return history

    def plot_history(self, history, log=False):
        plotter = plt.semilogy if log else plt.plot
        plt.figure()
        plotter(history.history['kl_loss'])
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss")
        plt.figure()
        plotter(history.history['reconstruction_loss'])
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.figure()
        plotter(history.history['loss'])
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")        
    
    
