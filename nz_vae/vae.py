import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import matplotlib.pyplot as plt

# The new kind of layer that we need for sampling
class Sampling(layers.Layer):
    """
    This is a new layer type that is added to the 
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
def save_model(filename, encoder, decoder, latent_dim):
    with open(filename + '.encoder.json', 'w') as f:
        f.write(encoder.to_json())
    with open(filename + '.decoder.json', 'w') as f:
        f.write(decoder.to_json())
    with open(filename + '.info.json', 'w') as f:
        json.dump({'latent_dim':latent_dim}, f)

def load_model(filename):
    with open(filename + '.encoder.json', 'r') as f:
        encoder = keras.models.model_from_json(f.read(), custom_objects={'Sampling':Sampling})
    with open(filename + '.decoder.json', 'r') as f:
        decoder = keras.models.model_from_json(f.read())
    with open(filename + '.info.json', 'r') as f:
        d = json.load(f)
        latent_dim = d['latent_dim']
    return encoder, decoder, latent_dim

def load_data():
    nz_data = np.loadtxt("data/realizations.txt.gz").reshape((2400, 6, 79)).astype(np.float32)
    return VAE.normalize_nz(nz_data)
    return nz_data

class VAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim,  kl_weight=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
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
        data = self.normalize_nz(data)
        with tf.GradientTape() as tape:
            # encode and decode again
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction = self.normalize_nz(reconstruction)
            
            # loss value from how different our reconstruction
            # looks from the data
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction, axis=(1, 2))
                )
            )
            
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
    
    def generate_nz_draws(self, n):
        z_sample = np.random.normal(size=(n, self.latent_dim))
        nz_generated = self.decoder.predict(z_sample)

        return self.normalize_nz(nz_generated)
    
    def fit_to_nz_data(self, nz_data, epochs, batch_size=2400):
        self.compile(optimizer=keras.optimizers.Adam())
        history = self.fit(nz_data, epochs=epochs, batch_size=batch_size, verbose=0)        
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
    
    
    
class VAE2(VAE):
    def train_step(self, data):
        data = self.normalize_nz(data)
        with tf.GradientTape() as tape:
            # encode and decode again
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction = self.normalize_nz(reconstruction)
            
            reconstruction_loss = tf.reduce_sum(tf.square(tf.subtract(reconstruction, data)))

            # loss from how different the z values are from a normal distribution
            kl_loss1 = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss1, axis=1))
            
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