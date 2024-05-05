from typing import Tuple

import numpy as np
from tensorflow.keras import layers, models, Model
from tensorflow.keras.activations import silu
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from marketmimic.constants import LATENT_DIM, DISCRIMINATOR_LEARNING_RATE, GENERATOR_LEARNING_RATE, SEQUENCE_LENGTH, \
    BETA_1, BETA_2, GAN_SIZE
from marketmimic.metric import *


class SplitLayer(layers.Layer):
    def __init__(self, index_start, index_end, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.index_start = index_start
        self.index_end = index_end

    def call(self, inputs):
        return inputs[:, :, self.index_start:self.index_end]

    def get_config(self):
        config = super(SplitLayer, self).get_config()
        config.update({
            "index_start": self.index_start,
            "index_end": self.index_end
        })
        return config


def build_generator(latent_dim: int = LATENT_DIM) -> models.Model:
    """
    Builds and returns the generator model with LSTM layers, specialized for separate handling
    of price and volume features, with intermediate cross-communication between the two branches.

    Args:
        latent_dim: Dimension of the latent space (input noise vector).

    Returns:
        A TensorFlow Keras model representing the generator with specialized branches for
        price and volume that exchange information.
    """
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, latent_dim))

    price_path = SplitLayer(0, 1)(input_layer)
    volume_path = SplitLayer(1, 2)(input_layer)

    join_path = layers.Concatenate()([price_path, volume_path])
    join_path = layers.Dense(int(GAN_SIZE * 32), name='Dense_Join_1', kernel_initializer=HeNormal(), activation=silu,
                             kernel_regularizer=l2(0.01))(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32), activation=silu)(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32), activation=silu)(join_path)
    join_path = layers.TimeDistributed(layers.Dense(int(GAN_SIZE * 16), activation=silu))(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32), activation=silu)(join_path)
    join_path = BatchNormalization()(join_path)

    # Initial Common LSTM Layer
    price_path = layers.LSTM(int(GAN_SIZE * 128), kernel_initializer=HeNormal(), return_sequences=True, activation=silu,
                             kernel_regularizer=l2(0.01))(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 128), activation=silu, kernel_initializer=HeNormal())(price_path)
    # price_path = layers.Dropout(0.5)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 128), activation=silu)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 8), activation=silu)(price_path)
    price_path = BatchNormalization()(price_path)

    # Initial Common LSTM Layer
    volume_path = layers.LSTM(int(GAN_SIZE * 32), kernel_initializer=HeNormal(), return_sequences=True,
                              activation=layers.PReLU(), kernel_regularizer=l2(0.01))(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 32), activation=layers.PReLU(), kernel_initializer=HeNormal())(
        volume_path)
    # volume_path = layers.Dropout(0.5)(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 32), activation=layers.PReLU())(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 8), activation=layers.PReLU())(volume_path)
    volume_path = BatchNormalization()(volume_path)

    # Final output layers with different activation functions
    final_price = layers.Dense(int(GAN_SIZE * 2), activation=silu, name='Final_Price')(price_path)
    final_volume = layers.Dense(int(GAN_SIZE * 2), activation=layers.PReLU(), name='Final_Volume')(volume_path)

    # Concatenate the outputs of both branches
    final_output = layers.MultiHeadAttention(num_heads=3, key_dim=3)(final_price, final_volume, join_path)
    final_output = layers.Dense(2, activation=silu, name='Final_Output')(final_output)

    model = models.Model(inputs=input_layer, outputs=final_output, name="Generator")
    return model


def build_discriminator(latent_dim: int = LATENT_DIM) -> Model:
    """
    Builds and returns the discriminator model with separate pathways for price and volume.

    Args:
        latent_dim: Dimension of the latent space.

    Returns:
        A TensorFlow Keras model representing the discriminator with separate pathways.
    """
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, latent_dim))

    price_path = SplitLayer(0, 1)(input_layer)
    volume_path = SplitLayer(1, 2)(input_layer)

    # Price path
    price_path = layers.LSTM(int(GAN_SIZE * 32), kernel_initializer=HeNormal(), return_sequences=True, activation=silu,
                             kernel_regularizer=l2(0.01))(price_path)
    # price_path = layers.MultiHeadAttention(num_heads=2, key_dim=1)(price_path, price_path)
    # price_path = layers.LSTM(int(size * 128))(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 4), kernel_initializer=HeNormal(), activation=silu)(price_path)
    price_path = layers.Dropout(0.5)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 16), activation=silu)(price_path)
    price_path = layers.Reshape((-1, int(GAN_SIZE * 16)))(price_path)
    price_path = layers.LSTM(int(GAN_SIZE * 16), return_sequences=True, activation=silu)(price_path)
    # price_path = layers.LSTM(int(size * 32), return_sequences=False)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 16), activation=silu)(price_path)
    price_path = BatchNormalization()(price_path)

    # Volume path
    volume_path = layers.LSTM(int(GAN_SIZE * 32), kernel_initializer=HeNormal(), return_sequences=True,
                              activation=layers.PReLU(), kernel_regularizer=l2(0.01))(volume_path)
    # volume_path = layers.MultiHeadAttention(num_heads=2, key_dim=1)(volume_path, volume_path)
    # volume_path = layers.LSTM(int(size * 128))(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 4), kernel_initializer=HeNormal(), activation=layers.PReLU())(volume_path)
    volume_path = layers.Dropout(0.5)(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 16), activation=layers.PReLU())(volume_path)
    volume_path = layers.Reshape((-1, int(GAN_SIZE * 16)))(volume_path)
    volume_path = layers.LSTM(int(GAN_SIZE * 16), return_sequences=True, activation=layers.PReLU())(volume_path)
    # volume_path = layers.LSTM(int(size * 32), return_sequences=False)(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 16), activation=layers.PReLU())(volume_path)
    volume_path = BatchNormalization()(volume_path)

    # MultiHeadAttention
    combined_path = layers.MultiHeadAttention(num_heads=2, key_dim=int(GAN_SIZE * 16))(price_path, volume_path)

    # Asegurarse que sigue habiendo una dimensión temporal
    combined_path = layers.TimeDistributed(layers.Dense(int(GAN_SIZE * 16), activation=silu))(combined_path)
    combined_path = layers.GlobalAveragePooling1D()(combined_path)

    final_output = layers.Dense(1, activation='sigmoid')(combined_path)

    model = models.Model(inputs=input_layer, outputs=final_output, name="Discriminator")
    return model


# def build_gan(latent_dim: int = LATENT_DIM,
#               dis_lr: float = DISCRIMINATOR_LEARNING_RATE,
#               gen_lr: float = GENERATOR_LEARNING_RATE,
#               loss_func: callable = least_squares_loss,
#               metrics: callable = dtw_distance,
#               ) -> Tuple[Model, Model, Model, Adam, Adam]:
#     """
#     Builds and compiles both the generator and discriminator to form the GAN.
#     :param metrics: function to measure the distance between the distribution of generated data and real data.
#     :param loss_func: function to measure the performance of a classification model.
#     :param gen_lr: generator learning rate (default: constant GENERATOR_LEARNING_RATE)
#     :param dis_lr: discriminator learning rate (default: constant DISCRIMINATOR_LEARNING_RATE)
#     :param latent_dim: Dimension of the latent space.
#     :return: A tuple containing the generator, discriminator, and the GAN model.
#     :example: generator, discriminator, gan = build_gan(LATENT_DIM)
#     """
#     # Create and compile the generator and discriminator
#     generator = build_generator(latent_dim)
#     discriminator = build_discriminator()
#
#     # Exponential decay of the learning rate
#     lr_schedule = ExponentialDecay(
#         initial_learning_rate=gen_lr,
#         decay_steps=1000,
#         decay_rate=0.96,
#         staircase=True)
#
#     # Optimizers with a custom learning rate
#     # gen_optimizer = Adam(learning_rate=lr_schedule, beta_1=BETA_1, beta_2=BETA_2)
#     gen_optimizer = Adam(learning_rate=gen_lr, beta_1=BETA_1, beta_2=BETA_2)
#     disc_optimizer = Adam(learning_rate=dis_lr, beta_1=BETA_1, beta_2=BETA_2)
#
#
#     # Compile the discriminator
#     discriminator.compile(loss=loss_func, optimizer=disc_optimizer, metrics=[BinaryCrossentropy()])
#     # discriminator.compile(loss='binary_crossentropy', optimizer=disc_optimizer, metrics=['accuracy'])
#
#
#     # Ensure the discriminator's weights are not updated during the GAN training
#     discriminator.trainable = False
#
#     # Create and compile the GAN
#     gan_input = layers.Input(shape=(None, latent_dim))
#     fake_data = generator(gan_input)
#
#     # If the generator produces a list of outputs (price and volume), concatenate them
#     combined_output = fake_data
#
#     gan_output = discriminator(combined_output)
#     gan = models.Model(gan_input, gan_output, name="GAN")
#     gan.compile(loss=loss_func, optimizer=gen_optimizer)
#     discriminator.trainable = True
#
#     return generator, discriminator, gan, gen_optimizer, disc_optimizer


def build_gan(latent_dim: int = LATENT_DIM,
              dis_lr: float = DISCRIMINATOR_LEARNING_RATE,
              gen_lr: float = GENERATOR_LEARNING_RATE,
              ) -> Tuple[Model, Model, Model, Adam, Adam]:
    """
    Builds both the generator and discriminator, and returns these along with their optimizers.
    Does not compile the GAN since it's not directly used for training with .fit().

    Args:
        latent_dim: Dimension of the latent space.
        gen_lr: Generator learning rate.
        dis_lr: Discriminator learning rate.

    Returns:
        A tuple containing the generator, discriminator, and the GAN model, along with the optimizers for both generator and discriminator.
    """
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    gen_optimizer = Adam(learning_rate=gen_lr, beta_1=BETA_1, beta_2=BETA_2, clipvalue=1.0)
    disc_optimizer = Adam(learning_rate=dis_lr, beta_1=BETA_1, beta_2=BETA_2, clipvalue=1.0)

    # Ensure the discriminator's weights are not updated during the GAN model usage
    discriminator.trainable = False

    # Setup GAN model for structural purposes, without compiling
    gan_input = layers.Input(shape=(None, latent_dim))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = models.Model(gan_input, gan_output, name="GAN")

    # Restore trainability if needed elsewhere
    discriminator.trainable = True

    return generator, discriminator, gan, gen_optimizer, disc_optimizer


def generate_noise(batch_size: int, latent_dim: int = LATENT_DIM) -> np.ndarray:
    """
    Generates a Gaussian noise vector with the specified batch size and latent dimension.

    Args:
        batch_size (int): Number of noise samples to generate.
        latent_dim (int): Dimension of the latent space.

    Returns:
        np.ndarray: Random noise vectors.
    """
    return np.random.normal(0, 1, size=(batch_size, SEQUENCE_LENGTH, latent_dim)).astype('float32')


def generate_data(generator: Model, num_samples: int, latent_dim: int = LATENT_DIM) -> np.ndarray:
    """
    Generates new data using the trained generator model of a GAN.

    Args:
        generator (keras.Model): The trained generator model of the GAN.
        latent_dim (int): Dimension of the latent space used during GAN training.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated data samples.
    """
    # Generate random noise
    noise = generate_noise(num_samples, latent_dim)

    # Generate data from noise
    generated_data = generator.predict(noise)

    return generated_data  # Shape (num_samples, SEQUENCE_LENGTH, latent_dim)
