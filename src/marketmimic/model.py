from typing import Tuple

import numpy as np
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from marketmimic.constants import LATENT_DIM, DISCRIMINATOR_LEARNING_RATE, GENERATOR_LEARNING_RATE, SEQUENCE_LENGTH, \
    BETA_1, BETA_2
from marketmimic.loss import *
from marketmimic.metric import *


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

    # Initial Common LSTM Layer
    x = layers.LSTM(1024, return_sequences=True)(input_layer)
    x = layers.LSTM(1024, return_sequences=True)(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(0.5)(x)

    # Price path
    price_path = layers.Dense(64)(x)
    price_path = layers.LSTM(64, return_sequences=True)(price_path)

    # Volume path
    volume_path = layers.Dense(64)(x)
    volume_path = layers.LSTM(64, return_sequences=True)(volume_path)

    # Combine information from both branches and allow exchange before the final output
    combined_path = layers.Concatenate(axis=-1)([price_path, volume_path])
    combined_path = layers.Dense(64, activation='relu')(combined_path)

    # Salidas finales separadas
    # Final output layers with different activation functions
    final_price = layers.Dense(1, activation='softplus')(combined_path)  # salida lineal para 'Price'
    final_volume = layers.Dense(1, activation='relu')(combined_path)  # usar relu para 'Volume'

    # Concatenate the outputs of both branches
    # output = layers.Concatenate()([final_price, final_volume])

    model = models.Model(inputs=input_layer, outputs=[final_price, final_volume], name="Generator")
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

    # Initial Common LSTM Layer
    x = layers.LSTM(512, return_sequences=True)(input_layer)
    x = layers.Dropout(0.5)(x)

    # Price path
    price_path = layers.LSTM(256, return_sequences=True)(x)
    price_path = layers.LSTM(128)(price_path)
    price_path = layers.Dense(32, activation='leaky_relu')(price_path)

    # Volume path
    volume_path = layers.LSTM(256, return_sequences=True)(x)
    volume_path = layers.LSTM(128)(volume_path)
    volume_path = layers.Dense(32, activation='leaky_relu')(volume_path)

    # Combine the outputs of both branches
    combined_path = layers.Concatenate()([price_path, volume_path])
    final_output = layers.Dense(1, activation='sigmoid')(combined_path)  # Salida binaria para clasificación

    model = models.Model(inputs=input_layer, outputs=final_output, name="Discriminator")
    return model


def build_gan(latent_dim: int = LATENT_DIM,
              dis_lr: float = DISCRIMINATOR_LEARNING_RATE,
              gen_lr: float = GENERATOR_LEARNING_RATE,
              loss_func: callable = least_squares_loss,
              metrics: callable = dtw_distance,
              ) -> Tuple[Model, Model, Model]:
    """
    Builds and compiles both the generator and discriminator to form the GAN.
    :param metrics: function to measure the distance between the distribution of generated data and real data.
    :param loss_func: function to measure the performance of a classification model.
    :param gen_lr: generator learning rate (default: constant GENERATOR_LEARNING_RATE)
    :param dis_lr: discriminator learning rate (default: constant DISCRIMINATOR_LEARNING_RATE)
    :param latent_dim: Dimension of the latent space.
    :return: A tuple containing the generator, discriminator, and the GAN model.
    :example: generator, discriminator, gan = build_gan(LATENT_DIM)
    """
    # Create and compile the generator and discriminator
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Exponential decay of the learning rate
    lr_schedule = ExponentialDecay(
        initial_learning_rate=gen_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)

    # Optimizers with a custom learning rate
    gen_optimizer = Adam(learning_rate=lr_schedule, beta_1=BETA_1, beta_2=BETA_2)
    disc_optimizer = Adam(learning_rate=dis_lr, beta_1=BETA_1, beta_2=BETA_2)

    # Compile the discriminator
    discriminator.compile(loss=loss_func, optimizer=disc_optimizer, metrics=[metrics])

    # Ensure the discriminator's weights are not updated during the GAN training
    discriminator.trainable = False

    # Create and compile the GAN
    gan_input = layers.Input(shape=(None, latent_dim))
    fake_data = generator(gan_input)

    # If the generator produces a list of outputs (price and volume), concatenate them
    if isinstance(fake_data, list):
        combined_output = layers.Concatenate(axis=-1)(fake_data)
    else:
        combined_output = fake_data

    gan_output = discriminator(combined_output)
    gan = models.Model(gan_input, gan_output, name="GAN")
    gan.compile(loss=loss_func, optimizer=gen_optimizer)

    return generator, discriminator, gan


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
    # If the generator produces a list of outputs, concatenate them before sending to the discriminator
    if isinstance(generated_data, list):
        generated_data = layers.Concatenate(axis=-1)(generated_data)

    return generated_data  # Shape (num_samples, SEQUENCE_LENGTH, latent_dim)
