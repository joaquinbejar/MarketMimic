from typing import Tuple

import numpy as np
# Using Wasserstein loss
from tensorflow import reduce_mean, Tensor
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from marketmimic.constants import LATENT_DIM, DISCRIMINATOR_LEARNING_RATE, GENERATOR_LEARNING_RATE, SEQUENCE_LENGTH, \
    BETA_1, BETA_2


def wasserstein_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the Wasserstein loss for a batch of predicted and true values.

    The Wasserstein loss function is often used in training generative adversarial networks (GANs)
    to measure the distance between the distribution of generated data and real data. This loss
    function helps in stabilizing the training process of GANs by providing a more meaningful
    and smooth gradient signal.

    Args:
        y_true (Tensor): The ground truth values, usually -1 or 1 indicating real or fake samples.
        y_pred (Tensor): The predicted values from the discriminator.

    Returns:
        Tensor: The computed Wasserstein loss as a single scalar tensor.
    """
    return reduce_mean(y_true * y_pred)


def build_generator(latent_dim: int = LATENT_DIM) -> models.Model:
    """
    Builds and returns the generator model with LSTM layers.
    Args:
        latent_dim: Dimension of the latent space (input noise vector).
    Returns:
        A TensorFlow Keras model representing the generator with LSTM architecture.
    """
    model = models.Sequential([
        layers.Input(shape=(None, latent_dim)),  # Input shape is (None, 2) for sequences of (price, volume)
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(128, return_sequences=True),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dense(1024),
        layers.LSTM(512, return_sequences=True),
        layers.Dense(1024),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(latent_dim, activation='relu'),
        layers.Reshape((SEQUENCE_LENGTH, latent_dim))
    ], name="Generator")
    return model


def build_discriminator(latent_dim: int = LATENT_DIM) -> Model:
    """
    Builds and returns the discriminator model.
    :return: A TensorFlow Keras model representing the discriminator.
    """
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, latent_dim)),
        layers.LSTM(1024, return_sequences=True),  # LSTM process sequences, keeping the time dimension
        layers.Dropout(0.3),
        layers.LSTM(32),  # LSTM process sequences
        layers.Dense(32, activation='leaky_relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ], name="Discriminator")
    return model


def build_gan(latent_dim: int = LATENT_DIM,
              dis_lr: float = DISCRIMINATOR_LEARNING_RATE,
              gen_lr: float = GENERATOR_LEARNING_RATE
              ) -> Tuple[Model, Model, Model]:
    """
    Builds and compiles both the generator and discriminator to form the GAN.
    :param epochs:
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
    discriminator.compile(loss=wasserstein_loss, optimizer=disc_optimizer, metrics=['accuracy'])

    # Ensure the discriminator's weights are not updated during the GAN training
    discriminator.trainable = False

    # Create and compile the GAN
    gan_input = layers.Input(shape=(None, latent_dim))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = models.Model(gan_input, gan_output, name="GAN")
    gan.compile(loss=wasserstein_loss, optimizer=gen_optimizer)

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

    return generated_data  # Shape (num_samples, SEQUENCE_LENGTH, latent_dim)
