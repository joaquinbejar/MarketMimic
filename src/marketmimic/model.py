from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from marketmimic.constants import LATENT_DIM

def build_generator(latent_dim: int) -> Model:
    """
    Builds and returns the generator model.
    :param latent_dim: Dimension of the latent space (input noise vector).
    :return: A TensorFlow Keras model representing the generator.
    """
    model = models.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(2, activation='linear')  # Two outputs for Price and Volume
    ])
    return model

def build_discriminator() -> Model:
    """
    Builds and returns the discriminator model.
    :return: A TensorFlow Keras model representing the discriminator.
    """
    model = models.Sequential([
        layers.Dense(64, input_dim=2),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(latent_dim: int) -> Tuple[Model, Model, Model]:
    """
    Builds and compiles both the generator and discriminator to form the GAN.
    :param latent_dim: Dimension of the latent space.
    :return: A tuple containing the generator, discriminator, and the GAN model.
    :example: generator, discriminator, gan = build_gan(LATENT_DIM)
    """
    # Create and compile the generator and discriminator
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator()

    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ensure the discriminator's weights are not updated during the GAN training
    discriminator.trainable = False

    # Create and compile the GAN
    gan_input = layers.Input(shape=(LATENT_DIM,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return generator, discriminator, gan


