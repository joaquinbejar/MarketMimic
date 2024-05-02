import warnings
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from marketmimic.constants import SMOOTH_FACTOR, SEQUENCE_LENGTH, SHOW_LOSS_EVERY
from marketmimic.data import create_sliding_windows

tf.config.run_functions_eagerly(True)


@tf.function
def train_step(generator: Model, discriminator: Model, gan: Model, real_data: np.ndarray, noise: np.ndarray,
               real_y: tf.Tensor, fake_y: tf.Tensor) -> Tuple[float, float, float]:
    """
    Performs a single training step for both the generator and discriminator.

    Args:
        generator (Model): The generator component of the GAN.
        discriminator (Model): The discriminator component of the GAN.
        gan (Model): The composite model where the generator's output is fed to the discriminator.
        real_data (np.ndarray): A batch of real data samples.
        noise (np.ndarray): A batch of random noise vectors.
        real_y (tf.Tensor): Labels for real data (typically ones).
        fake_y (tf.Tensor): Labels for fake data (typically zeros).

    Returns:
        Tuple[float, float, float]: The discriminator loss on real data, discriminator loss on fake data, and
        generator loss.
    """
    with warnings.catch_warnings():
        # ignore UserWarning
        warnings.simplefilter('ignore', UserWarning)
        # Train discriminator with real data
        d_loss_real, accuracy_real = discriminator.train_on_batch(real_data, real_y)
        # Generate fake data
        fake_data = generator(noise, training=True)
        # Train discriminator with fake data
        d_loss_fake, accuracy_fake = discriminator.train_on_batch(fake_data, fake_y)
        # Train the generator
        gan_output = gan.train_on_batch(noise, real_y)
        g_loss = gan_output[0]  # Assuming the first element is the loss
        return d_loss_real, d_loss_fake, g_loss


def train_gan(generator: Model, discriminator: Model, gan: Model, dataset: np.ndarray, epochs: int,
              batch_size: int) -> None:
    """
    Trains the Generative Adversarial Network.

    Args:
        generator (Model): The generator model.
        discriminator (Model): The discriminator model.
        gan (Model): The composite GAN model.
        dataset (np.ndarray): The complete dataset for training.
        epochs (int): The number of epochs to train the models.
        batch_size (int): The size of each training batch.

    Returns:
        None: This function does not return any values but will print the training progress.
    """
    sequence_data = create_sliding_windows(dataset, SEQUENCE_LENGTH)

    for epoch in range(epochs):
        idx = np.random.randint(0, sequence_data.shape[0], batch_size)
        real_data = sequence_data[idx].astype('float32')  # Datos reales son ahora secuencias

        noise = np.random.normal(0, 1, size=(batch_size, SEQUENCE_LENGTH, 2)).astype('float32')

        real_y = tf.ones((batch_size, 1), dtype=tf.float32) * (1 - SMOOTH_FACTOR)
        fake_y = tf.zeros((batch_size, 1), dtype=tf.float32) + SMOOTH_FACTOR

        d_loss_real, d_loss_fake, g_loss = train_step(generator, discriminator, gan, real_data, noise, real_y, fake_y)

        if epoch % SHOW_LOSS_EVERY == 0:
            # Compute average discriminator loss directly using NumPy values
            avg_d_loss = (d_loss_real + d_loss_fake) / 2
            print(f"Epoch: {epoch} [D loss: {avg_d_loss:.4f}, G loss: {g_loss:.4f}]")
