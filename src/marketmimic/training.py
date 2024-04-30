import numpy as np
from tensorflow.keras.models import Model

from marketmimic.constants import LATENT_DIM


def train_gan(generator: Model,
              discriminator: Model,
              gan: Model,
              dataset: np.ndarray,
              epochs: int = 1000,
              batch_size: int = 32) -> None:
    """
    Train a Generative Adversarial Network (GAN).

    Args:
    generator (Model): The generator component of the GAN.
    discriminator (Model): The discriminator component of the GAN.
    gan (Model): The combined GAN model.
    dataset (np.ndarray): The dataset to train on.
    epochs (int, optional): Number of epochs to train for. Defaults to 1000.
    batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
    None
    """
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of instances
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_data = dataset[idx]

        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_data = generator.predict(noise)

        # Create labels
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))

        # Train the discriminator (real classified as 1 and fake as 0)
        d_loss_real = discriminator.train_on_batch(real_data, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Progress
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")
