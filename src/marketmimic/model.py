from typing import Tuple

from tensorflow import reduce_mean
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam

from marketmimic.constants import LATENT_DIM, DISCRIMINATOR_LEARNING_RATE, GENERATOR_LEARNING_RATE


# Using Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return reduce_mean(y_true * y_pred)


def build_generator(latent_dim: int = LATENT_DIM) -> Model:
    """
    Builds and returns the generator model.
    :param latent_dim: Dimension of the latent space (input noise vector).
    :return: A TensorFlow Keras model representing the generator.
    """
    model = models.Sequential([
        # Use Input to specify the input shape
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128),
        layers.Dropout(0.4),
        layers.LeakyReLU(negative_slope=0.2),  # Use negative_slope instead of alpha
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.2),
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
        layers.Input(shape=(2,)),  # Two inputs for Price and Volume
        layers.Dense(64),
        layers.Dropout(0.4),
        layers.LeakyReLU(negative_slope=0.2),  # Use negative_slope instead of alpha
        layers.Dense(32),
        layers.LeakyReLU(negative_slope=0.2),  # Use negative_slope instead of alpha
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_gan(latent_dim: int = LATENT_DIM,
              dis_lr: float = DISCRIMINATOR_LEARNING_RATE,
              gen_lr: float = GENERATOR_LEARNING_RATE
              ) -> Tuple[Model, Model, Model]:
    """
    Builds and compiles both the generator and discriminator to form the GAN.
    :param latent_dim: Dimension of the latent space.
    :return: A tuple containing the generator, discriminator, and the GAN model.
    :example: generator, discriminator, gan = build_gan(LATENT_DIM)
    """
    # Create and compile the generator and discriminator
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Optimizers with a custom learning rate
    gen_optimizer = Adam(learning_rate=gen_lr, beta_1=0.5)
    disc_optimizer = Adam(learning_rate=dis_lr, beta_1=0.5)

    # Compile the discriminator
    discriminator.compile(loss=wasserstein_loss, optimizer=disc_optimizer, metrics=['accuracy'])

    # Ensure the discriminator's weights are not updated during the GAN training
    discriminator.trainable = False

    # Create and compile the GAN
    gan_input = layers.Input(shape=(latent_dim,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss=wasserstein_loss, optimizer=gen_optimizer)

    return generator, discriminator, gan
