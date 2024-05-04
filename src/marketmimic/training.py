import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from marketmimic.constants import SEQUENCE_LENGTH, SHOW_LOSS_EVERY
from marketmimic.data import create_sliding_windows


def train_gan(generator: Model, discriminator: Model,
              gen_optimizer: tf.keras.optimizers.Optimizer,
              disc_optimizer: tf.keras.optimizers.Optimizer,
              dataset: np.ndarray, epochs: int, batch_size: int) -> None:
    """
    Trains the GAN using GradientTape for gradient application.
    """
    sequence_data = create_sliding_windows(dataset, SEQUENCE_LENGTH)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        for idx in range(0, sequence_data.shape[0], batch_size):
            batch_data = sequence_data[idx:idx + batch_size]
            current_batch_size = batch_data.shape[0]  # Actual size of the batch, which might be less than batch_size
            noise = tf.random.normal((current_batch_size, SEQUENCE_LENGTH, generator.input_shape[-1]))

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_data = generator(noise, training=True)
                real_output = discriminator(batch_data, training=True)
                fake_output = discriminator(fake_data, training=True)

                # Use current_batch_size to create labels of the correct size
                real_labels = tf.ones_like(real_output)
                fake_labels = tf.zeros_like(fake_output)

                disc_loss = tf.reduce_mean(tf.losses.binary_crossentropy(real_labels, real_output) +
                                           tf.losses.binary_crossentropy(fake_labels, fake_output))
                gen_loss = tf.reduce_mean(tf.losses.binary_crossentropy(real_labels, fake_output))

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if idx % SHOW_LOSS_EVERY == 0:  # Adjust logging frequency according to your preference
                print(
                    f"Batch {idx // batch_size + 1}/{(sequence_data.shape[0] + batch_size - 1) // batch_size}: Disc Loss = {disc_loss.numpy()}, Gen Loss = {gen_loss.numpy()}")


def train_gan_fit(gan: Model, dataset: np.ndarray, epochs: int, batch_size: int) -> None:
    """
    Trains the GAN using the .fit() method which manages the training loop.

    Args:
        gan (Model): The compiled GAN model.
        dataset (np.ndarray): The complete dataset for training.
        epochs (int): The number of epochs to train the models.
        batch_size (int): The size of each training batch.
    """
    sequence_data = create_sliding_windows(dataset, SEQUENCE_LENGTH)

    # Generamos ruido aleatorio como entrada al generador
    noise = np.random.normal(0, 1, (len(sequence_data), SEQUENCE_LENGTH, 2))

    # Las etiquetas para el GAN deben ser consistentes con lo que el discriminador espera para "datos falsos"
    # generalmente esto es '1' para todos los datos generados ya que queremos engañar al discriminador
    fake_y = np.ones((len(sequence_data), 1), dtype=np.float32)

    gan.fit(x=noise, y=fake_y, epochs=epochs, batch_size=batch_size, shuffle=True)
