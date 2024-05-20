import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from marketmimic.constants import SEQUENCE_LENGTH, SHOW_LOSS_EVERY, SMOOTH_FACTOR
from marketmimic.data import create_sliding_windows

def reinitialize_weights(model):
    for layer in model.layers:
        # Re-inicializar los pesos para capas densas
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'kernel'):
            layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape))

        # Re-inicializar los sesgos si la capa los usa
        if hasattr(layer, 'bias_initializer') and hasattr(layer, 'bias') and layer.use_bias:
            layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape))

        # Específico para capas LSTM o RNN
        if isinstance(layer, tf.keras.layers.LSTM) or isinstance(layer, tf.keras.layers.RNN):
            # Peso de las entradas
            if hasattr(layer, 'cell'):
                if hasattr(layer.cell, 'kernel_initializer') and hasattr(layer.cell, 'kernel'):
                    layer.cell.kernel.assign(layer.cell.kernel_initializer(shape=layer.cell.kernel.shape))

                # Peso recurrente
                if hasattr(layer.cell, 'recurrent_initializer') and hasattr(layer.cell, 'recurrent_kernel'):
                    layer.cell.recurrent_kernel.assign(layer.cell.recurrent_initializer(shape=layer.cell.recurrent_kernel.shape))

                # Sesgo
                if hasattr(layer.cell, 'bias_initializer') and hasattr(layer.cell, 'bias'):
                    layer.cell.bias.assign(layer.cell.bias_initializer(shape=layer.cell.bias.shape))


def train_gan(generator: Model, discriminator: Model,
              gen_optimizer: tf.keras.optimizers.Optimizer,
              disc_optimizer: tf.keras.optimizers.Optimizer,
              dataset: np.ndarray, epochs: int, batch_size: int) -> None:
    sequence_data = create_sliding_windows(dataset, SEQUENCE_LENGTH)
    best_loss = float('inf')
    weights_found = False
    reinitialize_weights(generator)
    reinitialize_weights(discriminator)
    best_weights_gen = generator.get_weights()
    best_weights_disc = discriminator.get_weights()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        for idx in range(0, sequence_data.shape[0], batch_size):
            batch_data = sequence_data[idx:idx + batch_size]
            current_batch_size = batch_data.shape[0]
            noise = tf.random.normal((current_batch_size, SEQUENCE_LENGTH, generator.input_shape[-1]))

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_data = generator(noise, training=True)
                real_output = discriminator(batch_data, training=True)
                fake_output = discriminator(fake_data, training=True)

                real_labels = tf.ones_like(real_output)
                fake_labels = tf.zeros_like(fake_output)

                disc_loss = tf.reduce_mean(tf.losses.binary_crossentropy(real_labels, real_output) +
                                           tf.losses.binary_crossentropy(fake_labels, fake_output))
                gen_loss = tf.reduce_mean(tf.losses.binary_crossentropy(real_labels, fake_output))

            if tf.math.is_nan(disc_loss) or tf.math.is_nan(gen_loss):
                if weights_found:
                    print("NaN detected, reverting to best model weights!")
                    generator.set_weights(best_weights_gen)
                    discriminator.set_weights(best_weights_disc)
                else:
                    print("NaN detected, reinitializing weights!")
                    reinitialize_weights(generator)
                    reinitialize_weights(discriminator)


                # Optionally reduce learning rate here
                gen_current_lr = gen_optimizer.learning_rate.numpy()
                new_lr = gen_current_lr * SMOOTH_FACTOR
                gen_optimizer.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr} in Generator")

                disc_current_lr = disc_optimizer.learning_rate.numpy()
                new_lr = disc_current_lr * SMOOTH_FACTOR
                disc_optimizer.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr} in Discriminator")

                break  # Break the current epoch's loop and restart training

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if idx % SHOW_LOSS_EVERY == 0:
                print(f"Epoch {epoch + 1}/{epochs} Batch {idx // batch_size + 1}/{(sequence_data.shape[0] + batch_size - 1) // batch_size}: Disc Loss = {disc_loss.numpy()}, Gen Loss = {gen_loss.numpy()}")

            # Update best loss and save weights
            total_loss = disc_loss + gen_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_weights_gen = generator.get_weights()
                best_weights_disc = discriminator.get_weights()
                weights_found = True



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
