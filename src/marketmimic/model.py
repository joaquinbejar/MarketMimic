from typing import Tuple

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras import layers, models, Model
from tensorflow.keras.activations import silu
from tensorflow.keras.initializers import HeNormal, RandomUniform, GlorotUniform
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
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, latent_dim), name='InputLayer_Generator')

    price_path = SplitLayer(0, 1, name='SplitLayer_PricePath')(input_layer)
    volume_path = SplitLayer(1, 2, name='SplitLayer_VolumePath')(input_layer)

    join_path = layers.Concatenate(name='Concatenate_PriceVolume')([price_path, volume_path])
    join_path = layers.Dense(int(GAN_SIZE * 32),
                             name='Dense_1_JoinPath',
                             kernel_initializer=HeNormal(),
                             activation=silu,
                             kernel_regularizer=l2(0.01))(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32),
                             name='Dense_2_JoinPath',
                             activation=silu)(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32),
                             name='Dense_3_JoinPath',
                             activation=silu)(join_path)
    join_path = layers.TimeDistributed(layers.Dense(int(GAN_SIZE * 16),
                                                    name='TD_Dense_JoinPath',
                                                    activation=silu),
                                       name='TimeDistributed_JoinPath'
                                       )(join_path)
    join_path = layers.Dense(int(GAN_SIZE * 32),
                             name='Dense_4_JoinPath',
                             activation=silu)(join_path)
    join_path = BatchNormalization(name='BatchNorm_JoinPath')(join_path)

    # Initial Common LSTM Layer
    price_path = layers.LSTM(int(GAN_SIZE * 128),
                             kernel_initializer=HeNormal(),
                             return_sequences=True,
                             activation=silu,
                             kernel_regularizer=l2(0.01),
                             name='LSTM_1_PricePath')(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 128),
                              activation=silu,
                              name='Dense_1_PricePath',
                              kernel_initializer=HeNormal())(price_path)
    price_path = layers.Dropout(0.5, name='Dropout_PricePath')(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 128),
                              name='Dense_2_PricePath',
                              activation=silu)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 8),
                              name='Dense_3_PricePath',
                              activation=silu)(price_path)
    price_path = BatchNormalization(name='BatchNorm_PricePath')(price_path)

    # Initial Common LSTM Layer
    volume_path = layers.LSTM(int(GAN_SIZE * 32),
                              kernel_initializer=HeNormal(),
                              return_sequences=True,
                              activation=layers.PReLU(),
                              kernel_regularizer=l2(0.01),
                              name='LSTM_1_VolumePath')(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 32),
                               activation=layers.PReLU(),
                               name='Dense_1_VolumePath',
                               kernel_initializer=HeNormal())(volume_path)
    volume_path = layers.Dropout(0.5, name='Dropout_VolumePath')(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 32),
                               name='Dense_2_VolumePath',
                               activation=layers.PReLU())(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 8),
                               name='Dense_3_VolumePath',
                               activation=layers.PReLU())(volume_path)
    volume_path = BatchNormalization(name='BatchNorm_VolumePath')(volume_path)

    # Final output layers with different activation functions
    final_price = layers.Dense(int(GAN_SIZE * 2),
                               activation=silu,
                               name='Final_Price')(price_path)
    final_volume = layers.Dense(int(GAN_SIZE * 2),
                                activation=layers.PReLU(),
                                name='Final_Volume')(volume_path)

    # Concatenate the outputs of both branches
    final_output = layers.MultiHeadAttention(
        num_heads=3,
        name='MultiHeadAttention_PriceVolume',
        key_dim=3)(final_price, final_volume, join_path)

    # Asegurarse que sigue habiendo una dimensión temporal
    final_output = layers.TimeDistributed(layers.Dense(int(GAN_SIZE * 16),
                                                       activation=silu,
                                                       name='TD_Dense_FinalOutput'),
                                          name='TimeDistributed_CombinedPath')(final_output)

    final_output = layers.Dense(2,
                                activation='sigmoid',
                                name='FinalOutput')(final_output)

    final_output = layers.ReLU(name='ActivationReLU')(final_output)
    model = models.Model(inputs=input_layer, outputs=final_output, name="Generator")
    return model


def build_generator_simple(latent_dim: int = LATENT_DIM) -> models.Model:
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

    # price_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(price_path)
    price_path = layers.LSTM(int(GAN_SIZE * 128),
                             return_sequences=True,
                             activation=silu,
                             kernel_regularizer=l2(0.01),
                             kernel_initializer=RandomUniform(),
                             name='LSTM_1_PricePath')(price_path)
    price_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(price_path)
    # price_path = layers.Dropout(0.5)(price_path)

    # Volume path
    # volume_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(volume_path)
    volume_path = layers.LSTM(int(GAN_SIZE * 32),
                              kernel_initializer=RandomUniform(),
                              return_sequences=True,
                              activation=layers.PReLU(),
                              kernel_regularizer=l2(0.01),
                              name='LSTM_1_VolumePath')(volume_path)
    volume_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(volume_path)
    # volume_path = layers.Dropout(0.5)(volume_path)

    final_output = layers.Concatenate()([price_path, volume_path])
    final_output = layers.Dense(2,
                                activation='sigmoid',
                                kernel_regularizer=l2(0.01),
                                kernel_initializer=GlorotUniform(),
                                name='Final_Output')(final_output)

    # Aplicar ReLU en la salida para asegurar no negatividad
    final_output = layers.ReLU()(final_output)
    model = models.Model(inputs=input_layer, outputs=final_output, name="Generator")
    return model


def build_generator_gru(latent_dim: int = LATENT_DIM) -> models.Model:
    """
    Builds and returns a generator model for time series data using LSTM and Conv1D layers,
    with intermediate cross-communication between the two branches for price and volume features.

    Args:
        latent_dim: Dimension of the latent space (input noise vector).
        seq_length: Length of the input sequences.
        feature_dim: Number of features in the output data (e.g., OHLC).

    Returns:
        A TensorFlow Keras model representing the generator.
    """
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, latent_dim), name='InputLayer_Generator')

    # Initial LSTM layers for price and volume paths
    price_path = layers.LSTM(64, return_sequences=True, kernel_initializer=initializers.HeNormal(),
                             name='LSTM_PricePath')(input_layer)
    volume_path = layers.LSTM(64, return_sequences=True, kernel_initializer=initializers.HeNormal(),
                              name='LSTM_VolumePath')(input_layer)

    # Concatenate the LSTM outputs
    combined_path = layers.Concatenate(name='Concatenate_PriceVolume')([price_path, volume_path])

    # Conv1D layers to capture local patterns
    combined_path = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                                  kernel_initializer=initializers.HeNormal(), name='Conv1D_1')(combined_path)
    combined_path = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                                  kernel_initializer=initializers.HeNormal(), name='Conv1D_2')(combined_path)
    combined_path = layers.BatchNormalization(name='BatchNorm_CombinedPath')(combined_path)

    # Attention mechanism to focus on relevant parts of the sequence
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='MultiHeadAttention')(combined_path,
                                                                                                     combined_path)

    # More LSTM layers to capture long-term dependencies
    lstm_output = layers.LSTM(64, return_sequences=True, kernel_initializer=initializers.HeNormal(), name='LSTM_Final')(
        attention_output)

    # TimeDistributed layer to apply the same dense transformation at each time step
    final_output = layers.TimeDistributed(
        layers.Dense(2, activation='sigmoid', kernel_initializer=initializers.GlorotUniform()),
        name='TimeDistributed_Output')(lstm_output)

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
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, latent_dim), name='InputLayer_Discriminator')

    price_path = SplitLayer(0, 1, name='SplitLayer_PricePath')(input_layer)
    volume_path = SplitLayer(1, 2, name='SplitLayer_VolumePath')(input_layer)

    # Price path
    price_path = layers.LSTM(int(GAN_SIZE * 32),
                             kernel_initializer=HeNormal(),
                             return_sequences=True,
                             activation=silu,
                             kernel_regularizer=l2(0.01),
                             name='LSTM_1_PricePath')(price_path)

    price_path = layers.Dense(int(GAN_SIZE * 64),
                              kernel_initializer=HeNormal(),
                              activation=silu,
                              name='Dense_1_PricePath')(price_path)
    price_path = layers.Dropout(0.5,
                                name='Dropout_PricePath')(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 32),
                              activation=silu,
                              name='Dense_2_PricePath')(price_path)
    price_path = layers.Reshape((-1, int(GAN_SIZE * 16)),
                                name='Reshape_PricePath'
                                )(price_path)
    price_path = layers.LSTM(int(GAN_SIZE * 20),
                             return_sequences=True,
                             activation=silu,
                             name='LSTM_2_PricePath')(price_path)
    # price_path = layers.LSTM(int(size * 32), return_sequences=False)(price_path)
    price_path = layers.Dense(int(GAN_SIZE * 128),
                              activation=silu,
                              name='Dense_3_PricePath')(price_path)
    price_path = BatchNormalization(name='BatchNorm_PricePath')(price_path)

    # Volume path
    volume_path = layers.LSTM(int(GAN_SIZE * 32),
                              kernel_initializer=HeNormal(),
                              return_sequences=True,
                              activation=layers.PReLU(),
                              kernel_regularizer=l2(0.01),
                              name='LSTM_1_VolumePath')(volume_path)

    volume_path = layers.Dense(int(GAN_SIZE * 64),
                               kernel_initializer=HeNormal(),
                               activation=layers.PReLU(),
                               name='Dense_1_VolumePath')(volume_path)
    volume_path = layers.Dropout(0.5,
                                 name='Dropout_VolumePath')(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 32),
                               activation=layers.PReLU(),
                               name='Dense_2_VolumePath')(volume_path)
    volume_path = layers.Reshape((-1, int(GAN_SIZE * 16)),
                                 name='Reshape_VolumePath')(volume_path)
    volume_path = layers.LSTM(int(GAN_SIZE * 20),
                              return_sequences=True,
                              activation=layers.PReLU(),
                              name='LSTM_2_VolumePath')(volume_path)
    # volume_path = layers.LSTM(int(size * 32), return_sequences=False)(volume_path)
    volume_path = layers.Dense(int(GAN_SIZE * 128),
                               activation=layers.PReLU(),
                               name='Dense_3_VolumePath')(volume_path)
    volume_path = BatchNormalization(name='BatchNorm_VolumePath')(volume_path)

    # MultiHeadAttention
    combined_path = layers.MultiHeadAttention(num_heads=2,
                                              key_dim=int(GAN_SIZE * 32),
                                              name='MultiHeadAttention_PriceVolumePath')(price_path, volume_path)

    # Asegurarse que sigue habiendo una dimensión temporal
    combined_path = layers.TimeDistributed(layers.Dense(int(GAN_SIZE * 32),
                                                        activation=silu,
                                                        name='TD_Dense_CombinedPath'),
                                           name='TimeDistributed_CombinedPath')(combined_path)
    combined_path = layers.GlobalAveragePooling1D(name='GlobalAverage_CombinedPath')(combined_path)

    final_output = layers.Dense(1,
                                activation='sigmoid',
                                name='Dense_CombinedPath')(combined_path)
    final_output = layers.Activation('sigmoid',
                                     name='ActivationSigmoid')(final_output)
    model = models.Model(inputs=input_layer, outputs=final_output, name="Discriminator")
    return model


def build_discriminator_simple(latent_dim: int = LATENT_DIM) -> Model:
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
    price_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='BatchNorm_PricePath')(price_path)
    price_path = layers.LSTM(int(GAN_SIZE * 128),
                             return_sequences=True,
                             activation=silu,
                             kernel_regularizer=l2(0.01),
                             kernel_initializer=GlorotUniform(),
                             name='LSTM_1_PricePath')(price_path)
    price_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(price_path)
    price_path = layers.Dropout(0.5, name='Dropout_PricePath')(price_path)

    # Volume path
    volume_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='BatchNorm_VolumePath')(volume_path)
    volume_path = layers.LSTM(int(GAN_SIZE * 32),
                              kernel_initializer=HeNormal(),
                              return_sequences=True,
                              activation=layers.PReLU(),
                              kernel_regularizer=l2(0.01),
                              name='LSTM_1_VolumePath')(volume_path)
    volume_path = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(volume_path)
    volume_path = layers.Dropout(0.5, name='Dropout_VolumePath')(volume_path)

    final_output = layers.Concatenate(name='Concatenate_PriceVolume')([price_path, volume_path])

    final_output = layers.Dense(1,
                                # activation='sigmoid',
                                kernel_regularizer=l2(0.01),
                                kernel_initializer=GlorotUniform(),
                                name='Final_Output'
                                )(final_output)
    final_output = layers.Activation('sigmoid')(final_output)
    model = models.Model(inputs=input_layer, outputs=final_output, name="Discriminator")
    return model


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
    # discriminator = build_discriminator()
    discriminator = build_discriminator_simple()

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
