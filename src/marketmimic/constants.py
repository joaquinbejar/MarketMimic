LATENT_DIM = 2  # Dimension space of latent variables
DISCRIMINATOR_LEARNING_RATE = 0.0002
GENERATOR_LEARNING_RATE = 0.0001
BETA_1 = 0.5  # Lower beta_1 for more "nervous" updates
BETA_2 = 0.999  # Default value for beta_2 a lower beta_2 improves stability
SMOOTH_FACTOR = 0.9  # Factor to reduce learning rate when NaNs are detected
SEQUENCE_LENGTH = 16
DEFAULT_COLUMNS = ['Price', 'Volume']
SHOW_LOSS_EVERY = 10
GAN_ARCH_VERSION = 0.1
GAN_SIZE = 1
EPOCHS = 3
BATCH_SIZE = 1024
SMOOTH_REAL_LABEL = 0.9
SMOOTH_FAKE_LABEL = 0.1
SGD_MOMENTUM = 0.9