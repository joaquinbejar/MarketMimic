LATENT_DIM = 2  # Dimension space of latent variables
DISCRIMINATOR_LEARNING_RATE = 0.0001
GENERATOR_LEARNING_RATE = 0.0002
BETA_1 = 0.5  # Lower beta_1 for more "nervous" updates
BETA_2 = 0.9  # Default value for beta_2 a lower beta_2 improves stability
SMOOTH_FACTOR = 0.1  # Factor to smooth the labels for the discriminator
SEQUENCE_LENGTH = 100
DEFAULT_COLUMNS = ['Price', 'Volume']
