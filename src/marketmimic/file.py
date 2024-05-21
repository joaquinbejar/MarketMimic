from datetime import datetime
from typing import Tuple, Callable, Optional

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from marketmimic.constants import GAN_ARCH_VERSION


def load_model_from_file(path: str, loss_function: callable, metric: Optional[callable] = None) -> Model:
    """
    Loads a TensorFlow/Keras model from a specified path.

    Args:
        path (str): The path from where to load the model.
        loss_function (callable): The loss function used to compile the model.
        metric (callable): The metric function used to compile the model. Default is None.
    Returns:
        Model: The loaded TensorFlow/Keras model.

    """
    if metric is None:
        model = load_model(path, custom_objects={loss_function.__name__: loss_function})
    else:
        model = load_model(path, custom_objects={loss_function.__name__: loss_function, 'MeanSquaredError': metric})
    print(f"Model loaded from {path}")
    return model


def save_model_to_file(model: Model, path: str) -> None:
    """
    Saves the entire model to a file.

    Args:
        model (Model): The TensorFlow/Keras model to save.
        path (str): The path where to save the model.
    """
    model.save(path)
    print(f"Model saved to {path}")


def generate_filename_with_timestamp(base_name: str, extension: str = '.keras') -> str:
    """
    Generates a filename that includes the current timestamp.

    Args:
        base_name (str): The base name for the file without extension.
        extension (str): The file extension, default is '.h5' for Keras models.

    Returns:
        str: A string that combines the base name, the current timestamp, and the extension.
    """
    # Get the current timestamp and format it as YYYYMMDD-HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Combine the base name, timestamp, and extension
    filename = f"{base_name}_{timestamp}_v{GAN_ARCH_VERSION}{extension}"
    return filename


def save_models(generator: Model, discriminator: Model, gan: Model, path: str = '../models/') -> Tuple[str, str, str]:
    """
    Saves the generator, discriminator, and GAN models to files with timestamps in their names.

    Args:
        generator (Model): The generator model to save.
        discriminator (Model): The discriminator model to save.
        gan (Model): The GAN model to save.
        path (str): The directory path to save the models. Defaults to '../models/'.

    Returns:
        Tuple[str, str, str]: Filenames of the saved generator, discriminator, and GAN models.
    """
    # Save to file
    generator_filename = generate_filename_with_timestamp('generator')
    save_model_to_file(generator, path + generator_filename)

    discriminator_filename = generate_filename_with_timestamp('discriminator')
    save_model_to_file(discriminator, path + discriminator_filename)

    gan_filename = generate_filename_with_timestamp('gan')
    save_model_to_file(gan, path + gan_filename)

    return generator_filename, discriminator_filename, gan_filename


def load_models(
        generator_filename: str,
        discriminator_filename: str,
        gan_filename: str,
        loss_func: Callable,
        metrics_func: Callable,
        path: str = '../models/'
) -> Tuple[Model, Model, Model]:
    """
    Loads the generator, discriminator, and GAN models from files, using custom loss and metrics functions.

    Args:
        generator_filename (str): The filename of the generator model to load.
        discriminator_filename (str): The filename of the discriminator model to load.
        gan_filename (str): The filename of the GAN model to load.
        loss_func (Callable): The loss function used to compile the models.
        metrics_func (Callable): The metrics function used to compile the GAN model.
        path (str): The directory path from where to load the models. Defaults to '../models/'.

    Returns:
        Tuple[Model, Model, Model]: The loaded generator, discriminator, and GAN models.
    """
    # Load from file
    generator = load_model_from_file(path + generator_filename, loss_function=loss_func)
    discriminator = load_model_from_file(path + discriminator_filename, loss_function=loss_func, metric=metrics_func)
    gan = load_model_from_file(path + gan_filename, loss_function=loss_func, metric=metrics_func)
    # discriminator = Model()

    return generator, discriminator, gan
