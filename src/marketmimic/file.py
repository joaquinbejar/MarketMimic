from datetime import datetime

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def load_model_from_file(path: str, loss_funcion: callable, metric=callable) -> Model:
    """
    Loads a TensorFlow/Keras model from a specified path.

    Args:
        path (str): The path from where to load the model.

    Returns:
        Model: The loaded TensorFlow/Keras model.
    """
    model = load_model(path, custom_objects={loss_funcion.__name__: loss_funcion, metric.__name__: metric})
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
    filename = f"{base_name}_{timestamp}{extension}"
    return filename
