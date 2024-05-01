import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Prepares and scales data from a DataFrame for model training.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be scaled.

    Returns:
        Tuple[np.ndarray, MinMaxScaler]: A tuple containing the scaled data as a NumPy array
        and the scaler used for transformations.

    Example usage
    Assuming 'df' is your DataFrame loaded with data
    data_scaled, scaler = prepare_data(df)
    """
    try:
        # Convert DataFrame to a NumPy array
        data = df.values

        # Normalize the data using MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Convert to type float32, which is more suitable for TensorFlow training
        data_scaled = data_scaled.astype('float32')

        return data_scaled, scaler
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return np.array([]), MinMaxScaler()


def inverse_scale_data(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inversely transforms scaled data back to its original scale using the provided MinMaxScaler.

    Args:
        scaled_data (np.ndarray): The scaled data to be transformed back to the original scale.
        scaler (MinMaxScaler): The scaler used to originally transform the data.

    Returns:
        np.ndarray: Data transformed back to its original scale.
    """
    try:
        # Use the inverse_transform method of the scaler to revert the data
        original_data = scaler.inverse_transform(scaled_data)
        return original_data
    except Exception as e:
        print(f"Error reversing scale of data: {str(e)}")
        return np.array([])
