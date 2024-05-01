from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


# TODO: create tests for this function
def inverse_scale_data(scaled_data: np.ndarray, scaler: MinMaxScaler, column_names: list,
                       index: np.ndarray = None) -> pd.DataFrame:
    """
    Inversely transforms scaled data back to its original scale using the provided MinMaxScaler and returns a DataFrame
    with specified column names and index.

    Args:
        scaled_data (np.ndarray): The scaled data to be transformed back to the original scale.
        scaler (MinMaxScaler): The scaler used to originally transform the data.
        column_names (list): List of column names for the DataFrame.
        index (np.ndarray): Array containing the index values for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame transformed back to its original scale with specified columns and index.
    """
    try:
        # Use the inverse_transform method of the scaler to revert the data
        original_data = scaler.inverse_transform(scaled_data)

        # Create a DataFrame using the transformed data, with the specified column names and index
        df = pd.DataFrame(data=original_data, columns=column_names)

        df['Price'] = df['Price'].astype(float)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].round(2)

        if 'Volume' in df.columns and pd.api.types.is_numeric_dtype(df['Volume']):
            df['Volume'] = df['Volume'].astype(int)

        if index is not None:
            df.index = index
        df.index.name = 'epoch'

        return df
    except Exception as e:
        print(f"Error reversing scale of data: {str(e)}")
        return pd.DataFrame()
