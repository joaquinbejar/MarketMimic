from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from marketmimic.constants import DEFAULT_COLUMNS, LATENT_DIM


class DataPreparationException(Exception):

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InverseDateException(Exception):

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, MinMaxScaler]]:
    """
    Prepares and scales data from a DataFrame for model training, scaling each column separately.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be scaled.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the scaled DataFrame
        and a dictionary of scalers used for each column transformations.
    """
    if df is None or df.empty:
        raise DataPreparationException("Input DataFrame is empty or None")

    if 'Price' not in df.columns or 'Volume' not in df.columns:
        raise DataPreparationException("Input DataFrame must contain 'Price' and 'Volume' columns")

    try:
        # Separate data
        price_data = df[['Price']].values
        volume_data = df[['Volume']].values

        # Normalize the data using MinMaxScaler for each column separately
        scalers = {
            'Price': MinMaxScaler(),
            'Volume': MinMaxScaler()
        }
        price_scaled = scalers['Price'].fit_transform(price_data)
        volume_scaled = scalers['Volume'].fit_transform(volume_data)

        # Combine scaled data back into a DataFrame
        df_scaled = pd.DataFrame(data=np.hstack([price_scaled, volume_scaled]), columns=['Price', 'Volume'])

        # Convert to type float32, which is more suitable for TensorFlow training
        data_scaled = df_scaled.astype('float32').values

        return data_scaled, scalers
    except Exception as e:
        raise DataPreparationException(f"Error preparing data: {str(e)}")


def inverse_scale_data(scaled_data: np.ndarray,
                       scalers: Dict[str, MinMaxScaler],
                       index: np.ndarray = None) -> pd.DataFrame:
    """
    Inversely transforms scaled data back to its original scale using provided MinMaxScalers
    and returns a DataFrame with specified column names and index.

    This function takes scaled numerical data and applies inverse transformations to return
    the data to its original scale. The function is specifically designed to work with financial
    data, where 'Price' and 'Volume' are common columns.

    Args:
        scaled_data (np.ndarray): The scaled data to be transformed back to the original scale.
                                  The array should have exactly two columns.
        scalers (Dict[str, MinMaxScaler]): A dictionary containing the scalers used to originally
                                           transform the data, keyed by column name.
        index (np.ndarray, optional): Array containing the index values for the DataFrame. If provided,
                                      this will be used as the DataFrame index.

    Returns:
        pd.DataFrame: A DataFrame transformed back to its original scale with specified columns and index.
                      If there's an error during the process, an empty DataFrame is returned.

    Raises:
        ValueError: If `scaled_data` does not have exactly two columns.
        Exception: Outputs an error message if any other error occurs during data transformation.
    """
    try:
        if len(scaled_data.shape) == 3:  # inverse sliding windows if necessary
            scaled_data = invert_sliding_windows(scaled_data)

        if scaled_data.shape[1] != LATENT_DIM:
            raise InverseDateException("Scaled data array must have exactly two columns.")

        # Revert scaling using the appropriate scaler
        price_original = scalers['Price'].inverse_transform(scaled_data[:, 0].reshape(-1, 1))
        volume_original = scalers['Volume'].inverse_transform(scaled_data[:, 1].reshape(-1, 1))

        # Create a DataFrame using the transformed data, with the specified column names and index
        df = pd.DataFrame(data=np.hstack([price_original, volume_original]), columns=DEFAULT_COLUMNS)

        df['Price'] = df['Price'].astype(float)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].round(2)

        # np.round avoid small discrepancies in volume
        if 'Volume' in df.columns and pd.api.types.is_numeric_dtype(df['Volume']):
            df['Volume'] = np.round(df['Volume']).astype(int)

        if index is not None:
            df.index = index
        df.index.name = 'epoch'

        return df
    except Exception as e:
        raise InverseDateException(f"Error reversing scale of data: {str(e)}")


def create_sliding_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Creates overlapping sliding windows from a given time series data.

    This function takes a time series array and generates sliding windows from it,
    each window having a specified size. This is commonly used in machine learning tasks
    where the sequence or temporal structure of the data is important.

    Args:
        data (np.ndarray): The time series data as a NumPy array, where each row is a time step.
        window_size (int): The number of time steps in each window, also known as the size of the window.

    Returns:
        np.ndarray: An array of sliding windows, each window containing 'window_size' consecutive time steps from the data.
    """
    num_samples = data.shape[0] - window_size + 1
    windows = np.array([data[i:i + window_size] for i in range(num_samples)])
    return windows


def invert_sliding_windows(windows: np.ndarray) -> np.ndarray:
    """
    Inverts sliding windows to reconstruct the original time series data.

    This function is used to reconstruct the original time series data from its sliding windows.
    It is assumed that each window overlaps the previous window by all but one time step,
    and this function takes the last element from each window to reconstruct the series.

    Args:
        windows (np.ndarray): An array of sliding windows, where each window is a sequence of time steps.

    Returns:
        np.ndarray: The reconstructed time series, which concatenates the last elements of each window
                    and the initial elements from the first window.
    """
    # Take the last element from each window to form the main part of the series
    reconstructed_series = windows[:, -1, :]

    # Additionally, the initial elements from the first window are needed to complete the series
    first_elements = windows[0, :-1, :]

    # Concatenate the initial elements with the rest of the reconstructed series
    complete_series = np.vstack((first_elements, reconstructed_series))

    return complete_series
