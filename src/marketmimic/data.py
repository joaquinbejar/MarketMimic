from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from marketmimic.constants import DEFAULT_COLUMNS


# def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
#     """
#     Prepares and scales data from a DataFrame for model training.
#
#     Args:
#         df (pd.DataFrame): The input DataFrame containing the data to be scaled.
#
#     Returns:
#         Tuple[np.ndarray, MinMaxScaler]: A tuple containing the scaled data as a NumPy array
#         and the scaler used for transformations.
#
#     Example usage
#     Assuming 'df' is your DataFrame loaded with data
#     data_scaled, scaler = prepare_data(df)
#     """
#     if df is None or df.empty:
#         raise TypeError("Input DataFrame is empty or None")
#     try:
#         # Convert DataFrame to a NumPy array
#         data = df.values
#
#         # Normalize the data using MinMaxScaler
#         scaler = MinMaxScaler()
#         data_scaled = scaler.fit_transform(data)
#
#         # Convert to type float32, which is more suitable for TensorFlow training
#         data_scaled = data_scaled.astype('float32')
#
#         return data_scaled, scaler
#     except Exception as e:
#         print(f"Error preparing data: {str(e)}")
#         return np.array([]), MinMaxScaler()

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
        if scaled_data.shape[1] != 2:
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

