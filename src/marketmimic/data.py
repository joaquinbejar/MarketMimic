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

# Example usage
# Assuming 'df' is your DataFrame loaded with data
# data_scaled, scaler = prepare_data(df)
