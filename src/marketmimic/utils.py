import random
import zipfile
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_data(zip_file: str, txt_file: str) -> pd.DataFrame:
    '''
    Load data from a zip file and a txt file inside the zip file
    The file should contain market data in Tick format: Date, Time, Price, Volume
    :param zip_file: Path to the zip file
    :param txt_file: Path to the txt file inside the zip file
    :return: A pandas DataFrame with the data
    '''
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            with z.open(txt_file) as f:
                return pd.read_csv(f)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def join_date_time(df: pd.DataFrame,
                   date_col: str,
                   time_col: str,
                   datetime_col: str = 'Datetime',
                   index_column: str = 'epoch') -> pd.DataFrame:
    '''
    Join date and time columns into a single datetime column and set it as the index in unix epoch format
    :param df: DataFrame with the data from the txt file
    :param date_col: Date column name
    :param time_col: Time column name
    :param datetime_col: New column name with the datetime (default: 'Datetime', optional and internal use)
    :param index_column: New column name with the index in unix epoch format (default: 'epoch', optional)
    :return: The DataFrame with the new columns

    example: df = join_date_time(df, 'Date', 'Time')
    '''
    try:
        df[datetime_col] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%d/%m/%Y %H:%M:%S.%f')
    except ValueError:
        # if there is an error, try with a different format or handle mixed formats
        df[datetime_col] = pd.to_datetime(df[date_col] + ' ' + df[time_col], dayfirst=True, errors='coerce')

    # Remove date and time columns
    df[index_column] = (df[datetime_col].astype('int64') // 10 ** 6)
    df.set_index(index_column, inplace=True)
    df.drop([date_col, time_col, datetime_col], axis=1, inplace=True)

    return df


class RandomDataGenerator:
    """
    A class to generate random data based on the differences in a given DataFrame.

    Attributes:
        df (pd.DataFrame): The original DataFrame from which differences are calculated.
        dtype_dict (Dict[str, type]): Dictionary specifying the data types for DataFrame columns.
        df_diff (pd.DataFrame): DataFrame containing the differences between consecutive rows of `df`.
        epoch_diff (List[int]): List of differences in the 'epoch' column.
        price_diff (List[float]): List of differences in the 'Price' column.
        volume (List[int]): List of volumes directly from `df`.
    """

    def __init__(self, df: pd.DataFrame):
        self.dtype_dict = {'epoch': int, 'Price': float, 'Volume': int}
        self.df = df
        self.df_diff = df.reset_index().diff().dropna()
        self.epoch_diff = self.df_diff.epoch.to_list()
        self.price_diff = self.df_diff.Price.to_list()
        self.volume = self.df.Volume.to_list()

    def GenerateItem(self, current_value: Dict[str, float]) -> Dict[str, float]:
        """
        Generates a new dictionary item based on random choices from the differences data.

        Args:
            current_value (Dict[str, float]): The last value from which the new item will start.

        Returns:
            Dict[str, float]: A new data item with 'epoch', 'Price', and 'Volume' updated.
        """
        try:
            item = {'epoch': int(current_value['epoch'] + random.choice(self.epoch_diff)),
                    'Price': round(current_value['Price'] + random.choice(self.price_diff), 2),
                    'Volume': int(random.choice(self.volume))}
            if item['Price'] > 0 and item['Volume'] > 0:
                return item
            else:
                return self.GenerateItem(current_value)
        except Exception as e:
            print(f"Error during item generation: {e}")
            return {}

    def GenerateDF(self, current_value: Dict[str, float], size: int) -> pd.DataFrame:
        """
        Generates a DataFrame of new data items.

        Args:
            current_value (Dict[str, float]): The initial value to start data generation.
            size (int): The number of new data items to generate.

        Returns:
            pd.DataFrame: A DataFrame of generated data items.
        """
        results = []
        last_value = current_value
        for i in range(size):
            item_to_insert = self.GenerateItem(last_value)
            results.append(item_to_insert)
            last_value = item_to_insert
        return pd.DataFrame(results)

    def Plot(self, size: int = None):
        """
        Plots the original and generated data, marking the transition point with a vertical line.

        Args:
            size (int): Number of data points to generate and plot from the end of `df_orig`.
        """
        if not size:
            df_reset = self.df.reset_index()
            last_value = df_reset.iloc[-1].to_dict()
            df_gen = self.GenerateDF(last_value, len(df_reset))
        else:
            df_reset = self.df.iloc[-size:].reset_index()
            last_value = df_reset.iloc[-1].to_dict()
            df_gen = self.GenerateDF(last_value, size)

        plt.figure(figsize=(24, 13))
        plt.plot(df_reset['epoch'], df_reset['Price'], marker='o', linestyle='-', color='b', label='Original Data')
        plt.plot(df_gen['epoch'], df_gen['Price'], marker='o', linestyle='-', color='r', label='Generated Data')
        start_epoch = df_gen['epoch'].iloc[0]
        plt.axvline(x=start_epoch, color='green', label='Start of Generated Data', linestyle='--')
        plt.title('Price Evolution Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


def convert_to_ohlcv(df_ticks: pd.DataFrame, resample: str = 'S') -> pd.DataFrame:
    """
    Converts a DataFrame with tick data (Price, Volume) indexed by epoch timestamps (milliseconds)
    into an OHLCV DataFrame aggregated by second, where missing 'open', 'high', and 'low' values
    are filled using the 'close' value of the same timestamp.

    Args:
        df_ticks (pd.DataFrame): DataFrame containing tick data with 'Price' and 'Volume', indexed by 'epoch'.
        resample (str): Resampling frequency for the data aggregation (default: 'S' for seconds) other values
        can be 'T' for minutes, '5T' for 5 minutes, 'H' for hours, 'D' for days, etc.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data aggregated by second, with missing OHLC values filled from 'close'.
    """
    # Convert the index to a datetime format
    df_ticks.index = pd.to_datetime(df_ticks.index, unit='ms')

    # Resample data by second and apply OHLCV aggregation
    ohlcv = df_ticks['Price'].resample(resample).ohlc()
    ohlcv['Volume'] = df_ticks['Volume'].resample(resample).sum()

    # Forward fill the 'close' values to handle NaNs
    ohlcv['close'].ffill(inplace=True)

    # Fill 'open', 'high', 'low' NaNs with the corresponding 'close' value
    # Creating a dictionary to fill NaN values based on 'close'
    fill_values = {'open': ohlcv['close'], 'high': ohlcv['close'], 'low': ohlcv['close']}
    ohlcv.fillna(value=fill_values, inplace=True)

    return ohlcv
