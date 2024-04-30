import zipfile

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
