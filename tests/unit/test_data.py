import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.marketmimic.utils import load_data, join_date_time
from src.marketmimic.data import prepare_data, DataPreparationException, inverse_scale_data, InverseDateException


class TestPrepareData(unittest.TestCase):
    def test_prepare_data_valid_input(self):
        df = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=['Price', 'Volume'])
        data_scaled, scalers = prepare_data(df)
        self.assertIsInstance(data_scaled, np.ndarray)
        self.assertIsInstance(scalers, dict)
        self.assertEqual(len(scalers), 2)
        self.assertIn('Price', scalers)
        self.assertIn('Volume', scalers)
        self.assertIsInstance(scalers['Price'], MinMaxScaler)
        self.assertIsInstance(scalers['Volume'], MinMaxScaler)
        self.assertEqual(data_scaled.dtype, 'float32')

    def test_prepare_data_none_input(self):
        df = None
        with self.assertRaises(DataPreparationException):
            prepare_data(df)

    def test_prepare_data_empty_input(self):
        df = pd.DataFrame()
        with self.assertRaises(DataPreparationException):
            prepare_data(df)

    def test_prepare_data_no_Price_input(self):
        df = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=['epoch', 'Volume'])
        with self.assertRaises(DataPreparationException):
            prepare_data(df)

    def test_prepare_data_no_Volume_input(self):
        df = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=['Price', 'volume'])
        with self.assertRaises(DataPreparationException):
            prepare_data(df)


class TestInverseScaleData(unittest.TestCase):

    def setUp(self):
        self.scalers = {'Price': MinMaxScaler(), 'Volume': MinMaxScaler()}
        self.price_data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
        self.volume_data = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
        self.scaled_data = np.hstack([self.scalers['Price'].fit_transform(self.price_data),
                                      self.scalers['Volume'].fit_transform(self.volume_data)])
        self.index = np.array(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'])

    def test_inverse_scale_data_success(self):
        result = inverse_scale_data(self.scaled_data, self.scalers, self.index)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['Price', 'Volume'])
        self.assertTrue(all(result.index == self.index))
        self.assertEqual(len(result), len(self.price_data))

    def test_inverse_scale_data_failure_wrong_number_of_columns(self):
        scaled_data = np.array([10, 20, 30, 40, 50])
        self.assertRaises(InverseDateException, inverse_scale_data, scaled_data, self.scalers, self.index)

    def test_inverse_scale_data_failure_general_exception(self):
        self.assertRaises(InverseDateException, inverse_scale_data, self.scaled_data, {}, self.index)


class TestFullData(unittest.TestCase):

    def setUp(self):
        # load data
        zip_file = 'data/AAPL-Tick-Standard.txt.zip'
        txt_file = 'AAPL-Tick-Standard.txt'

        # Load data
        df = load_data(zip_file, txt_file)
        self.df = join_date_time(df, 'Date', 'Time')

    def test_prepare_data(self):
        data_scaled, scalers = prepare_data(self.df)
        self.assertIsInstance(data_scaled, np.ndarray)
        self.assertIsInstance(scalers, dict)
        self.assertEqual(len(scalers), 2)
        self.assertIn('Price', scalers)
        self.assertIn('Volume', scalers)
        self.assertIsInstance(scalers['Price'], MinMaxScaler)
        self.assertIsInstance(scalers['Volume'], MinMaxScaler)
        self.assertEqual(data_scaled.dtype, 'float32')

        original_data = inverse_scale_data(data_scaled, scalers, self.df.index)
        self.assertIsInstance(original_data, pd.DataFrame)
        self.assertListEqual(list(original_data.columns), ['Price', 'Volume'])
        self.assertTrue(all(original_data.index == self.df.index))
        self.assertEqual(len(original_data), len(self.df))
        self.assertTrue(all(original_data.Price.values == self.df.Price.values))
        self.assertTrue(all(original_data.Volume.values == self.df.Volume.values))


if __name__ == '__main__':
    unittest.main()
