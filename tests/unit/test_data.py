import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.marketmimic.data import prepare_data, DataPreparationException, inverse_scale_data, InverseDateException, \
    create_sliding_windows, invert_sliding_windows
from src.marketmimic.utils import load_data, join_date_time


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


class TestSlidingWindows(unittest.TestCase):
    def setUp(self):
        # This will run before every test
        self.data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    def test_create_sliding_windows(self):
        # Test the creation of sliding windows with a specific window size
        window_size = 3
        expected_output = np.array([
            [[1, 2], [2, 3], [3, 4]],
            [[2, 3], [3, 4], [4, 5]],
            [[3, 4], [4, 5], [5, 6]]
        ])
        result = create_sliding_windows(self.data, window_size)
        np.testing.assert_array_equal(result, expected_output,
                                      "create_sliding_windows should create correct sliding windows")

    def test_invert_sliding_windows(self):
        # Assuming the sliding windows are correct, test if they can be inverted to the original series
        windows = np.array([
            [[1, 2], [2, 3], [3, 4]],
            [[2, 3], [3, 4], [4, 5]],
            [[3, 4], [4, 5], [5, 6]]
        ])
        expected_series = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        result = invert_sliding_windows(windows)
        np.testing.assert_array_equal(result, expected_series,
                                      "invert_sliding_windows should reconstruct the original series correctly")

    def test_window_inversion_consistency(self):
        # Ensure that creating then inverting windows returns to the original series
        window_size = 3
        windows = create_sliding_windows(self.data, window_size)
        result = invert_sliding_windows(windows)
        np.testing.assert_array_equal(result, self.data,
                                      "Window creation and inversion should be consistent and return original data")


class TestFullDataSlidingWindows(unittest.TestCase):

    def setUp(self):
        # load data
        zip_file = 'data/AAPL-Tick-Standard.txt.zip'
        txt_file = 'AAPL-Tick-Standard.txt'

        # Load data
        df = load_data(zip_file, txt_file)
        self.df = join_date_time(df, 'Date', 'Time')

    def test_sliding_and_inverse(self):
        data_scaled = self.df.values
        self.assertIsInstance(data_scaled, np.ndarray)
        secuence_data = create_sliding_windows(data_scaled, 3)
        self.assertIsInstance(secuence_data, np.ndarray)
        self.assertEqual(secuence_data.shape, (len(data_scaled) - 3 + 1, 3, 2))
        inverse_data = invert_sliding_windows(secuence_data)
        self.assertIsInstance(inverse_data, np.ndarray)
        self.assertEqual(inverse_data.shape, data_scaled.shape)
        self.assertTrue(all(inverse_data.flatten() == data_scaled.flatten()))

    def test_scaling_sliding_and_inverse(self):
        data_scaled, scalers = prepare_data(self.df)
        print('\ndata_scaled: ', data_scaled[:5])
        secuence_data = create_sliding_windows(data_scaled, 3)
        print('secuence_data: ', secuence_data[:5])
        self.assertIsInstance(secuence_data, np.ndarray)
        self.assertEqual(secuence_data.shape, (len(data_scaled) - 3 + 1, 3, 2))
        inverse_data = invert_sliding_windows(secuence_data)
        print('inverse_data: ', inverse_data[:5])
        self.assertIsInstance(inverse_data, np.ndarray)
        self.assertEqual(inverse_data.shape, data_scaled.shape)
        original_data = inverse_scale_data(inverse_data, scalers, self.df.index)
        print('original_data: ', original_data[:5])
        self.assertIsInstance(original_data, pd.DataFrame)
        self.assertListEqual(list(original_data.columns), ['Price', 'Volume'])
        self.assertTrue(all(original_data.index == self.df.index))
        self.assertEqual(len(original_data), len(self.df))
        self.assertTrue(all(original_data.Price.values == self.df.Price.values))
        self.assertTrue(all(original_data.Volume.values == self.df.Volume.values))


if __name__ == '__main__':
    unittest.main()
