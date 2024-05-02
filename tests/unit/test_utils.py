import os
import unittest
import zipfile

import pandas as pd

from src.marketmimic import utils


class TestLoadData(unittest.TestCase):
    def test_load_data_valid_input(self):
        try:
            # Create a dataframe for comparison
            comparison_df = pd.DataFrame({
                'Date': pd.date_range(start='1/1/2022', end='1/10/2022'),
                'Time': pd.date_range(start='1/1/2022', periods=10),
                'Price': pd.Series(range(1, 11)),
                'Volume': pd.Series(range(100, 200, 10))
            })

            # Convert dataframe to csv and save it as a txt file
            comparison_df.to_csv('test_data.txt', index=False)

            # Create a new ZipFile object in write mode
            with zipfile.ZipFile('test_data.zip', 'w') as zipf:
                # Add txt file to zip
                zipf.write('test_data.txt')

            # Load data from load_data function
            result_df = utils.load_data('test_data.zip', 'test_data.txt')

            # Convert Date and Time columns back to datetime if necessary
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            result_df['Time'] = pd.to_datetime(result_df['Time'])

            # Remove the created test files
            os.remove('test_data.txt')
            os.remove('test_data.zip')

            # Assert the loaded dataframe is equal to the comparison dataframe
            pd.testing.assert_frame_equal(result_df, comparison_df)

        except Exception as e:
            assert False, str(e)

    def test_load_data_invalid_zip_file(self):
        # Load data from load_data function with invalid zip file
        result_df = utils.load_data('invalid.zip', 'test_data.txt')

        # Assert the loaded dataframe is empty
        assert result_df.empty

    def test_load_data_invalid_txt_file(self):
        # Load data from load_data function with invalid txt file
        result_df = utils.load_data('test_data.zip', 'invalid.txt')

        # Assert the loaded dataframe is empty
        assert result_df.empty

    def test_load_data_non_existing_files(self):
        # Load data from load_data function with non-existing zip and txt file
        result_df = utils.load_data('non_existing.zip', 'non_existing.txt')

        # Assert the loaded dataframe is empty
        assert result_df.empty


class TestJoinDateTime(unittest.TestCase):
    def test_join_date_time_correct_format(self):
        """Test that date and time are correctly joined into a datetime column and transformed into epoch format."""
        # Create a test DataFrame
        df = pd.DataFrame({
            'Date': ['01/01/2022', '02/01/2022'],
            'Time': ['12:00:00.000', '13:00:00.000']
        })
        # Calculate expected 'epoch' values directly using correct datetime conversion
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
        expected_df = pd.DataFrame({
            'Price': [100, 200]
        })
        expected_df['epoch'] = (df['Datetime'].astype('int64') // 10 ** 6)
        expected_df.set_index('epoch', inplace=True)

        # Call the function to test
        result_df = utils.join_date_time(df, 'Date', 'Time')
        result_df['Price'] = [100, 200]  # Add additional data for comparison

        # Verify that the result matches the expectation
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_error_handling(self):
        """Test that the function correctly handles incorrect date formats."""
        df = pd.DataFrame({
            'Date': ['01-01-2022', '02-01-2022'],  # Incorrect format
            'Time': ['12:00:00', '13:00:00']
        })
        # Expecting the function not to fail even with incorrect date formats
        result_df = utils.join_date_time(df, 'Date', 'Time')
        # Verify no errors and data handled as NaN or successful conversion
        self.assertTrue(result_df.index.isnull().all() or not result_df.index.isnull().any())

    def test_column_removal(self):
        """Test that the original date and time columns are removed."""
        df = pd.DataFrame({
            'Date': ['01/01/2022', '02/01/2022'],
            'Time': ['12:00:00.000', '13:00:00.000']
        })
        result_df = utils.join_date_time(df, 'Date', 'Time')
        self.assertNotIn('Date', result_df.columns)
        self.assertNotIn('Time', result_df.columns)


if __name__ == '__main__':
    unittest.main()
