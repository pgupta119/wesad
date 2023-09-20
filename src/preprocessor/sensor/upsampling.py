import pandas as pd


class UpSampler:
    """
    Class to handle the upsampling of sensor data.
    Facilitates the process of resampling sensor data to a target frequency.
    """

    def __init__(self, fs_dict, column_names):
        """
        Constructor to initialize the UpSampler object.

        Parameters:
        - fs_dict: dict, a dictionary containing sampling frequencies for each sensor type.
        - column_names: dict, a dictionary with sensor type as keys and respective column names as values.
        """
        self.fs_dict = fs_dict
        self.column_names = column_names
        self.dfs = {}  # Dictionary to hold the intermediate dataframes for each sensor type

    def _create_dataframe(self, data, key):
        """
        Private method to create a pandas DataFrame from given data.

        Parameters:
        - data: dict, the input data.
        - key: str, the sensor type to be processed.

        Returns:
        - df_temp: DataFrame, the resulting dataframe with time-delta indexed rows.
        """
        df_temp = pd.DataFrame(data[key], columns=self.column_names[key])
        timedelta_index = pd.to_timedelta((1 / self.fs_dict[key]) * df_temp.index, unit='s')
        df_temp.index = timedelta_index
        return df_temp

    def resample_data(self, data, sensor_data):
        """
        Resample the given data based on the defined target frequency.

        Parameters:
        - data: dict, the input label data.
        - sensor_data: dict, the input sensor data.

        Returns:
        - df: DataFrame, the resampled data.
        """
        for key in self.column_names.keys():
            data_source = data if key == 'label' else sensor_data
            df_temp = self._create_dataframe(data_source, key)
            self.dfs[key] = df_temp  # Store the intermediate dataframe for later use

        df = self._join_dataframes()  # Join all individual dataframes into one
        df['label'] = df['label'].fillna(method='bfill')  # Back-fill any NaN values in the 'label' column
        df = self.filter_labels(df)  # Filter out specified labels
        df.reset_index(drop=True, inplace=True)  # Reset the index

        return df

    @staticmethod
    def filter_labels(data):
        """
        Filter out specific labels from the data.

        Parameters:
        - data: DataFrame, the input data with labels.

        Returns:
        - data: DataFrame, the data after removing specified labels.
        """
        labels_to_remove = [0, 4, 5, 6, 7]
        for label in labels_to_remove:
            data = data[data['label'] != label]
        return data

    def _join_dataframes(self):
        """
        Private method to join all individual sensor dataframes into one.

        Returns:
        - base_df: DataFrame, the resulting dataframe after joining.
        """
        # Start with the first key in the list and join the rest
        base_df = self.dfs[list(self.column_names.keys())[0]]
        for key in list(self.column_names.keys())[1:]:
            base_df = base_df.join(self.dfs[key], how='outer')  # Outer join to ensure all rows are included
        return base_df
