import pandas as pd
import os
import logger
from constants import LOG_FILE

# Set up a logging instance to log important information
log = logger.setup_logger(LOG_FILE)


class Trimmer:
    """
    The Trimmer class provides functionalities to trim and clean wrist sensor data.
    It aims to ensure that the wrist data's length aligns with the chest data's length.
    """

    def __init__(self):
        """
        Constructor for the Trimmer class. Currently does nothing special.
        """
        pass

    def trim_wrist_data(self, wrist_data, chest_data):
        """
        Trim the wrist data to align its length with the chest data's length.

        Steps involved:
        1. Determine the number of samples to trim from the wrist data.
        2. Identify rows with a significant number of NaN values.
        3. Delete the rows with high NaN count until the wrist data's length matches the chest data.
        4. Reset the wrist data's index after deletion.

        Parameters:
        - wrist_data: DataFrame containing the wrist sensor data.
        - chest_data: DataFrame containing the chest sensor data.

        Returns:
        - wrist_data: Trimmed DataFrame containing the wrist sensor data.
        """

        # Calculate the number of samples to trim based on the difference in lengths between wrist and chest data
        sample_trim = len(wrist_data) - len(chest_data)

        # For each row in the wrist data, count the number of NaN values
        nan_count_per_row = wrist_data.isnull().sum(axis=1)

        # Create a mask to identify rows with 4 or more NaN values
        mask = (nan_count_per_row >= 4)

        # Based on the mask, get the indices of the rows to be deleted. Limit the number of rows to the calculated sample_trim
        rows_to_delete = wrist_data[mask].index[:sample_trim]

        # Drop the identified rows from the wrist data
        wrist_data = wrist_data.drop(rows_to_delete)

        # Reset the wrist data's index after row deletion
        wrist_data.reset_index(drop=True, inplace=True)

        return wrist_data

