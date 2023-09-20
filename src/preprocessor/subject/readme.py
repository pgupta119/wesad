import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
from constants import PROCESSED_DATA_DIRECTORY

def save_to_parquet(processed_data, path):
    """
    Save processed data to parquet.
    If the Parquet file already exists, append data rows with unique IDs not already in the existing data.
    If the processed_data is empty, no changes are made.
    If the Parquet file doesn't exist, create a new one.
    """

    # Check if processed_data is empty
    is_empty = False
    if isinstance(processed_data, list) and not processed_data:
        is_empty = True
    elif isinstance(processed_data, pd.DataFrame) and processed_data.empty:
        is_empty = True

    if is_empty:
        print("The provided data is empty. No changes were made.")
        return

    # Convert processed_data to DataFrame if it's a list
    if isinstance(processed_data, list):
        processed_data_df = pd.DataFrame(processed_data)
    else:
        processed_data_df = processed_data

    # If the Parquet file exists, attempt to load it
    if os.path.exists(path):
        try:
            existing_data = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading Parquet file: {e}")
            existing_data = pd.DataFrame()  # Create an empty DataFrame as a fallback

        # Filter out rows from processed_data_df with unique_ids already in existing_data
        if "unique_id" in processed_data_df.columns:
            processed_data_df = processed_data_df[~processed_data_df["unique_id"].isin(existing_data["unique_id"])]

        # If there are rows left in processed_data_df after filtering, append and save
        if not processed_data_df.empty:
            combined_data = pd.concat([existing_data, processed_data_df], ignore_index=True)
            combined_data.to_parquet(path, index=False)
            print(f"Data appended to {path}")
        else:
            print("No new unique IDs to add. No changes were made.")
    else:
        # If the Parquet file doesn't exist, save the new data
        processed_data_df.to_parquet(path, index=False)
        print(f"New data saved to {path}")


class ReadmePreprocessor:
    def __init__(self, data):
        """
        Initialize DataPreprocessor with file paths.
        """
        self.data = data

    def preprocess(self):
        """Preprocess the data: Convert categorical columns to one-hot encoding."""
        categorical_cols = ['gender', 'dominant_hand', 'coffee_today', 'coffee_last_hour',
                            'sport_today', 'smoker', 'smoke_last_hour', 'feel_ill_today']

        # One-hot encoding for categorical columns using pandas.get_dummies
        encoded_df = pd.get_dummies(self.data[categorical_cols], drop_first=True).astype(int)

        # Concatenate original data and encoded data
        self.data = pd.concat([self.data.drop(categorical_cols, axis=1), encoded_df], axis=1)
        all_samples = self.data
        save_to_parquet(all_samples, path=f'{PROCESSED_DATA_DIRECTORY}/readme_data.parquet')

        return self.data
