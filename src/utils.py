import os
import pandas as pd
import logging

log = logging.getLogger(__name__)

from datetime import datetime

def save_to_parquet(processed_data, path):
    """
    Save processed data to parquet.
    If parquet already exists and is valid, read it, concatenate with new data and save it back.
    If processed_data is empty or the existing parquet file is invalid, appropriate actions are taken.
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

            # Drop rows in processed_data_df that have IDs already in existing_data
            if "unique_id" in processed_data_df.columns and "unique_id" in existing_data.columns:
                processed_data_df = processed_data_df[~processed_data_df["unique_id"].isin(existing_data["unique_id"])]
        except Exception as e:
            print(f"Error reading Parquet file: {e}")
            existing_data = None

        # If existing_data is successfully loaded, concatenate and save
        if existing_data is not None and not processed_data_df.empty:
            combined_data = pd.concat([existing_data, processed_data_df], ignore_index=True)
            combined_data.to_parquet(path, index=False)
            print(f"Data appended to {path}")
        else:
            # Existing Parquet was invalid or processed_data_df was empty after removing duplicates, so overwrite with new data
            processed_data_df.to_parquet(path, index=False)
            print(f"Existing file was invalid or no new data. New data saved to {path}")
    else:
        # If the Parquet file doesn't exist, save the new data
        processed_data_df.to_parquet(path, index=False)
        print(f"New data saved to {path}")


def create_versioned_output_directory(output_path):
    """
    Creates a versioned output directory for the current run
    :return:
    """
    version = 1
    while True:
        today = datetime.today().strftime('%Y-%m-%d')
        versioned_dir = os.path.join(output_path, f"{today}/v{version}")
        if not os.path.exists(versioned_dir):
            os.makedirs(versioned_dir)
            log.info(f"Created versioned output directory: {versioned_dir}")
            return versioned_dir
        version += 1