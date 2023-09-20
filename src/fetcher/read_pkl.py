import os
import glob


class PickleFileFetcher:
    def __init__(self, base_directory):
        """
        Initialize the PickleFileFetcher class.

        Parameters:
        - base_directory (str): The root directory from where to start the search for .pkl files.

        Attributes:
        - base_directory (str): Root directory to start the search.
        - processed_files (set): A set to store paths of the files that have been identified.
                                 This helps in keeping track of processed files.
        """
        self.base_directory = base_directory
        self.processed_files = set()

    def find_unprocessed_pkl_file_locations(self):
        """
        Finds all unprocessed pickle (.pkl) files within the base_directory.

        Logic:
        1. Use the `glob` module to recursively find all .pkl files within the base directory.
        2. Filter out files that have names starting with 'S'.
        3. Add these files to the processed_files set for record keeping.

        Returns:
        - pkl_files (list): List of paths to the located .pkl files that start with 'S'.
        """
        pkl_files = []

        # Get a list of all .pkl files recursively from the base directory.
        all_files = glob.glob(os.path.join(self.base_directory, '**/*.pkl'), recursive=True)

        for file_path in all_files:
            # Extract the filename from the full file path.
            filename = os.path.basename(file_path)

            # Check if the filename starts with 'S'.
            if filename.startswith('S'):
                pkl_files.append(file_path)
                # Record this file as processed by adding it to the processed_files set.
                self.processed_files.add(file_path)

        return pkl_files
