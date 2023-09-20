# Import necessary libraries
import re
import pandas as pd
from pathlib import Path
import logger  # Custom logger module
import os

# Constants related to parsing files
from src.parser.constants import PARSE_FILE_SUFFIX, VALUE_EXTRACT_KEYS
# Logging and data directory constants
from constants import LOG_FILE, RAW_DATA_DIRECTORY

# Initialize the logger for this module
log = logger.setup_logger(LOG_FILE)

# Get the name of the current module to use in logs for context
MODULE_FILENAME = os.path.basename(__file__)


class ReadmeParser:
    """
    Class to parse 'README' files from a given directory.

    Attributes:
    - readme_locations: Dictionary mapping subject_directory to its file path.
    """

    def __init__(self):
        """
        Initialize the ReadmeParser by creating a mapping of subject_directory to its file path.
        The subject directories are named in the format 'S<number>'.
        """
        # Use a dictionary comprehension to map each subject directory to its path in the RAW_DATA_DIRECTORY
        self.readme_locations = {
            subject_directory: Path(RAW_DATA_DIRECTORY) / subject_directory
            for subject_directory in os.listdir(Path(RAW_DATA_DIRECTORY))
            if re.match('^S[0-9]{1,2}$', subject_directory)
        }

    def parse_readme(self, subject_id):
        """
        Parse the README file of a specific subject.

        Parameters:
        - subject_id: Identifier for the subject.

        Returns:
        - Dictionary containing parsed key-value pairs from the README.
        - None if README doesn't exist or has an error.
        """
        # Get the path for the README using the subject_id
        readme_path = self.readme_locations.get(subject_id)
        if not readme_path:
            return None

        # Read the README file content
        with open(readme_path / f"{subject_id}{PARSE_FILE_SUFFIX}", 'r') as f:
            readme_lines = f.read().split('\n')

        # Initialize an empty dictionary to store parsed data
        readme_dict = {}

        # Iterate over each line of the README content
        for item in readme_lines:
            # Check against predefined keys to extract specific values
            for key, config in VALUE_EXTRACT_KEYS.items():
                search_key = config['search_key']
                delimiter = config['delimiter']

                # If the current line starts with our search key, we'll extract the value
                if item.startswith(search_key):
                    _, v = item.split(delimiter, 1)
                    readme_dict[key] = v.strip()
                    break

        return readme_dict

    def parse_all_readmes(self):
        """
        Parse all README files in the given directory and combine them into a single DataFrame.

        Returns:
        - DataFrame containing all parsed README data.
        - None if no valid READMEs were found or if there were errors in parsing.
        """
        dframes = []  # List to store individual DataFrames from each parsed README

        # Iterate over each subject's README
        for subject_id, _ in self.readme_locations.items():
            log.info(f'Parsing Readme files for subject {subject_id} in {MODULE_FILENAME}')
            log.debug(f'Parsing for subject {subject_id} in {MODULE_FILENAME}')
            readme_dict = self.parse_readme(subject_id)

            # Convert the parsed dictionary to DataFrame
            if readme_dict:
                df = pd.DataFrame(readme_dict, index=[subject_id])
                df['unique_id'] = subject_id
                dframes.append(df)

        # Combine individual DataFrames into one
        if dframes:
            df = pd.concat(dframes)
            return df
        return None
