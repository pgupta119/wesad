import os
import pickle
import logger
from src.fetcher.read_pkl import PickleFileFetcher
from src.parser.readme_parser import ReadmeParser
from src.preprocessor.subject.readme import ReadmePreprocessor
from src.preprocessor.sensor.wrist.wrist import WristDataProcessor
from src.preprocessor.sensor.chest.chest import ChestDataProcessor
from src.preprocessor.sensor.upsampling import UpSampler
from src.feature_extractor.wrist import WristFeatureExtractor
from src.feature_extractor.chest import ChestFeatureExtractor
from src.preprocessor.sensor.wrist.trim import Trimmer
from src.analyzer.eda import EDA
from src.models.train import WESADLDA
from constants import LOG_FILE, RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY
import warnings

# Ignore all warnings for better clarity
warnings.simplefilter('ignore')

# Initialize a logger instance
log = logger.setup_logger(LOG_FILE)


def process_subject_data(file_location):
    """
    Process the subject data which includes preprocessing the wrist and chest data,
    feature extraction, and model training.
    """
    # Extract subject ID from the filename
    subject_id = os.path.splitext(os.path.basename(file_location))[0]

    # Process wrist data
    wrist_processor = WristDataProcessor(file_location)
    wrist_data = wrist_processor.process_data()

    # Process chest data
    chest_processor = ChestDataProcessor(file_location)
    chest_data = chest_processor.process_data()

    # Define sampling frequencies for wrist and chest data
    fs_dict_wrist = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    fs_dict_chest = {'ACC': 700, 'ECG': 700, 'EMG': 700, 'EDA': 700, 'Temp': 700, 'Resp': 700, 'label': 700}

    # Load wrist and chest combined data
    with open(file_location, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Define column names for wrist data
    column_names_wrist = {
        'ACC': ['ACC_x', 'ACC_y', 'ACC_z'],
        'BVP': ['BVP'],
        'EDA': ['EDA'],
        'TEMP': ['TEMP'],
        'label': ['label']
    }

    # Resample wrist data to target frequency
    resampler_w = UpSampler(fs_dict_wrist, column_names_wrist)
    wrist_result = resampler_w.resample_data(data, wrist_data)

    # Define column names for chest data
    coulmn_names_chest = {
        'ACC': ['ACC_x', 'ACC_y', 'ACC_z'],
        'ECG': ['ECG'],
        'EMG': ['EMG'],
        'EDA': ['EDA'],
        'Temp': ['Temp'],
        'Resp': ['Resp'],
        'label': ['label']
    }

    # Resample chest data to target frequency
    resampler_c = UpSampler(fs_dict_chest, coulmn_names_chest)
    chest_result = resampler_c.resample_data(data, chest_data)

    # Extract features from the wrist data
    trimmer = Trimmer()
    wrist_result = trimmer.trim_wrist_data(wrist_result, chest_result)
    w_processor = WristFeatureExtractor(wrist_result, fs_dict_wrist, 60)
    w_samples = w_processor.process(subject_id)

    # Extract features from the chest data
    c_processor = ChestFeatureExtractor(chest_result, fs_dict_chest, 60)
    c_samples = c_processor.process(subject_id)

    # Exploratory data analysis
    data_directory = f'{PROCESSED_DATA_DIRECTORY}'
    base_output_directory = f'{OUTPUT_DATA_DIRECTORY}plots'
    eda = EDA(data_directory, base_output_directory)
    log.info("Starting EDA process...")
    eda.main()
    log.info("EDA process completed.")

    # Model training and feature importance
    model = WESADLDA()
    model.transform()
    log.debug(f'Processing finished for subject {subject_id}')


def main():
    """
    Main execution function. It orchestrates the entire pipeline from data reading,
    preprocessing, feature extraction, and model training.
    """
    try:
        # Parse and preprocess README files
        readme_parser = ReadmeParser()
        readme_data = readme_parser.parse_all_readmes()
        readme_preprocessor = ReadmePreprocessor(readme_data)
        readme_preprocessor.preprocess()

        # Find and process all unprocessed Pickle files from raw data directory
        pickle_file_fetcher = PickleFileFetcher(RAW_DATA_DIRECTORY)
        file_locations = pickle_file_fetcher.find_unprocessed_pkl_file_locations()

        for file_location in file_locations:
            log.info(f'Processing data for file: {file_location}')
            process_subject_data(file_location)
    except Exception as e:
        log.error(f'An error occurred: {str(e)}', exc_info=True)


if __name__ == '__main__':
    main()
