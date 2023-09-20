# Necessary imports
from src.preprocessor.sensor.sensor import SensorDataProcessor
import pandas as pd
from scipy.signal import savgol_filter
from constants import LOG_FILE, RAW_DATA_DIRECTORY
import logger

log = logger.setup_logger(LOG_FILE)

class ChestDataProcessor(SensorDataProcessor):
    """
    Processor class for chest sensor data. Extends the SensorDataProcessor class.
    This class contains methods to preprocess and clean sensor data specific to the chest.
    """

    def __init__(self, file_path):
        """
        Initialize the ChestDataProcessor.

        Parameters:
        - file_path: The path to the file containing chest sensor data.

        Attributes:
        - fs_dict: A dictionary representing sampling rates of various sensors.
        """
        super().__init__(file_path)  # Call the base class constructor
        self.fs_dict = {
            'ACC': 700, 'ECG': 700, 'EMG': 700, 'TEMP': 700,
            'EDA': 700, 'RESP': 700, 'label': 700
        }

    def process_data(self):
        """
        Processes the chest sensor data.
        This involves preprocessing the accelerometer data (ACC) and applying
        appropriate filters to other sensor data.

        Returns:
        - chest_data: A dictionary containing the preprocessed data.
        """

        # Extract the chest data from the main data
        chest_data = self.data['signal']['chest']


        # Preprocess accelerometer data (ACC)
        # Convert the ACC data into a DataFrame for easier manipulation
        acc_df = pd.DataFrame(chest_data['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])

        # Apply FIR filter to each ACC axis data for noise removal
        acc_df['ACC_x'] = self.apply_FIR_filter(acc_df['ACC_x'])
        acc_df['ACC_y'] = self.apply_FIR_filter(acc_df['ACC_y'])
        acc_df['ACC_z'] = self.apply_FIR_filter(acc_df['ACC_z'])

        # Update the ACC data in chest_data after preprocessing
        chest_data['ACC'] = acc_df[['ACC_x', 'ACC_y', 'ACC_z']].values

        # Apply smoothing for other sensors' data
        for i in ['ECG', 'EMG', 'EDA', 'Resp', 'Temp']:
            chest_data[i] = self.smooth_signal(chest_data[i])

        # Apply a bandpass filter to the ECG data to retain frequencies between 0.7 and 3.7 Hz
        chest_data['ECG'] = self.butter_bandpass_filter(chest_data['ECG'], 0.7, 3.7, self.fs_dict['ECG'])

        # Apply a lowpass filter to the EDA data to retain frequencies below 5 Hz
        chest_data['EDA'] = self.butter_lowpass_filter(chest_data['EDA'], 5, self.fs_dict['EDA'], order=2)

        # Apply a lowpass filter to the EMG data to retain frequencies below 0.5 Hz
        chest_data['EMG'] = self.butter_lowpass_filter(chest_data['EMG'], 0.5, self.fs_dict['EMG'], order=3)

        # Apply a bandpass filter to the Resp data to retain frequencies between 0.1 and 0.35 Hz
        chest_data['Resp'] = self.butter_bandpass_filter(chest_data['Resp'], 0.1, 0.35, self.fs_dict['RESP'])

        return chest_data
