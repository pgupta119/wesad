from src.preprocessor.sensor.sensor import SensorDataProcessor
import pandas as pd
from constants import LOG_FILE
import logger

# Initialize a logger for logging relevant information and errors
log = logger.setup_logger(LOG_FILE)

class WristDataProcessor(SensorDataProcessor):
    """
    The WristDataProcessor class inherits from SensorDataProcessor and is responsible for
    processing sensor data specifically from the wrist.
    """

    def __init__(self, file_path):
        """
        Constructor to initialize the WristDataProcessor object.

        Parameters:
        - file_path: str, The path to the file containing wrist sensor data.
        """
        super().__init__(file_path)
        # Define the sampling frequency dictionary for each sensor data type
        self.fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}

    def process_data(self):
        """
        Method to process the wrist sensor data. This includes preprocessing steps for ACC, BVP, EDA, and TEMP data.

        Returns:
        - wrist_data: dict, The processed wrist sensor data.
        """
        # Retrieve the wrist sensor data
        wrist_data = self.data['signal']['wrist']

        # Preprocess Accelerometer (ACC) data
        acc_df = pd.DataFrame(wrist_data['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
        # Apply FIR filter to smooth the ACC data
        acc_df['ACC_x'] = self.apply_FIR_filter(acc_df['ACC_x'])
        acc_df['ACC_y'] = self.apply_FIR_filter(acc_df['ACC_y'])
        acc_df['ACC_z'] = self.apply_FIR_filter(acc_df['ACC_z'])
        wrist_data['ACC'] = acc_df[['ACC_x', 'ACC_y', 'ACC_z']].values

        # Preprocess Blood Volume Pulse (BVP) data using a bandpass filter
        wrist_data['BVP'] = self.butter_bandpass_filter(wrist_data['BVP'], 0.7, 3.7, self.fs_dict['BVP'])

        # Preprocess Electrodermal Activity (EDA) data using a lowpass filter
        wrist_data['EDA'] = self.butter_lowpass_filter(wrist_data['EDA'], 1, self.fs_dict['EDA'], order=6)

        # Smooth the Temperature (TEMP) data using a smoothing function, applied twice
        wrist_data['TEMP'] = self.smooth_signal(wrist_data['TEMP'])
        wrist_data['TEMP'] = self.smooth_signal(wrist_data['TEMP'])

        return wrist_data
