import pandas as pd
import numpy as np
import scipy
import scipy.signal as scisig
import time
from constants import LOG_FILE, PROCESSED_DATA_DIRECTORY
import warnings
from src.utils import save_to_parquet

# Suppress specified warnings for cleaner output.
warnings.simplefilter('ignore')


class WristFeatureExtractor:
    def __init__(self, data, fs_dict, window_in_seconds):
        """
        Initialize the WristFeatureExtractor class.

        Parameters:
        - data: DataFrame containing the sensor readings.
        - fs_dict: Dictionary containing sampling frequencies for each label.
        - window_in_seconds: Duration for the sliding window.
        """
        self.data = data
        self.fs_dict = fs_dict
        self.window_in_seconds = window_in_seconds

    def process(self, unique_id):
        """
        Process the data to extract features for given labels and save to parquet format.

        Parameters:
        - unique_id: Identifier for the data segment being processed.

        Returns:
        - DataFrame containing extracted features.
        """
        # Group data by the 'label' column.
        grouped = self.data.groupby('label')
        baseline = grouped.get_group(1)
        stress = grouped.get_group(2)
        amusement = grouped.get_group(3)

        # Get sample features for each label group.
        baseline_samples = self.get_samples(baseline, 1, unique_id)
        stress_samples = self.get_samples(stress, 2, unique_id)
        amusement_samples = self.get_samples(amusement, 3, unique_id)

        # Combine all samples, prefix columns for distinction, and save to parquet format.
        all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
        all_samples = all_samples.add_prefix('wrist_')
        save_to_parquet(all_samples, path=f'{PROCESSED_DATA_DIRECTORY}wrist_data.parquet')
        return all_samples

    def get_samples(self, data, label, unique_id):
        """
        Extract features for the data within the specified window and stride length.

        Parameters:
        - data: DataFrame containing the sensor readings for a specific label.
        - label: Identifier for the type of data (e.g., baseline, stress).
        - unique_id: Identifier for the data segment being processed.

        Returns:
        - DataFrame containing extracted features for each window.
        """
        global feat_names
        feat_names = None
        stride_length = int(self.fs_dict['label'] * 5)  # Sliding length of 5 seconds.
        window_len = self.fs_dict['label'] * self.window_in_seconds
        # Calculate the number of sliding windows.
        n_windows = int((len(data) - window_len) / stride_length) + 1

        samples = []
        for i in range(n_windows):
            # Determine start and end indices for each window.
            start_idx = stride_length * i
            end_idx = start_idx + window_len
            w = data[start_idx: end_idx]
            w = pd.concat([w, self.get_net_accel(w)])

            # Rename columns for clarity.
            cols = list(w.columns)
            cols[-1] = 'net_acc'
            w.columns = cols

            # Compute window statistics.
            wstats = self.get_window_stats(data=w, label=label)
            x = pd.DataFrame(wstats).drop('label', axis=0)
            y = x['label'][0]
            x.drop('label', axis=1, inplace=True)
            if feat_names is None:
                feat_names = [f'{row}_{col}' for row in x.index for col in x.columns]

            # Construct DataFrame from computed statistics.
            wdf = pd.DataFrame(x.values.flatten()).T
            wdf.columns = feat_names
            wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)
            wdf['BVP_peak_freq'] = self.get_peak_freq(w['BVP'].dropna())
            wdf['TEMP_slope'] = self.get_slope(w['TEMP'].dropna())
            wdf['Time_stamp'] = time.time()
            wdf['unique_id'] = unique_id
            samples.append(wdf)

        return pd.concat(samples)

    @staticmethod
    def get_slope(series):
        """
        Compute the slope of a series using linear regression.

        Parameters:
        - series: A pandas Series.

        Returns:
        - Slope value.
        """
        linreg = scipy.stats.linregress(np.arange(len(series)), series)
        return linreg[0]

    @staticmethod
    def get_window_stats(data, label=-1):
        """
        Calculate statistical features for a given window.

        Parameters:
        - data: DataFrame segment.
        - label: Identifier for the type of data.

        Returns:
        - Dictionary containing statistical features.
        """
        mean_features = np.mean(data, axis=0)
        std_features = np.std(data, axis=0)
        min_features = np.amin(data, axis=0)
        max_features = np.amax(data, axis=0)
        features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                    'label': label}
        return features

    @staticmethod
    def get_net_accel(data):
        """
        Calculate net acceleration using the Pythagorean theorem.

        Parameters:
        - data: DataFrame containing accelerometer readings.

        Returns:
        - Net acceleration series.
        """
        return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

    @staticmethod
    def get_peak_freq(x):
        """
        Extract peak frequency from a signal.

        Parameters:
        - x: Signal data.

        Returns:
        - Peak frequency value.
        """
        f, Pxx = scisig.periodogram(x, fs=8)
        psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
        peak_freq = psd_dict[max(psd_dict.keys())]
        return peak_freq
