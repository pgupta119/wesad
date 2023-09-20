import pandas as pd
import numpy as np
import scipy
import scipy.signal as scisig
import time
import warnings
import logger
from constants import LOG_FILE
from constants import PROCESSED_DATA_DIRECTORY

# Disable certain warnings for cleaner output
warnings.simplefilter('ignore')

from src.utils import save_to_parquet

log = logger.setup_logger(LOG_FILE)


class ChestFeatureExtractor:
    def __init__(self, data, fs_dict, window_in_seconds):
        self.data = data
        self.fs_dict = fs_dict  # Dictionary containing sampling frequencies for each label
        self.window_in_seconds = window_in_seconds  # Duration of the sliding window

    def process(self, unique_id):
        # Group data by the 'label' column
        grouped = self.data.groupby('label')
        baseline = grouped.get_group(1)
        stress = grouped.get_group(2)
        amusement = grouped.get_group(3)

        # Get sample features for each label group
        baseline_samples = self.get_samples(baseline, 1, unique_id)
        stress_samples = self.get_samples(stress, 2, unique_id)
        amusement_samples = self.get_samples(amusement, 3, unique_id)

        # Combine all samples and save to parquet file
        all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
        all_samples = all_samples.add_prefix('chest_')
        save_to_parquet(all_samples, path=f'{PROCESSED_DATA_DIRECTORY}/chest_data.parquet')

        return all_samples

    def get_samples(self, data, label, unique_id):
        global feat_names
        feat_names = None
        stride_length = int(self.fs_dict['label'] * 5)  # sliding length of 5 seconds
        window_len = self.fs_dict['label'] * self.window_in_seconds
        n_windows = int((len(data) - window_len) / stride_length) + 1  # Calculate the number of sliding windows
        samples = []

        for i in range(n_windows):
            # Determine start and end indices for each window
            start_idx = stride_length * i
            end_idx = start_idx + window_len
            w = data[start_idx: end_idx]
            w = pd.concat([w, self.get_net_accel(w)])

            # Rename columns for uniformity
            cols = list(w.columns)
            cols[-1] = 'net_acc'
            w.columns = cols

            # Compute window statistics
            wstats = self.get_window_stats(data=w, label=label)
            x = pd.DataFrame(wstats).drop('label', axis=0)
            y = x['label'][0]
            x.drop('label', axis=1, inplace=True)
            if feat_names is None:
                feat_names = [f'{row}_{col}' for row in x.index for col in x.columns]

            # Construct DataFrame from computed stats
            wdf = pd.DataFrame(x.values.flatten()).T
            wdf.columns = feat_names
            wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)

            # Compute slope and assign unique identifier
            wdf['TEMP_slope'] = self.get_slope(w['Temp'].dropna())
            wdf['Time_stamp'] = time.time()
            wdf['unique_id'] = unique_id
            samples.append(wdf)

        return pd.concat(samples)

    @staticmethod
    def get_slope(series):
        """Compute the slope of a series using linear regression."""
        linreg = scipy.stats.linregress(np.arange(len(series)), series)
        return linreg[0]

    @staticmethod
    def get_window_stats(data, label=-1):
        """Calculate various statistical features for a given window."""
        mean_features = np.mean(data, axis=0)
        std_features = np.std(data, axis=0)
        min_features = np.amin(data, axis=0)
        max_features = np.amax(data, axis=0)
        features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                    'label': label}
        return features

    @staticmethod
    def get_net_accel(data):
        """Calculate net acceleration using the Pythagorean theorem."""
        return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

    @staticmethod
    def get_peak_freq(x):
        """Get the peak frequency from the periodogram of a signal."""
        f, Pxx = scisig.periodogram(x, fs=8)  # Use a sampling frequency of 8 Hz for the periodogram
        psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
        peak_freq = psd_dict[max(psd_dict.keys())]
        return peak_freq
