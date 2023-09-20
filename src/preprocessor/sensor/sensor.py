import pickle
import pandas as pd
import scipy.signal as scisig
from scipy.signal import butter, lfilter, savgol_filter

class SensorDataProcessor:
    """
    Base class for processing sensor data.
    Provides functionalities for loading data, and applying various filters to process the data.
    """

    def __init__(self, file_path):
        """
        Constructor for initializing the SensorDataProcessor object.

        Parameters:
        - file_path: str, path to the file containing sensor data.
        """
        self.file_path = file_path
        self.fs_dict = None  # Initialize sampling frequency dictionary
        self.load_data()     # Load data immediately upon object instantiation

    def load_data(self):
        """Load data from the specified file_path using pickle."""
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def process_data(self):
        """Abstract method to process data, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def apply_FIR_filter(self, data, cutoff=0.4, numtaps=64):
        """
        Apply a Finite Impulse Response (FIR) filter on the data.

        Parameters:
        - data: list/ndarray, the input data to be filtered.
        - cutoff: float, the cutoff frequency for the filter.
        - numtaps: int, the number of taps for the FIR filter.

        Returns:
        - Filtered data.
        """
        f = cutoff / (self.fs_dict['ACC'] / 2.0)
        FIR_coeff = scisig.firwin(numtaps, f)
        return scisig.lfilter(FIR_coeff, 1, data)

    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        """
        Design a Butterworth bandpass filter.

        Parameters:
        - lowcut: float, lower cutoff frequency.
        - highcut: float, higher cutoff frequency.
        - fs: float, sampling frequency.
        - order: int, the order of the filter.

        Returns:
        - b, a: ndarray, filter coefficients.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        """
        Apply a Butterworth bandpass filter to the data.

        Parameters:
        - data: list/ndarray, the input data to be filtered.
        - lowcut: float, lower cutoff frequency.
        - highcut: float, higher cutoff frequency.
        - fs: float, sampling frequency.
        - order: int, the order of the filter.

        Returns:
        - y: list/ndarray, the filtered data.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass_filter(self, data, cutoff, fs, order=3):
        """
        Apply a Butterworth lowpass filter to the data.

        Parameters:
        - data: list/ndarray, the input data to be filtered.
        - cutoff: float, the cutoff frequency for the filter.
        - fs: float, sampling frequency.
        - order: int, the order of the filter.

        Returns:
        - y: list/ndarray, the filtered data.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def smooth_signal(self, temp_data):
        """
        Smooth the input data using a Savitzky-Golay filter.

        Parameters:
        - temp_data: list/ndarray, the input data to be smoothed.

        Returns:
        - Smoothed data.
        """
        return savgol_filter(temp_data, window_length=11, polyorder=3, mode='nearest')
