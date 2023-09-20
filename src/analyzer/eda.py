# Importing necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logger
from constants import LOG_FILE

# Setting up logging
log = logger.setup_logger(LOG_FILE)


# EDA (Exploratory Data Analysis) Class
class EDA:
    # Initializer method
    def __init__(self, data_directory, base_output_directory):
        self.data_directory = data_directory  # Directory to load data from
        self.base_output_directory = base_output_directory  # Base directory to save plots

    # Method to create directory if not exists
    def create_directory(self, path):
        log.info("Creating directory...")
        if not os.path.exists(path):
            os.makedirs(path)

    # Method to get the next version directory name for saving plots
    def get_next_version(self, directory_base):
        log.info("Getting next version...")
        version = 1
        while True:
            directory = os.path.join(directory_base, f'v{version}/')
            if not os.path.exists(directory):
                return f"v{version}"
            version += 1

    # Method to initialize and return the directory path where plots for the current run will be saved
    def initialize_directory_for_run(self, data_type):
        log.info("Initializing directory for run...")
        today = datetime.today().strftime('%Y-%m-%d')
        directory_base = os.path.join(self.base_output_directory, f'{data_type}/{today}_plot')
        current_version = self.get_next_version(directory_base)
        return os.path.join(directory_base, current_version)

    # Method to save plots based on given type
    def save_plot(self, df, column, plot_type, data_type, directory):
        log.info("Saving plot...")
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure(figsize=(10, 5))

        # Plot based on the type provided
        if plot_type == 'Distribution':
            sns.histplot(df[column], kde=True)
        elif plot_type == 'Boxplot':
            sns.boxplot(x=df[column])
        elif plot_type == 'Correlation_Matrix':
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')

        plt.title(f'{plot_type} of {column}')
        plt.savefig(os.path.join(directory, f'{plot_type}_{column}.png'))
        plt.close()

    # Method to load data from given path
    def load_data(self, data_path):
        log.info("Loading data...")
        return pd.read_parquet(data_path)

    # Method to plot distributions of all columns
    def plot_distributions(self, df, data_type, directory):
        log.info("Plotting distributions...")
        for column in df.columns:
            self.save_plot(df, column, 'Distribution', data_type, directory)

    # Method to plot boxplots of all columns
    def plot_boxplots(self, df, data_type, directory):
        log.info("Plotting boxplots...")
        for column in df.columns:
            self.save_plot(df, column, 'Boxplot', data_type, directory)

    # Method to plot correlation matrix of the dataframe
    def plot_correlation_matrix(self, df, data_type, directory):
        log.info("Plotting correlation matrix...")
        self.save_plot(df, 'correlation_matrix', 'Correlation_Matrix', data_type, directory)

    # Main method to run the entire EDA process
    def main(self):
        log.info("Starting EDA process...")
        data_types = ['wrist_data', 'chest_data']  # Different types of data to be analyzed
        data_names = ['wrist', 'chest']  # Names of the datasets to be used while dropping columns
        index = 0

        for data_type in data_types:
            current_version = self.initialize_directory_for_run(data_type)
            data_path = os.path.join(self.data_directory, f'{data_type}.parquet')
            df = self.load_data(data_path)

            # Drop unnecessary columns like unique ID and timestamps
            df = df.drop(columns={f'{data_names[index]}_unique_id', f'{data_names[index]}_Time_stamp'})
            df = df.dropna()  # Drop rows with missing values
            index += 1

            # Plotting the distributions, boxplots, and correlation matrix for each dataset
            self.plot_distributions(df, data_type, current_version)
            self.plot_boxplots(df, data_type, current_version)
            self.plot_correlation_matrix(df, data_type, current_version)
