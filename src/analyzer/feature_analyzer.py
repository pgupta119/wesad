import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from scipy.stats import chi2_contingency
import logger
from constants import LOG_FILE
from src.utils  import create_versioned_output_directory
log = logger.setup_logger(LOG_FILE)


class FeatureAnalysis:
    def __init__(self, data_directory, output_directory):
        self.data_directory = data_directory
        self.output_directory = output_directory

    def load_data(self, data_path):
        return pd.read_parquet(data_path)

    def random_forest_importance(self, data, target_column):
        """
        Implements random forest importance feature selection
        :param data:
        :param target_column:
        :return:
        """
        X = data.drop(columns=target_column)
        y = data[target_column]

        model = RandomForestClassifier()
        model.fit(X, y)

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return feature_importance

    def recursive_feature_elimination(self, data, target_column, n_features_to_select=10):
        """
        recursive feature elimination using random forest
        :param data:
        :param target_column:
        :param n_features_to_select:
        :return:
        """
        X = data.drop(columns=target_column)
        y = data[target_column]

        model = RandomForestClassifier()
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_]
        return selected_features

    def chi_squared_test(self, data, target_column):
        """
        chi squared test for feature selection
        :param data:
        :param target_column:
        :return:
        """
        significant_features = []
        for column in data.columns:
            if column != target_column:
                contingency_table = pd.crosstab(data[column], data[target_column])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                if p < 0.05:  # assuming 5% level of significance
                    significant_features.append(column)
        return significant_features

    def plot_feature_importance(self, importance_df, title):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(title)
        output_path = os.path.join(self.output_directory, f"{title}.png")
        plt.savefig(output_path)
        # plt.show()

    def feature_importance(self, data, target_column):
        importance = self.random_forest_importance(data, target_column)
        self.plot_feature_importance(importance, f'{target_column} Feature Importance')

    def main(self):
        # Log the start of the process
        log.info("Starting feature analysis...")
        self.output_directory = create_versioned_output_directory(self.output_directory)

        wrist_data_path = os.path.join(self.data_directory, 'wrist_data.parquet')
        chest_data_path = os.path.join(self.data_directory, 'chest_data.parquet')
        readme_data_path = os.path.join(self.data_directory, 'readme_data.parquet')
        log.info("Loading data")
        wrist_data = self.load_data(wrist_data_path)
        chest_data = self.load_data(chest_data_path)
        readme_data = self.load_data(readme_data_path)
        log.info("Data loaded successfully")
        log.info("Combining data...")
        combined_data = pd.concat([wrist_data, chest_data], axis=1)
        combined_data.drop(columns={'wrist_unique_id', 'wrist_label'}, inplace=True)
        combined_data.rename(columns={'chest_unique_id': 'common_unique_id', 'chest_label': 'label'}, inplace=True)
        readme_data.rename(columns={'unique_id': 'common_unique_id'}, inplace=True)
        combined_data = pd.merge(combined_data, readme_data, on='common_unique_id')
        combined_data.drop(columns=['common_unique_id', 'wrist_Time_stamp', 'chest_Time_stamp'], inplace=True)
        chest_data.drop(columns={'chest_unique_id', 'chest_Time_stamp'}, inplace=True)
        wrist_data.drop(columns={'wrist_unique_id', 'wrist_Time_stamp'}, inplace=True)
        wrist_data = wrist_data.dropna()
        chest_data = chest_data.dropna()
        combined_data = combined_data.dropna()
        # Perform feature analysis for different data sets
        log.info("Performing feature analysis...")
        self.feature_importance(wrist_data, 'wrist_label')
        self.feature_importance(chest_data, 'chest_label')
        self.feature_importance(combined_data, 'label')
        wrist_selected = self.recursive_feature_elimination(wrist_data, 'wrist_label')
        chest_selected = self.recursive_feature_elimination(chest_data, 'chest_label')
        combined_selected = self.recursive_feature_elimination(combined_data, 'label')

        # Log the end of the process
        log.info("Feature analysis completed.")
        return wrist_data, chest_data, combined_data, wrist_selected, chest_selected, combined_selected
