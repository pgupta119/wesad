import pytest
import os
import pandas as pd
from src.analyzer.feature_analyzer import FeatureAnalysis

# Fixture to create an instance of FeatureAnalysis for testing
# @pytest.fixture
# def feature_analysis_instance():
#     data_directory = "path/to/your/data"
#     output_directory = "path/to/your/output"
#     return FeatureAnalysis(data_directory, output_directory)
#
# # Test cases for the FeatureAnalysis class
# class TestFeatureAnalysis:
#
#     @pytest.mark.parametrize("data_path", ["path/to/your/data/wrist_data.parquet"])
#     def test_load_data(self, feature_analysis_instance, data_path):
#         data = feature_analysis_instance.load_data(data_path)
#         assert isinstance(data, pd.DataFrame)
#
#     @pytest.mark.parametrize("data, target_column", [(pd.DataFrame({'feature1': [1, 2, 3], 'label': [0, 1, 0]}), 'label')])
#     def test_random_forest_importance(self, feature_analysis_instance, data, target_column):
#         importance = feature_analysis_instance.random_forest_importance(data, target_column)
#         assert isinstance(importance, pd.DataFrame)
#
#     @pytest.mark.parametrize("data, target_column, n_features_to_select", [
#         (pd.DataFrame({'feature1': [1, 2, 3], 'label': [0, 1, 0]}), 'label', 2)
#     ])
#     def test_recursive_feature_elimination(self, feature_analysis_instance, data, target_column, n_features_to_select):
#         selected_features = feature_analysis_instance.recursive_feature_elimination(data, target_column, n_features_to_select)
#         assert isinstance(selected_features, pd.Index)
#
#     @pytest.mark.parametrize("data, target_column", [(pd.DataFrame({'feature1': [1, 2, 3], 'label': [0, 1, 0]}), 'label')])
#     def test_chi_squared_test(self, feature_analysis_instance, data, target_column):
#         significant_features = feature_analysis_instance.chi_squared_test(data, target_column)
#         assert isinstance(significant_features, list)
