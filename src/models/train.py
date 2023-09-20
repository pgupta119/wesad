# Importing necessary libraries and modules.
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from itertools import cycle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from constants import LOG_FILE, OUTPUT_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
from src.analyzer.feature_analyzer import FeatureAnalysis
from src.utils import create_versioned_output_directory

# Suppress any warnings to maintain clean output.
warnings.simplefilter('ignore')
import logger

from constants import LOG_FILE, PROCESSED_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY
# Set up logging for tracking progress and potential issues.
log = logger.setup_logger(LOG_FILE)

class WESADLDA:
    def __init__(self):
        pass

    def plot_roc_curve(self, y_test, y_prob, n_classes, output_directory):
        """
        Plot ROC curve for multi-class data.

        Parameters:
        - y_test: Actual test labels.
        - y_prob: Predicted probabilities from the model.
        - n_classes: Number of unique classes.
        - output_directory: Directory to save the ROC curve.
        """
        # Convert test labels to binary format for ROC.
        y_test_bin = label_binarize(y_test, classes=[1, 2, 3])

        # Initialize dictionaries for true positive and false positive rates.
        fpr, tpr, roc_auc = {}, {}, {}

        # Calculate ROC curve for each class.
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate micro-average ROC curve.
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot the ROC curves.
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_directory, "ROC_Curve.png"))
        # plt.show()

    def transform(self):
        """
        Load data, train models, and evaluate them.

        Returns:
        X_train, X_test, y_train, y_test : Data splits.
        """
        # Load and preprocess the data.
        data_directory = f'{PROCESSED_DATA_DIRECTORY}'
        output_directory = f'{OUTPUT_DATA_DIRECTORY}features'
        feature_analysis = FeatureAnalysis(data_directory, output_directory)
        (wrist_data, chest_data, combined_data, wrist_selected, chest_selected, combined_selected) = feature_analysis.main()

        # Iterate through selected feature sets.
        for feature_selected in [wrist_selected, chest_selected, combined_selected]:
            feature_selected = feature_selected.tolist()
            feature_selected.append('label')
            samples = combined_data[feature_selected]

            # Split features from the target labels.
            samples = samples.dropna()
            X = samples.iloc[:, :28]
            y = samples.iloc[:, samples.columns == 'label'].values.ravel()

            # Split the dataset into training and testing.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Normalize the features using Standard Scaler.
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Define models and their respective hyperparameters for tuning.
            models = {
                "LDA": LinearDiscriminantAnalysis(n_components=2),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier()
            }

            param_grids = {
                "LDA": {"solver": ["svd", "lsqr", "eigen"]},
                "DecisionTree": {"max_depth": [None, 2, 3, 4], "min_samples_leaf": [1, 2, 4]},
                "RandomForest": {"n_estimators": [2, 3, 4], "max_depth": [None, 2, 4], "min_samples_leaf": [1, 2, 4]}
            }

            # Create a new directory for output.
            output_directory = f'{OUTPUT_DATA_DIRECTORY}model_roc_graph'
            output_directory = create_versioned_output_directory(output_directory)

            # Train and evaluate each model using GridSearchCV and log the results using mlflow.
            for model_name, model in models.items():
                with mlflow.start_run() as run:
                    experiment_id = run.info.experiment_id
                    run_id = run.info.run_id
                    mlflow.log_param("experiment_id", experiment_id)
                    mlflow.log_param("run_id", run_id)
                    grid_search = GridSearchCV(model, param_grids[model_name], cv=2)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_

                    # Predict
                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test) if model_name != "LDA" else None

                    # Evaluation Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovo') if y_prob is not None else None
                    log.info("Accuracy, Precision, Recall, f1, roc_auc")
                    if roc_auc:
                        n_classes = y_prob.shape[1]
                        self.plot_roc_curve(y_test, y_prob, n_classes, output_directory)
                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                    })
                    if roc_auc:
                        mlflow.log_metrics({"roc_auc": roc_auc})

                        # Log the best model
                    mlflow.sklearn.log_model(best_model, "best_model")

                mlflow.end_run()
        return X_train, X_test, y_train, y_test
