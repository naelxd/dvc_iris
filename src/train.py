import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

def train_model(train_data_path: str, test_data_path: str, model_output_dir: str, params_path: str, metrics_output_path: str):
    """
    Train a machine learning model on the training dataset and evaluate on the test dataset for different n_estimators.
    
    Parameters:
    - train_data_path (str): Path to the training dataset file.
    - test_data_path (str): Path to the test dataset file.
    - model_output_dir (str): Directory to save the trained models.
    - params_path (str): Path to the parameters file (params.yaml).
    - metrics_output_path (str): Path to save the evaluation metrics.
    """
    # Load parameters
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Load training and test data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Prepare features and target variable for both train and test sets
    X_train = train_data.drop(columns="class")
    y_train = train_data["class"]
    X_test = test_data.drop(columns="class")
    y_test = test_data["class"]

    # Prepare for saving metrics
    all_metrics = []

    # Train models for different values of n_estimators
    for n_estimators in params["model"]["n_estimators_values"]:
        # Initialize the model with the current value of n_estimators
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=params["model"]["max_depth"],
            random_state=params["model"]["seed"]
        )

        # Train the model
        model.fit(X_train, y_train)

        # Save the trained model
        model_output_path = f"{model_output_dir}/model_{n_estimators}.pkl"
        joblib.dump(model, model_output_path)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' handles class imbalance
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store metrics for this model
        metrics = {
            "n_estimators": n_estimators,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        all_metrics.append(metrics)

        print(f"Model trained with n_estimators={n_estimators} and saved to {model_output_path}")
        print(f"Evaluation metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

    # Save all metrics to file
    with open(metrics_output_path, "w") as f:
        yaml.dump(all_metrics, f)

if __name__ == "__main__":
    # Paths to input files and output files
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    model_output_dir = "models"  # Directory to save trained models
    params_file = "params.yaml"
    metrics_file = "metrics.yaml"

    # Train the model and evaluate
    train_model(train_file, test_file, model_output_dir, params_file, metrics_file)

