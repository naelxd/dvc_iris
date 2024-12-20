import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

def preprocess_data(input_path: str, train_output_path: str, test_output_path: str, test_size: float, random_state: int):
    """
    Preprocess the Iris dataset by adding column names and splitting it into train and test sets.

    Parameters:
    - input_path (str): Path to the raw dataset file.
    - train_output_path (str): Path to save the training dataset file.
    - test_output_path (str): Path to save the testing dataset file.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.
    """
    # Define column names for the Iris dataset
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found.")

    # Read the dataset
    data = pd.read_csv(input_path, header=None, names=columns)

    # Split dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["class"])

    # Save train and test datasets
    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

    print(f"Train data saved to {train_output_path}")
    print(f"Test data saved to {test_output_path}")

if __name__ == "__main__":
    # Загрузить параметры из params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Пути к входным и выходным данным
    input_file = "data/iris.csv"
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    # Извлечь параметры для разделения данных
    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    preprocess_data(input_file, train_file, test_file, test_size, random_state)

