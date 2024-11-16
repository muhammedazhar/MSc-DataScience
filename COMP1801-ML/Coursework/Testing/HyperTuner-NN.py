import glob
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from typing import Dict, Tuple

import keras_tuner as kt  # Import Keras Tuner


# Device configuration
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS!")
        print(
            f"Is Apple MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}"
        )
        print(f"Is Apple MPS available? {torch.backends.mps.is_available()}\n")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device!")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


device = get_device()


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset.

    Args:
        data_path (str): The glob pattern to match CSV files.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Find and load the CSV file
    file_list = glob.glob(data_path)
    if len(file_list) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV file, found {len(file_list)} files."
        )
    df = pd.read_csv(file_list[0])
    print(f"Loaded dataset from: {file_list[0]}")

    # Identify categorical features
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_data = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(categorical_features)
    )

    # Combine with numerical columns
    processed_df = pd.concat(
        [df.drop(columns=categorical_features), encoded_df], axis=1
    )

    return processed_df


def build_model(hp: kt.HyperParameters) -> keras.Sequential:
    """
    Build a neural network model with hyperparameters.

    Args:
        hp (kt.HyperParameters): Hyperparameters for tuning.

    Returns:
        keras.Sequential: The compiled Keras Sequential model.
    """
    model = keras.Sequential()
    model.add(Input(shape=(input_dim,)))

    # Tune the number of units in each Dense layer
    for i in range(hp.Int("num_layers", 1, 3)):
        units = hp.Int(f"units_{i}", min_value=16, max_value=128, step=16)
        model.add(Dense(units=units, activation="relu"))
        model.add(BatchNormalization())
        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(rate=dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


def train_tuned_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[keras.Sequential, tf.keras.callbacks.History]:
    """
    Train the model using hyperparameter tuning.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Validation features.
        y_train (np.ndarray): Training targets.
        y_test (np.ndarray): Validation targets.

    Returns:
        Tuple[keras.Sequential, tf.keras.callbacks.History]: The best model and training history.
    """
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=150,
        executions_per_trial=1,
        directory="TuningLogs",
        project_name="Regression-NN",
    )

    stop_early = EarlyStopping(monitor="val_loss", patience=5)

    tuner.search(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[stop_early],
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Retrain the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=[stop_early],
        verbose=1,
    )

    print(f"\n"+"-"*65+"\nBest Hyperparameters:")
    print(f"Number of Layers: {best_hps.get('num_layers')}")
    for i in range(best_hps.get("num_layers")):
        print(
            f"Units in Layer {i+1}: {best_hps.get(f'units_{i}')}, "
            f"Dropout Rate: {best_hps.get(f'dropout_{i}')}"
        )
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print("-"*65)

    return model, history


def evaluate_model(
    model: keras.Sequential, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate model performance with multiple metrics.

    Args:
        model (Sequential): The trained Keras model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: A dictionary of evaluation metrics and predictions.
    """
    predictions = model.predict(X_test, verbose=0).flatten()

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R²": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
    }

    return metrics, predictions


def display_single_prediction(
    model: keras.Sequential,
    test_df: pd.DataFrame,
    scaler: MinMaxScaler,
    target: str,
    index: int = 0,
) -> None:
    """
    Display features and make prediction for a single sample.

    Args:
        model (Sequential): The trained Keras model.
        df (pd.DataFrame): DataFrame containing features and target.
        scaler (MinMaxScaler): The scaler used to scale features.
        target (str): The name of the target column.
        index (int): The index of the sample to predict.
    """
    # Get features of a random sample
    sample_features = test_df.drop(columns=target).iloc[np.random.randint(0, len(test_df))]

    # Display features with proper formatting
    print("Features of the random sample:")
    print(sample_features.to_string())
    print()

    # Convert to DataFrame with feature names
    sample_features_df = pd.DataFrame(
        [sample_features.values], columns=sample_features.index
    )

    # Scale the features
    sample_features_scaled = scaler.transform(sample_features_df)

    # Make prediction
    prediction = model.predict(sample_features_scaled, verbose=0)[0, 0]

    # Get original target value
    original = test_df.iloc[index][target]

    # Display results
    print(f"\nPredicted {target} : {prediction:.2f}")
    print(f"Actual {target}    : {original:.2f}")

    # Calculate and display prediction error
    error = abs(prediction - original)
    error_percentage = (error / original) * 100 if original != 0 else 0
    print(f"\nAbsolute Error: {error:.2f}")
    print(f"Error Percentage: {error_percentage:.2f}%")


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    df = load_and_preprocess_data("../Datasets/*.csv")

    target = "Lifespan"

    # Split features and target
    X = df.drop(columns=target)
    y = df[target]

    global input_dim
    input_dim = X.shape[1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with hyperparameter tuning
    model, history = train_tuned_model(
        X_train_scaled, X_test_scaled, y_train.values, y_test.values
    )

    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test_scaled, y_test.values)

    # Print metrics
    print(f"\n{'-'*65}\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")
    print("-" * 65)

    # Create test DataFrame with unscaled features and target
    test_df = X_test.copy()
    test_df[target] = y_test.values
    test_df.reset_index(drop=True, inplace=True)

    # Display prediction for a sample
    print("\nPrediction for a sample:")
    display_single_prediction(model, test_df, scaler, target, index=0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
