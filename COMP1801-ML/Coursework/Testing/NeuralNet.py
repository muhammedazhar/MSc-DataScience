try:
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
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau       # type: ignore
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input # type: ignore
    from tensorflow.keras.models import Sequential                                # type: ignore
    from tensorflow.keras.optimizers import Adam                                  # type: ignore
    from typing import Dict, Tuple

except Exception as e:
    print(f"Error : {e}")

# Device configuration
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS!")
        print(f"Is Apple MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
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


def create_model(input_dim: int) -> Sequential:
    """
    Create a neural network model.

    Args:
        input_dim (int): The number of input features.

    Returns:
        Sequential: The compiled Keras Sequential model.
    """
    model = keras.Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1),
        ]
    )

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    # Print model summary for verification
    model.summary()

    return model


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Train the model with early stopping and learning rate reduction.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training targets.
        y_test (np.ndarray): Test targets.

    Returns:
        Tuple[Sequential, tf.keras.callbacks.History]: The trained model and training history.
    """
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=1e-5
    )

    # Create and train model
    input_dim = X_train.shape[1]
    model = create_model(input_dim)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=450,
        batch_size=38,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    return model, history


def evaluate_model(
    model: Sequential, X_test: np.ndarray, y_test: np.ndarray
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


def plot_results(
    history: tf.keras.callbacks.History, y_test: np.ndarray, predictions: np.ndarray
) -> None:
    """
    Create comprehensive visualization of model performance.

    Args:
        history (tf.keras.callbacks.History): The training history.
        y_test (np.ndarray): True target values.
        predictions (np.ndarray): Predicted target values.
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Plot 1: Training History
    losses = pd.DataFrame(history.history)
    sns.lineplot(data=losses, ax=axes[0, 0], lw=2)
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")

    # Plot 2: Prediction vs Actual
    axes[0, 1].scatter(y_test, predictions, alpha=0.5)
    axes[0, 1].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
    )
    axes[0, 1].set_title("Predictions vs Actual Values")
    axes[0, 1].set_xlabel("Actual Values")
    axes[0, 1].set_ylabel("Predicted Values")

    # Plot 3: Error Distribution
    errors = y_test - predictions
    sns.histplot(errors, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Error Distribution")
    axes[1, 0].set_xlabel("Error")
    axes[1, 0].set_ylabel("Count")

    # Plot 4: Residuals
    axes[1, 1].scatter(predictions, errors, alpha=0.5)
    axes[1, 1].axhline(y=0, color="r", linestyle="--")
    axes[1, 1].set_title("Residual Plot")
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Residuals")

    plt.tight_layout()
    plt.show()


def display_single_prediction(
    model: keras.Sequential,
    df: pd.DataFrame,
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
    sample_features = df.drop(columns=target).iloc[np.random.randint(0, len(df))]

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
    original = df.iloc[index][target]

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model, history = train_model(
        X_train_scaled, X_test_scaled, y_train.values, y_test.values
    )

    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test_scaled, y_test.values)

    # Print metrics

    print(f"\n"+"-"*65+"\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")
    print("-"*65)
    # Plot results
    # plot_results(history, y_test.values, predictions)

    # Create test DataFrame with unscaled features and target
    test_df = X_test.copy()
    test_df[target] = y_test.values
    test_df.reset_index(drop=True, inplace=True)

    # Display prediction for first sample
    print("\nPrediction for a sample:")
    display_single_prediction(model, test_df, scaler, target, index=0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")