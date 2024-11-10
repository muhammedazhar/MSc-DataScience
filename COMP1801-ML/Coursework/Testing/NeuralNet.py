import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the dataset with proper error handling
    """
    try:
        # Find and load the CSV file
        file_list = glob.glob(data_path)
        if len(file_list) != 1:
            raise FileNotFoundError("Expected exactly one CSV file")

        df = pd.read_csv(file_list[0])
        print(f"Loaded dataset from: {file_list[0]}")

        # Define categorical columns
        categorical_cols = ['partType', 'microstructure', 'seedLocation', 'castType']

        # One-hot encode categorical variables
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        # Combine with numerical columns
        processed_df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

        return processed_df

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def create_model(input_dim):
    """
    Create an enhanced neural network model with proper input layer definition
    """
    model = Sequential([
        # Input layer using proper Input definition
        Input(shape=(input_dim,)),

        # First hidden layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Second hidden layer
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Third hidden layer
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),

        # Output layer
        Dense(1)
    ])

    # Compile with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # Print model summary for verification
    model.summary()

    return model

def train_model(X_train, X_test, y_train, y_test):
    """
    Train the model with early stopping and learning rate reduction
    """
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0.00001
    )

    # Create and train model
    model = create_model(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=400,
        batch_size=42,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with multiple metrics
    """
    predictions = model.predict(X_test, verbose=0)

    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'MSE': mean_squared_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'Variance Score': explained_variance_score(y_test, predictions)
    }

    return metrics, predictions

def plot_results(history, y_test, predictions):
    """
    Create comprehensive visualization of model performance
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Plot 1: Training History
    losses = pd.DataFrame(history.history)
    sns.lineplot(data=losses, ax=axes[0,0], lw=2)
    axes[0,0].set_title('Training and Validation Loss')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('Loss')

    # Plot 2: Prediction vs Actual
    axes[0,1].scatter(y_test, predictions, alpha=0.5)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,1].set_title('Predictions vs Actual Values')
    axes[0,1].set_xlabel('Actual Values')
    axes[0,1].set_ylabel('Predicted Values')

    # Plot 3: Error Distribution
    errors = y_test - predictions.flatten()
    sns.histplot(errors, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Error Distribution')
    axes[1,0].set_xlabel('Error')
    axes[1,0].set_ylabel('Count')

    # Plot 4: Residuals
    axes[1,1].scatter(predictions, errors, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_title('Residual Plot')
    axes[1,1].set_xlabel('Predicted Values')
    axes[1,1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

def display_single_prediction(model, df, scaler, index=0):
    """
    Display features and make prediction for a single part
    """
    # Get features of specified part type
    single_partType = df.drop('Lifespan', axis=1).iloc[index]

    # Display features with proper formatting
    print('Features of new part type:')
    print(single_partType.to_string())
    print()

    # Convert to DataFrame with feature names
    single_partType_df = pd.DataFrame([single_partType.values],
                                    columns=single_partType.index)

    # Scale the features
    single_partType_scaled = scaler.transform(single_partType_df)

    # Make prediction
    prediction = model.predict(single_partType_scaled, verbose=1)[0,0]

    # Get original lifespan
    original = df.iloc[index]['Lifespan']

    # Display results
    print(f'\nPrediction Lifespan: {prediction:.4f}')
    print(f'\nOriginal Lifespan: {original:.2f}')

    # Calculate and display prediction error
    error = abs(prediction - original)
    error_percentage = (error / original) * 100
    print(f'\nAbsolute Error: {error:.4f}')
    print(f'Error Percentage: {error_percentage:.2f}%')

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data('../Datasets/*.csv')

    # Split features and target
    X = df.drop('Lifespan', axis=1)
    y = df['Lifespan']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model, history = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test_scaled, y_test)

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Plot results
    plot_results(history, y_test, predictions)

    # Display prediction for first part
    print("\nPrediction for first part:")
    display_single_prediction(model, df, scaler, index=0)
