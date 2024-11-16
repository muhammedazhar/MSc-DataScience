"""
COMP1801-ML Coursework T1-Regression Implementation
---------------------------------------------------
Neural Network Pipeline for Metal Parts Lifespan Prediction
"""

try:
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import tensorflow as tf
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        root_mean_squared_error,
        r2_score,
        mean_absolute_error
    )
    from tensorflow.keras.models import Sequential                                 # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau        # type: ignore
    from tensorflow.keras.optimizers import Adam                                   # type: ignore

except Exception as e:
    print(f"Error: {e}")


class MetalPartsPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.history = None

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def load_data(self, filepath):
        """Load and prepare the dataset."""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        return df

    def create_neural_network(self, input_dim):
        """Create the neural network architecture."""
        model = Sequential([
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
            Dense(1)
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")

        # Print model summary for verification
        model.summary()

        return model

    def prepare_pipeline(self, df, target='Lifespan'):
        """Create preprocessing and model pipeline."""
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Identify column types
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns
                       if col not in categorical_cols]

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )

        return X, y

    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train the model and evaluate performance."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Create callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=10,
            min_lr=1e-5
        )

        # Create and train model
        input_dim = X_train_processed.shape[1]
        self.model = self.create_neural_network(input_dim)

        # Train the model
        self.history = self.model.fit(
            X_train_processed,
            y_train,
            validation_data=(X_test_processed, y_test),
            epochs=450,
            batch_size=38,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Make predictions
        predictions = self.model.predict(X_test_processed, verbose=0).flatten()

        # Calculate metrics
        metrics = {
            'RMSE': root_mean_squared_error(y_test, predictions),
            'R²': r2_score(y_test, predictions),
            'MAE': mean_absolute_error(y_test, predictions)
        }

        return metrics, (X_train, X_test, y_train, y_test)

    def make_random_predictions(self, X_test, y_test, n_predictions=5):
        """Generate random predictions from test set."""
        X_test_processed = self.preprocessor.transform(X_test)
        test_indices = np.random.choice(len(X_test), n_predictions, replace=False)

        print("\n=== Random Predictions from Test Set ===")

        for idx in test_indices:
            sample = X_test.iloc[[idx]]
            sample_processed = self.preprocessor.transform(sample)
            true_value = y_test.iloc[idx]
            prediction = self.model.predict(sample_processed, verbose=0)[0][0]

            error = abs(prediction - true_value)
            error_percentage = (error / true_value) * 100 if true_value != 0 else 0

            print("\n" + "="*50)
            print("Sample Features:")
            for feature, value in sample.iloc[0].items():
                print(f"{feature}: {value}")

            print(f"\nPrediction Results:")
            print(f"Predicted Lifespan: {prediction:.2f} hours")
            print(f"Actual Lifespan: {true_value:.2f} hours")
            print(f"Absolute Error: {error:.2f} hours")
            print(f"Error Percentage: {error_percentage:.2f}%")
            print("="*50)


def main():
    # Initialize predictor
    predictor = MetalPartsPredictor()

    # Load data
    data_path = Path('../Datasets').glob('*.csv')
    try:
        filepath = next(data_path)
    except StopIteration:
        raise FileNotFoundError("No CSV file found in the Datasets directory.")

    df = predictor.load_data(filepath)

    # Prepare and train model
    X, y = predictor.prepare_pipeline(df)
    metrics, (X_train, X_test, y_train, y_test) = predictor.train_and_evaluate(X, y)

    # Generate random predictions
    predictor.make_random_predictions(X_test, y_test)

    # Print metrics
    print("\n--- Performance of Neural Network with Pipeline ---")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R²  : {metrics['R²']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
