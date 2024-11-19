"""
COMP1801-ML Coursework T1-Regression Implementation
---------------------------------------------------
Neural Network Hyperparameter Tuning Pipeline with Command Line Arguments
"""

try:
    import pandas as pd
    import numpy as np
    import argparse
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
    from tensorflow.keras.models import Sequential        # type: ignore
    from tensorflow.keras.layers import (                 # type: ignore
        Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    from tensorflow.keras.optimizers import Adam          # type: ignore
    import keras_tuner as kt

except Exception as e:
    print(f"Error: {e}")


class NeuralNetTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.best_params = None
        self.input_dim = None
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        # Define parameter grids
        self.grid_params = {
            'num_layers': [1, 2, 3],
            'units_per_layer': [32, 64, 128],
            'dropout_rate': [0.1, 0.3, 0.5],
            'learning_rate': [1e-2, 1e-3, 1e-4]
        }

    def load_data(self, filepath):
        """Load and prepare the dataset."""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        return df

    def create_pipeline(self, df, target='Lifespan'):
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

        self.input_dim = len(numeric_cols) + sum(len(df[col].unique()) for col in categorical_cols)
        return X, y

    def build_model(self, hp, method='random'):
        """Build neural network model with specified hyperparameters."""
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        
        if method == 'random':
            # Random search hyperparameters
            n_layers = hp.Int('num_layers', 1, 3)
            for i in range(n_layers):
                units = hp.Int(f'units_{i}', min_value=32, max_value=128, step=32)
                model.add(Dense(units=units, activation='relu'))
                model.add(BatchNormalization())
                dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
                model.add(Dropout(rate=dropout_rate))
        else:
            # Grid search hyperparameters
            n_layers = hp['num_layers']
            units = hp['units_per_layer']
            dropout_rate = hp['dropout_rate']
            for _ in range(n_layers):
                model.add(Dense(units=units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(rate=dropout_rate))

        model.add(Dense(1))
        
        learning_rate = hp['learning_rate'] if method == 'grid' else hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def tune_hyperparameters(self, X, y, method='random', n_iter=100):
        """Tune hyperparameters using specified search method."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        if method.lower() == 'random':
            tuner = kt.RandomSearch(
                lambda hp: self.build_model(hp, 'random'),
                objective='val_loss',
                max_trials=n_iter,
                directory="TuningLogs",
                project_name="Regression-NN",
                overwrite=True
            )
            
            tuner.search(
                X_train_processed, y_train,
                validation_data=(X_test_processed, y_test),
                epochs=100,
                callbacks=[early_stopping],
                verbose=1
            )
            
            self.best_params = tuner.get_best_hyperparameters(1)[0]
            self.model = tuner.get_best_models(1)[0]
            
        else:  # grid search
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            
            # Iterate through all combinations
            for n_layers in self.grid_params['num_layers']:
                for units in self.grid_params['units_per_layer']:
                    for dropout in self.grid_params['dropout_rate']:
                        for lr in self.grid_params['learning_rate']:
                            current_params = {
                                'num_layers': n_layers,
                                'units_per_layer': units,
                                'dropout_rate': dropout,
                                'learning_rate': lr
                            }
                            
                            model = self.build_model(current_params, 'grid')
                            history = model.fit(
                                X_train_processed, y_train,
                                validation_data=(X_test_processed, y_test),
                                epochs=100,
                                callbacks=[early_stopping],
                                verbose=0
                            )
                            
                            val_loss = min(history.history['val_loss'])
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model = model
                                best_params = current_params
            
            self.model = best_model
            self.best_params = best_params

        # Make predictions and calculate metrics
        y_pred = self.model.predict(X_test_processed, verbose=0).flatten()
        
        metrics = {
            'RMSE': root_mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural Network Hyperparameter Tuning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python HyperTuner-NN.py --method random --iterations 100
  python HyperTuner-NN.py --method grid
  python HyperTuner-NN.py --method random --iterations 50 --seed 123
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['random', 'grid'],
        help='Search method (random or grid)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for random search (default: 100)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the dataset (default: ../Datasets/*.csv)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def main():
    # Set up argument parser
    parser = parse_arguments()
    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(parser.parse_args([])).values()):
        parser.print_help()
        print("\nNo arguments provided. Please enter the search method:")
        args.method = input("Enter search method (random/grid): ").lower()
        if args.method == 'random':
            try:
                args.iterations = int(input("Enter number of iterations (default 100): ") or "100")
            except ValueError:
                print("Invalid input. Using default value of 100 iterations.")
                args.iterations = 100

    # Validate search method
    if args.method not in ['random', 'grid']:
        print("Invalid method. Defaulting to random search.")
        args.method = 'random'

    # Initialize tuner with specified seed
    tuner = NeuralNetTuner(random_state=args.seed)

    # Load data
    data_path = args.data if args.data else '../Datasets/*.csv'
    try:
        filepath = next(Path().glob(data_path))
    except StopIteration:
        raise FileNotFoundError(f"No CSV file found at {data_path}")

    df = tuner.load_data(filepath)

    # Prepare pipeline
    X, y = tuner.create_pipeline(df)

    # Show configuration if verbose
    if args.verbose:
        print("\nConfiguration:")
        print(f"Search Method: {args.method}")
        print(f"Random Seed: {args.seed}")
        if args.method == 'random':
            print(f"Number of Iterations: {args.iterations}")
        print(f"Data Path: {filepath}")
        print()

    # Perform hyperparameter tuning
    metrics, (X_train, X_test, y_train, y_test) = tuner.tune_hyperparameters(
        X, y, 
        method=args.method,
        n_iter=args.iterations
    )

    # Print best parameters
    print("\nBest Parameters Found:")
    if args.method == 'random':
        print("Number of Layers:", tuner.best_params.get('num_layers'))
        for i in range(tuner.best_params.get('num_layers')):
            print(f"Layer {i+1}:")
            print(f"  Units: {tuner.best_params.get(f'units_{i}')}")
            print(f"  Dropout Rate: {tuner.best_params.get(f'dropout_{i}')}")
        print(f"Learning Rate: {tuner.best_params.get('learning_rate')}")
    else:
        for param, value in tuner.best_params.items():
            print(f"{param}: {value}")

    # Generate random predictions
    tuner.make_random_predictions(X_test, y_test)

    # Print metrics
    print("\n--- Performance of Tuned Neural Network ---")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R²  : {metrics['R²']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
