"""
COMP1801-ML Coursework T1-Regression Implementation
---------------------------------------------------
XGBoost Hyperparameter Tuning Pipeline
"""

try:
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import (
        train_test_split,
        RandomizedSearchCV,
        GridSearchCV
    )
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        root_mean_squared_error,
        r2_score,
        mean_absolute_error
    )
    from xgboost import XGBRegressor
    from scipy.stats import uniform, randint

except Exception as e:
    print(f"Error: {e}")


class XGBoostTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.best_params = None
        
        # Define parameter grids
        self.grid_params = {
            'regressor__max_depth': [3, 5, 7, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__n_estimators': [100, 200, 300],
            'regressor__min_child_weight': [1, 3, 5],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        self.random_params = {
            'regressor__max_depth': randint(3, 12),
            'regressor__learning_rate': uniform(0.01, 0.3),
            'regressor__n_estimators': randint(100, 500),
            'regressor__min_child_weight': randint(1, 7),
            'regressor__subsample': uniform(0.6, 0.4),
            'regressor__colsample_bytree': uniform(0.6, 0.4)
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
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )

        # Create base pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(random_state=self.random_state, n_jobs=-1))
        ])

        return X, y

    def tune_hyperparameters(self, X, y, method='random', n_iter=100):
        """Tune hyperparameters using specified search method."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        if method.lower() == 'random':
            search = RandomizedSearchCV(
                self.model,
                param_distributions=self.random_params,
                n_iter=n_iter,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=1,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:  # grid search
            search = GridSearchCV(
                self.model,
                param_grid=self.grid_params,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )

        # Perform search
        print(f"\nPerforming {method} Search CV...")
        search.fit(X_train, y_train)
        self.best_params = search.best_params_
        
        # Train final model with best parameters
        self.model = search.best_estimator_
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'RMSE': root_mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
        }

        return metrics, (X_train, X_test, y_train, y_test)

    def make_random_predictions(self, X_test, y_test, n_predictions=5):
        """Generate random predictions from test set."""
        test_indices = np.random.choice(len(X_test), n_predictions, replace=False)

        print("\n=== Random Predictions from Test Set ===")

        for idx in test_indices:
            sample = X_test.iloc[[idx]]
            true_value = y_test.iloc[idx]
            prediction = self.model.predict(sample)[0]

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
    # Initialize tuner
    tuner = XGBoostTuner()

    # Load data
    data_path = Path('../Datasets').glob('*.csv')
    try:
        filepath = next(data_path)
    except StopIteration:
        raise FileNotFoundError("No CSV file found in the Datasets directory.")

    df = tuner.load_data(filepath)

    # Prepare pipeline
    X, y = tuner.create_pipeline(df)

    # Perform hyperparameter tuning
    search_method = input("Enter search method (random/grid): ").lower()
    if search_method not in ['random', 'grid']:
        print("Invalid method. Defaulting to random search.")
        search_method = 'random'

    metrics, (X_train, X_test, y_train, y_test) = tuner.tune_hyperparameters(X, y, method=search_method)

    # Print best parameters
    print("\nBest Parameters Found:")
    for param, value in tuner.best_params.items():
        print(f"{param}: {value}")

    # Generate random predictions
    tuner.make_random_predictions(X_test, y_test)

    # Print metrics
    print("\n--- Performance of Tuned XGBoost ---")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R²  : {metrics['R²']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")