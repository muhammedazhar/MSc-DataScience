"""
COMP1801-ML Coursework T1-Regression Implementation
---------------------------------------------------
XGBoost Pipeline for Metal Parts Lifespan Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer
)
from xgboost import XGBRegressor


class MetalPartsPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None

    def load_data(self, filepath):
        """Load and prepare the dataset."""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        return df

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
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )

        # Create pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                max_depth=10,
                n_estimators=200,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        return X, y

    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train the model and evaluate performance."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        predictions = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'RMSE': root_mean_squared_error(y_test, predictions),
            'R²': r2_score(y_test, predictions),
            'MAE': mean_absolute_error(y_test, predictions)
        }

        # Perform cross-validation
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring=scorer
        )

        metrics.update({
            'CV_MAE_mean': -cv_scores.mean(),
            'CV_MAE_std': cv_scores.std()
        })

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
    print("\n--- Performance of XGBoost with Pipeline ---")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R²  : {metrics['R²']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")

    print("\nCross-Validation Results:")
    print(f"Mean CV MAE: {metrics['CV_MAE_mean']:.2f}")
    print(f"Standard Deviation of CV MAE: {metrics['CV_MAE_std']:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
