# Import required libraries
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sympy import im
from xgboost import XGBRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer
)

# Set up paths and directories
data_path = '../Datasets/*.csv'
destination = '../Models/'
os.makedirs(destination, exist_ok=True)

# Load the dataset
file_list = glob.glob(data_path)
if len(file_list) == 1:
    df = pd.read_csv(file_list[0])
    print(f"Loaded dataset: {file_list[0]}")
else:
    raise FileNotFoundError("No CSV file found or multiple CSV files found in the Datasets directory.")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}\n")

def apply_onehot_encoding(df):
    """
    Apply one-hot encoding to categorical columns
    """
    onehot_encoded_df = df.copy()
    ohe = OneHotEncoder(sparse_output=False, dtype=int, drop=None)
    encoded_data = ohe.fit_transform(onehot_encoded_df[categorical_cols].values)
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out(categorical_cols)
    )
    onehot_encoded_df = pd.concat(
        [onehot_encoded_df.drop(columns=categorical_cols), encoded_df],
        axis=1
    )
    return onehot_encoded_df, ohe

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name="XGBoost"):
    """
    Train and evaluate the XGBoost model
    """
    model = XGBRegressor(
        max_depth=10,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"--- Performance of {model_name} ---\n")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return model, predictions, (rmse, r2, mae)

def make_random_predictions(model, X_test, y_test, n_predictions=5):
    """
    Make predictions on random samples from the test set
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True test values
    n_predictions : int
        Number of random predictions to make
    """
    # Get random indices from test set
    test_indices = random.sample(range(len(X_test)), n_predictions)
    
    print("\n=== Random Predictions from Test Set ===")
    print("\nFormat: Feature Name: Scaled Value")
    print("=" * 50)
    
    for idx in test_indices:
        # Get the sample and convert it to DataFrame to preserve feature names
        sample = X_test.iloc[[idx]]  # Using [[idx]] to keep DataFrame format
        true_value = y_test.iloc[idx]
        prediction = model.predict(sample)[0]  # No need to reshape
        
        # Calculate prediction error
        error = abs(prediction - true_value)
        error_percentage = (error / true_value) * 100
        
        print("\nSample Features:")
        for feature, value in sample.iloc[0].items():
            print(f"{feature}: {value:.2f}")
        
        print(f"\n"+"=" * 50)
        print("Prediction Results:")
        print(f"Predicted Lifespan: {prediction:.2f} hours")
        print(f"Actual Lifespan: {true_value:.2f} hours")
        print(f"Absolute Error: {error:.2f} hours")
        print(f"Error Percentage: {error_percentage:.2f}%")
        print("=" * 50)

if __name__ == "__main__":
    # Apply one-hot encoding
    onehot_encoded_df, ohe = apply_onehot_encoding(df)
    
    # Important features based on analysis
    # important_features = [
    #     'partType_Blade', 'partType_Block', 'partType_Nozzle', 
    #     'partType_Valve', 'coolingRate', 'Nickel%', 'HeatTreatTime', 
    #     'Chromium%', 'quenchTime'
    # ]

    important_features = list(onehot_encoded_df.columns)
    
    # Prepare data with important features only
    reduced_df = onehot_encoded_df[important_features]
    X = reduced_df
    y = onehot_encoded_df['Lifespan']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Train and evaluate model
    model, predictions, metrics = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "XGBoost with Scaling"
    )
    
    # Perform cross-validation
    scorer = make_scorer(mean_absolute_error)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring=scorer
    )
    
    print("\nCross-Validation Results:")
    print(f"Mean CV MAE: {cv_scores.mean():.2f}")
    print(f"Standard Deviation of CV MAE: {cv_scores.std():.2f}")
    
    # Make random predictions
    print("\nGenerating random predictions from test set...")
    make_random_predictions(model, X_test_scaled, y_test)