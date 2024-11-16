# Import required libraries
import os
import glob
from idna import encode
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer
)
from sklearn.compose import ColumnTransformer

# Function to make random predictions
def make_random_predictions(model, X_test, y_test, n_predictions=5):
    test_indices = random.sample(range(len(X_test)), n_predictions)

    print("\n=== Random Predictions from Test Set ===")
    print("\nFormat: Feature Name: Value")
    print("=" * 50)

    for idx in test_indices:
        sample = X_test.iloc[[idx]]
        true_value = y_test.iloc[idx]
        prediction = model.predict(sample)[0]

        error = abs(prediction - true_value)
        error_percentage = (error / true_value) * 100 if true_value != 0 else 0

        print("\nSample Features:")
        for feature, value in sample.iloc[0].items():
            print(f"{feature}: {value}")

        print(f"\n{'=' * 50}")
        print("Prediction Results:")
        print(f"Predicted Lifespan: {prediction:.2f} hours")
        print(f"Actual Lifespan: {true_value:.2f} hours")
        print(f"Absolute Error: {error:.2f} hours")
        print(f"Error Percentage: {error_percentage:.2f}%")
        print("=" * 50)

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
print(f"\nCategorical columns: {categorical_cols}\n")

# Define target variable
target = 'Lifespan'

# Split the dataset
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing pipelines
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features = [col for col in numeric_features if col not in categorical_cols]

encoder_name = "One-Hot"
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', dtype=int, drop=None)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Create the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        max_depth=10,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Make random predictions
print("\nGenerating random predictions from test set...")
make_random_predictions(model, X_test, y_test)

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Transform the training and test data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after transformation
numeric_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()

# Combine feature names
transformed_feature_names = list(numeric_feature_names) + list(categorical_feature_names)

# Create DataFrames with transformed data
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)

# Inspect the shapes
print("\nOriginal 'X_train' shape   :", X_train.shape)
print("Transformed 'X_train' shape:", X_train_transformed_df.shape)

# Calculate metrics
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\n--- Performance of XGBoost with Pipeline ---\n")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")
print(f"MAE : {mae:.2f}")

# Perform cross-validation
scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring=scorer
)

print("\nCross-Validation Results:")
print(f"Mean CV MAE: {-cv_scores.mean():.2f}")
print(f"Standard Deviation of CV MAE: {cv_scores.std():.2f}")
