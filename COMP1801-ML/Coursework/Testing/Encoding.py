# Import libraries
try:
    # Importing general libraries
    import glob
    import pandas as pd

    # Importing libraries for data visualization
    import matplotlib.pyplot as plt
    import numpy as np

    # Importing libraries for model building
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

except Exception as e:
    print(f"Error : {e}")

# Define the categorical columns
categorical_cols = ['partType', 'microstructure', 'seedLocation', 'castType']

# Function to train and evaluate the model
def train_and_evaluate(X, y, encoding_type):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor model with the best parameters
    rf_model = RandomForestRegressor(
        max_depth=15,
        n_estimators=387,
        random_state=42
    )

    # Fit the model to the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model using various metrics
    rmse = root_mean_squared_error(y_test, y_pred)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R² Score
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    msle = mean_squared_log_error(y_test, y_pred)  # Mean Squared Log Error

    # Print the results
    print(f"--- {encoding_type} Encoding Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Log Error (MSLE): {msle:.2f}\n")

# Find the CSV file in the Datasets directory
data_path = '../Datasets/*.csv'
file_list = glob.glob(data_path)

for file in file_list:
    print(f"Found file: {file}")

# Ensure there is exactly one file
if len(file_list) == 1:
    # Load the dataset
    df = pd.read_csv(file_list[0])
    print(f"Loaded dataset: {file_list[0]}\n")
else:
    raise FileNotFoundError("No CSV file found or multiple CSV files found in the Datasets directory.")

# One-Hot Encoding
df_onehot_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
X_onehot = df_onehot_encoded.drop(columns=['Lifespan'])
y = df_onehot_encoded['Lifespan']
train_and_evaluate(X_onehot, y, "One-Hot")

# Label Encoding
df_label_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_label_encoded[col] = le.fit_transform(df_label_encoded[col])
    label_encoders[col] = le

X_label = df_label_encoded.drop(columns=['Lifespan'])
train_and_evaluate(X_label, y, "Label")
