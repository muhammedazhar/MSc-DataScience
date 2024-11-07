# Import libraries
try:
    # Importing general libraries
    import glob
    import pandas as pd
    import numpy as np

    # Model building libraries
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

    # Local imports
    from Helper import *

except Exception as e:
    print(f"Error : {e}")

# Define the categorical columns
categorical_cols = ['partType', 'microstructure', 'seedLocation', 'castType']
# Function to train and evaluate the model
def train_and_evaluate(X, y, encoding_type, n_iterations=10):
    # Lists to store metrics for each iteration
    rmse_scores = []
    r2_scores = []
    mae_scores = []
    msle_scores = []

    for i in range(n_iterations):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Initialize and train the model
        rf_model = RandomForestRegressor(
            max_depth=15,
            n_estimators=387,
            n_jobs=-1,  # This will still use all available cores
            verbose=0   # This will suppress the parallel processing messages
        )
        rf_model.fit(X_train, y_train)
        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate and store metrics
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        msle_scores.append(mean_squared_log_error(y_test, y_pred))

    # Print average results
    print(f"--- {encoding_type} Encoding Results (Averaged over {n_iterations} iterations) ---")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
    print(f"Average R²: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")
    print(f"Average MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"Average MSLE: {np.mean(msle_scores):.2f} ± {np.std(msle_scores):.2f}\n")

# One-Hot Encoding
df_onehot_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
X_onehot = df_onehot_encoded.drop(columns=['Lifespan'])
y = df_onehot_encoded['Lifespan']
train_and_evaluate(X_onehot, y, "One-Hot")
print(f"Shape of One-Hot Encoded Data: {X_onehot.shape}\n")

# Label Encoding
df_label_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_label_encoded[col] = le.fit_transform(df_label_encoded[col])
    label_encoders[col] = le

X_label = df_label_encoded.drop(columns=['Lifespan'])
train_and_evaluate(X_label, y, "Label")
print(f"Shape of Label Encoded Data: {X_label.shape}")
