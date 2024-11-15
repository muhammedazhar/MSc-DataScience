# Import libraries
try:
    # Importing general libraries
    import glob
    import pandas as pd

    import torch

    # Importing libraries for model building
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        root_mean_squared_error,
        r2_score,
        mean_absolute_error
    )

    # Importing libraries for data preprocessing
    from scipy.stats import randint
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

def evaluate_model(y_test, y_pred, encoder_name):
    """Evaluate model performance using multiple metrics"""
    rmse = root_mean_squared_error(y_test, y_pred)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)                   # R² Score
    mae = mean_absolute_error(y_test, y_pred)       # Mean Absolute Error

    print(f'--- {encoder_name} Performance ---\n')
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.2f}")
    print(f"MAE : {mae:.2f}")

def main():
    # Initialize encoders
    ohe = OneHotEncoder()
    le = LabelEncoder()

    # Find and load the CSV file
    data_path = '../Datasets/*.csv'
    file_list = glob.glob(data_path)

    if len(file_list) != 1:
        raise FileNotFoundError("No CSV file found or multiple CSV files found in the Datasets directory.")

    # Load the dataset
    df = pd.read_csv(file_list[0])
    print(f"Loaded dataset: {file_list[0]}")

    # Define categorical columns
    target = 'Lifespan'
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    label_features = 'partType'
    onehot_features = [col for col in df.select_dtypes(include=['object']).columns if col != label_features]

    # Label Encoding
    encoder_name = "Label"
    print(f"\n=== {encoder_name} Encoding ===\n")
    label_encoded_df = df.copy()
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        label_encoded_df[col] = le.fit_transform(label_encoded_df[col])
        label_encoders[col] = le

    # Prepare data for Label Encoding
    X = label_encoded_df.drop(columns=[target])
    y = label_encoded_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate default model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Label Encoding (Default)")

    # Hyperparameter tuning with RandomizedSearchCV
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(range(10, 50, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    search_cv = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,  # For reproducibility
        n_jobs=-1,
        scoring='r2'      # Using R² score for hyperparameter tuning
    )

    search_cv.fit(X_train, y_train)
    print("\nBest parameters for Label Encoding:")
    print(search_cv.best_params_)

    # Train and evaluate model with best parameters
    best_rf_model = RandomForestRegressor(**search_cv.best_params_, random_state=42)
    best_rf_model.fit(X_train, y_train)
    y_pred = best_rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Label Encoding (Tuned)")

    # Hybrid Encoding
    encoder_name = "Hybrid"
    print(f"\n=== {encoder_name} Encoding ===\n")
    hybrid_encoded_df = df.copy()
    hybrid_encoded_df['partType'] = le.fit_transform(hybrid_encoded_df['partType'])

    onehot_features = ['microstructure', 'seedLocation', 'castType']
    encoded_array = ohe.fit_transform(hybrid_encoded_df[onehot_features])
    encoded_df = pd.DataFrame(
        encoded_array.toarray(),
        columns=ohe.get_feature_names_out(onehot_features)
    )

    hybrid_encoded_df = pd.concat(
        [hybrid_encoded_df.drop(columns=onehot_features), encoded_df],
        axis=1
    )

    # Prepare data for Hybrid Encoding
    X = hybrid_encoded_df.drop(columns=[target])
    y = hybrid_encoded_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate default model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Hybrid Encoding (Default)")

    # Hyperparameter tuning
    search_cv.fit(X_train, y_train)
    print("\nBest parameters for Hybrid Encoding:")
    print(search_cv.best_params_)

    # Train and evaluate model with best parameters
    best_rf_model = RandomForestRegressor(**search_cv.best_params_, random_state=42)
    best_rf_model.fit(X_train, y_train)
    y_pred = best_rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Hybrid Encoding (Tuned)")

    # One-Hot Encoding
    encoder_name = "One-Hot"
    print(f"\n=== {encoder_name} Encoding ===\n")
    onehot_encoded_df = df.copy()
    encoded_array = ohe.fit_transform(onehot_encoded_df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_array.toarray(),
        columns=ohe.get_feature_names_out(categorical_features)
    )

    onehot_encoded_df = pd.concat(
        [onehot_encoded_df.drop(columns=categorical_features), encoded_df],
        axis=1
    )

    # Prepare data for One-Hot Encoding
    X = onehot_encoded_df.drop(columns=[target])
    y = onehot_encoded_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate default model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "One-Hot Encoding (Default)")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    search_type = "RandomizedSearchCV"
    final_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    final_search.fit(X_train, y_train)
    print("\nBest parameters for One-Hot Encoding:")
    print(final_search.best_params_)
    # Get the best estimator and parameters
    params = search_cv.best_params_

    print(f"Best parameters found for {encoder_name} by {search_type}")
    print(f"-" * 63)
    for param, value in params.items():
        print(f"    {param}={value},")
    print("-" * 63 + "\n")
    # Train and evaluate model with best parameters
    best_rf_model = RandomForestRegressor(**final_search.best_params_, random_state=42)
    best_rf_model.fit(X_train, y_train)
    y_pred = best_rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "One-Hot Encoding (Tuned)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
