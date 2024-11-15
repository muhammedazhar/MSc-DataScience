try:
    import glob
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        root_mean_squared_error,
        r2_score,
        mean_absolute_error
    )
    from sklearn.utils import shuffle
    import torch
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

def load_data(data_path='../Datasets/*.csv'):
    """
    Load data from the specified CSV file.

    Args:
        data_path (str): Glob pattern to locate the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If no CSV file or multiple CSV files are found.
    """
    file_list = glob.glob(data_path)
    if len(file_list) != 1:
        raise FileNotFoundError("No CSV file found or multiple CSV files found in the Datasets directory.")
    return pd.read_csv(file_list[0])

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate the model using RMSE, R², and MAE metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    metrics = {
        "RMSE": root_mean_squared_error(y_true, y_pred),  # Root Mean Squared Error
        "R²": r2_score(y_true, y_pred),                   # R-squared
        "MAE": mean_absolute_error(y_true, y_pred)        # Mean Absolute Error
    }

    print(f"\n--- {model_name} Results ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    return metrics

def get_base_model():
    """
    Get the base Random Forest Regressor model with predefined parameters.

    Returns:
        RandomForestRegressor: The base model.
    """
    return RandomForestRegressor(
        max_depth=15,      # Found through hyperparameter tuning
        n_estimators=387,  # Found through hyperparameter tuning
        random_state=42,   # Set random seed for reproducibility
        n_jobs=-1          # Utilize all available CPU cores
    )

def iterative_evaluation(X, y, model_creator, encoding_type, n_iterations):
    """
    Perform iterative evaluation over multiple splits to get average performance.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target variable.
        model_creator (callable): Function to create and return a model.
        encoding_type (str): Type of encoding used.
        n_iterations (int): Number of iterations.

    Returns:
        dict: Dictionary containing average metrics over iterations.
    """
    metrics_lists = {metric: [] for metric in ["RMSE", "R²", "MAE"]}

    for _ in range(n_iterations):
        # Shuffle data before splitting
        X_shuffled, y_shuffled = shuffle(X, y, random_state=None)
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffled, y_shuffled, test_size=0.2, random_state=None
        )
        model = model_creator()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        iteration_metrics = evaluate_model(y_test, y_pred, f"{encoding_type} Iteration")
        for metric, value in iteration_metrics.items():
            metrics_lists[metric].append(value)

    print(f"\n--- {encoding_type} Encoding Results (Averaged over {n_iterations} iterations) ---")
    avg_metrics = {}
    print(f"-"*75)
    for metric, values in metrics_lists.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"Average {metric}: {avg:.2f} ± {std:.2f}")
        avg_metrics[metric] = avg

    return avg_metrics

def determine_best_encoding(results):
    """
    Determine the best encoding method based on weighted metrics.

    Args:
        results (dict): Dictionary containing metrics for each encoding method.

    Returns:
        str, dict: Best encoding method and its metrics.
    """
    # Compare based on multiple metrics with weights
    weights = {
        "RMSE": -0.4,  # Negative because lower is better
        "R²": 0.4,     # Positive because higher is better
        "MAE": -0.2    # Negative because lower is better
    }

    scores = {}
    for encoding, metrics in results.items():
        score = sum(metrics[metric] * weights[metric] for metric in weights)
        scores[encoding] = score

    best_encoding = max(scores.items(), key=lambda x: x[1])[0]
    print(f"<<< Best Encoding Method >>>")
    print(f"Best method: {best_encoding}\n")
    print(f"="*55)
    print(f"Performance metrics for selected best encoding: {best_encoding}")
    for metric, value in results[best_encoding].items():
        print(f"{metric}: {value:.2f}")
    print(f"="*55)

    print("\nDetailed comparison of methods (weighted scores):")
    for encoding, score in scores.items():
        print(f"{encoding}: {score:.2f}")

    return best_encoding, results[best_encoding]

def main():
    # Load data
    df = load_data()

    # Define encoders
    ohe = OneHotEncoder(sparse_output=False, dtype=int, drop=None)
    le = LabelEncoder()

    # Define feature categories
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    onehot_features = ['microstructure', 'seedLocation', 'castType']
    label_features = 'partType'
    target = 'Lifespan'
    n_iterations = 10

    # Base model creator
    def base_model_creator():
        return get_base_model()

    results = {}

    # 1. One-Hot Encoding Approach
    print("\nExecuting One-Hot Encoding Approach...")
    df_onehot = df.copy()
    # Fit and transform the categorical columns
    encoded_data = ohe.fit_transform(df_onehot[categorical_features])

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out(categorical_features),
        index=df_onehot.index  # Ensure the index matches for proper concatenation
    )

    # Concatenate the encoded columns with the rest of the DataFrame
    df_onehot = pd.concat([df_onehot.drop(columns=categorical_features), encoded_df], axis=1)
    X_onehot = df_onehot.drop(columns=[target])
    y = df_onehot[target]
    model_creator_onehot = base_model_creator
    results['One-Hot'] = iterative_evaluation(X_onehot, y, model_creator_onehot, "One-Hot", n_iterations)
    print(f"Shape of One-Hot Encoded Data: {X_onehot.shape}")

    # 2. Label Encoding Approach
    print("\nExecuting Label Encoding Approach...")
    df_label = df.copy()
    for col in categorical_features:
        df_label[col] = le.fit_transform(df_label[col])
    X_label = df_label.drop(columns=[target])
    model_creator_label = base_model_creator
    results['Label'] = iterative_evaluation(X_label, y, model_creator_label, "Label", n_iterations)
    print(f"Shape of Label Encoded Data: {X_label.shape}")

    # 3. Hybrid Approach
    print("\nExecuting Hybrid Encoding Approach...")
    df_hybrid = df.copy()
    df_hybrid[label_features] = le.fit_transform(df_hybrid[label_features])
    # Reshape the data to handle multiple categorical columns
    encoded_data = ohe.fit_transform(df_hybrid[onehot_features].values)

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out(onehot_features)
    )

    # Combine with non-categorical columns if needed
    df_hybrid = pd.concat([df_hybrid.drop(columns=onehot_features), encoded_df], axis=1)
    X_hybrid = df_hybrid.drop(columns=[target])
    model_creator_hybrid = base_model_creator
    results['Hybrid'] = iterative_evaluation(X_hybrid, y, model_creator_hybrid, "Hybrid", n_iterations)
    print(f"Shape of Hybrid Encoded Data: {X_hybrid.shape}")

    # Determine and display the best encoding method with its metrics
    print("\nDetermining the best encoding method...\n")
    best_encoding, best_metrics = determine_best_encoding(results)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
