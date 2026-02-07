import pandas as pd

def feature_engineering(X):
    """
    Applies feature engineering transformations.

    - Drops the 'Area code' column if present.
    - Creates 'Avg_day_call_duration' by dividing 'Total day minutes' by 'Total day calls'.
    - Creates 'Service_calls_per_length' by dividing 'Customer service calls' by 'Account length'.

    Parameters:
    X (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Transformed dataset.
    """
    X = pd.DataFrame(X, columns=X.columns)

    # Drop 'Area code' column if present
    if 'Area code' in X.columns:
        X = X.drop(columns=['Area code'])

    # Create new features
    if 'Total day minutes' in X.columns and 'Total day calls' in X.columns:
        X["Avg_day_call_duration"] = X["Total day minutes"] / X["Total day calls"]
        X["Avg_day_call_duration"].replace([float('inf'), -float('inf')], 0, inplace=True)

    if 'Customer service calls' in X.columns and 'Account length' in X.columns:
        X["Service_calls_per_length"] = X["Customer service calls"] / X["Account length"]
        X["Service_calls_per_length"].replace([float('inf'), -float('inf')], 0, inplace=True)

    return X
