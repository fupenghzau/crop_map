import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(data, feature_columns, label_column):
    """
    Load data and separate features and labels. Supports both DataFrame input and file path input.

    Parameters:
    - data (str or pandas.DataFrame): The input data, which can be a file path (CSV) or a DataFrame.
    - feature_columns (list of str): List of feature column names.
    - label_column (str): The name of the label column.

    Returns:
    - X (numpy.ndarray): Array of features.
    - y (numpy.ndarray): Array of labels.
    """
    if isinstance(data, str):
        # If data is a file path, read the CSV file
        data = pd.read_csv(data)

    # Extract features and labels from the DataFrame
    X = data[feature_columns].values
    y = data[label_column].values
    return X, y

def preprocess_data(X, scaler=None):
    """
    Normalize the 3D data using a standard scaler. If a scaler is not provided, 
    a new one is fit to the data.

    Parameters:
    - X (numpy.ndarray): Input feature data of shape (num_ids, num_channels, sequence_length).
    - scaler (StandardScaler, optional): Pre-existing scaler for normalization. 
      If None, a new scaler is created and fitted.

    Returns:
    - X_scaled (numpy.ndarray): The normalized feature data of the same shape.
    - scaler (StandardScaler): The scaler used for normalization.
    """
    num_ids, num_channels, sequence_length = X.shape

    # Reshape X to (num_ids * sequence_length, num_channels) for scaling
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, num_channels)

    # Fit the scaler if it's not provided
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)

    # Reshape back to the original shape (num_ids, num_channels, sequence_length)
    X_scaled = X_scaled.reshape(num_ids, sequence_length, num_channels).transpose(0, 2, 1)

    return X_scaled, scaler


def fill_missing_values(data, id_column, date_column, feature_columns):
    """
    Fill missing values in the data by averaging prior and subsequent 
    observations for each ID. The date column is used to sort the data.

    Parameters:
    - data (pandas.DataFrame): The input data containing IDs, dates, and features.
    - id_column (str): The column name representing the ID of each observation.
    - date_column (str): The column name representing the date in 'month/day/year' format.
    - feature_columns (list of str): The list of feature columns where NaNs need to be filled.

    Returns:
    - data_filled (pandas.DataFrame): DataFrame with missing values filled.
    """
    data_filled = data.copy()

    # Convert the date column to datetime, considering two-digit year format
    data_filled[date_column] = pd.to_datetime(data_filled[date_column], format='%m/%d/%Y')

    # Sort data by ID and date for proper filling
    data_filled.sort_values(by=[id_column, date_column], inplace=True)

    # Fill NaN values using forward and backward fill, then take the mean
    for feature in feature_columns:
        data_filled[feature] = (data_filled.groupby(id_column)[feature].apply(
            lambda group: group.ffill().bfill()
        )).reset_index(level=0, drop=True)
    
    # For any remaining NaNs (e.g., all values for an ID are NaN), fill with the column mean
    # the alternative is to delete nan after the filling step above
    data_filled[feature_columns] = data_filled[feature_columns].fillna(data_filled[feature_columns].mean())
    data_filled.sort_index(inplace=True)

    return data_filled

def prepare_data_for_1d_input(data, fid_column, date_column, feature_columns, label_column, num_channels, sequence_length):
    """
    Prepare data for 1D CNN input by sorting, reshaping features, and returning the prepared features and labels.

    Parameters:
    - data (pandas.DataFrame): The input data containing IDs, dates, features, and labels.
    - fid_column (str): The column name representing the ID of each observation.
    - date_column (str): The column name representing the date for sorting.
    - feature_columns (list of str): List of feature column names.
    - label_column (str): The column name representing the label.
    - num_channels (int): The number of features (channels).
    - sequence_length (int): The number of time steps (e.g., 8 months).

    Returns:
    - reshaped_X (numpy.ndarray): Reshaped features of shape (num_ids, num_channels, sequence_length).
    - reshaped_y (numpy.ndarray): Reshaped labels of shape (num_ids, sequence_length).
    """
    # Sort data by fid and date
    data_sorted = data.sort_values(by=[fid_column, date_column])

    # Extract features and labels after sorting
    X = data_sorted[feature_columns].values
    y = data_sorted[label_column].values

    # Calculate the number of unique IDs
    num_ids = X.shape[0] // sequence_length

    # Reshape features to (num_ids, sequence_length, num_channels), then transpose to (num_ids, num_channels, sequence_length)
    reshaped_X = X.reshape(num_ids, sequence_length, num_channels).transpose(0, 2, 1)

    # Reshape labels to (num_ids, sequence_length)
    reshaped_y = y.reshape(num_ids, sequence_length)

    return reshaped_X, reshaped_y

