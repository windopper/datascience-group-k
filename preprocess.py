import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

scaler_types = ['standard', 'minmax']
encoder_types = ['onehot', 'label']


def preprocessing(scaler='standard', encoder='onehot'):
    """
    Preprocess the data by removing unnecessary columns and renaming them.
    """
    if scaler not in scaler_types:
        raise ValueError(
            f"Invalid scaler: {scaler}, please choose from {scaler_types}")
    if encoder not in encoder_types:
        raise ValueError(
            f"Invalid encoder: {encoder}, please choose from {encoder_types}")

    path = kagglehub.dataset_download(
        "teejmahal20/airline-passenger-satisfaction")
    train_df = pd.read_csv(path + "/train.csv")
    test_df = pd.read_csv(path + "/test.csv")

    # Drop unnecessary columns
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Drop duplicate columns
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    # log transformation
    # log
    log_cols = ['Arrival Delay in Minutes', 'Departure Delay in Minutes']
    train_df[log_cols] = np.log1p(train_df[log_cols])
    test_df[log_cols] = np.log1p(test_df[log_cols])

    # scaling
    scaling_cols = ['Age', 'Flight Distance',
                    'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    if scaler == 'standard':
        standard_scaler = StandardScaler()
        train_df[scaling_cols] = standard_scaler.fit_transform(
            train_df[scaling_cols])
        test_df[scaling_cols] = standard_scaler.transform(
            test_df[scaling_cols])
    elif scaler == 'minmax':
        minmax_scaler = MinMaxScaler()
        train_df[scaling_cols] = minmax_scaler.fit_transform(
            train_df[scaling_cols])
        test_df[scaling_cols] = minmax_scaler.transform(test_df[scaling_cols])

    unused_cols = ['Unnamed: 0', 'id']
    target_col = 'satisfaction'
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    # encoding
    if encoder == 'onehot':
        train_df = pd.get_dummies(train_df, columns=categorical_cols)
        test_df = pd.get_dummies(test_df, columns=categorical_cols)
    elif encoder == 'label':
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            train_df[col] = label_encoder.fit_transform(train_df[col])
            test_df[col] = label_encoder.transform(test_df[col])
    train_df[target_col] = train_df[target_col].replace(
        {'neutral or dissatisfied': 0, 'satisfied': 1})
    test_df[target_col] = test_df[target_col].replace(
        {'neutral or dissatisfied': 0, 'satisfied': 1})

    train_df = train_df.astype(
        {col: 'int' for col in train_df.columns if train_df[col].dtype == 'bool'})
    test_df = test_df.astype(
        {col: 'int' for col in test_df.columns if test_df[col].dtype == 'bool'})

    train_X = train_df.drop(unused_cols + [target_col], axis=1)
    train_y = train_df[target_col]
    test_X = test_df.drop(unused_cols + [target_col], axis=1)
    test_y = test_df[target_col]

    return train_X, train_y, test_X, test_y
