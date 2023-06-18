import pandas as pd
import numpy as np

import torch
# from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def data_sequence(data_source, train_size, val_size, depth, features=None, target=None):
    """
    data_source - list of path where the data is kept. Usually 2 sources.
    features - list of feature names to be included from data
    target - the target to be forecasted.

    """
    prediction_horizon = 1
    data1 = pd.read_csv(data_source[0], sep=' ')
    data2 = pd.read_csv(data_source[1], sep=' ')

    X_1 = np.zeros((len(data1), depth, len(features)))
    y_1 = np.zeros((len(data1), 1))  

    X_2 = np.zeros((len(data2), depth, len(features)))
    y_2 = np.zeros((len(data2), 1))  

    for i, name in enumerate(features):
        for j in range(depth):
            X_1[:, j, i] = data1[name].shift(depth - j - 1).fillna(method="bfill")
    y_1 = data1[target].shift(-prediction_horizon).fillna(method='ffill')

    for i, name in enumerate(features):
        for j in range(depth):
            X_2[:, j, i] = data2[name].shift(depth - j - 1).fillna(method="bfill")
    y_2 = data2[target].shift(-prediction_horizon).fillna(method='ffill')

    X_1 = X_1[depth:-prediction_horizon]
    y_1 = y_1[depth:-prediction_horizon]

    X_val = X_2[train_size - len(data1):train_size - len(data1) + val_size]
    y_val = y_2[train_size - len(data1):train_size - len(data1) + val_size]

    X_test = X_2[train_size - len(data1) + val_size:]
    y_test = y_2[train_size - len(data1) + val_size:]

    X_2 = X_2[:train_size - len(data1)]
    y_2 = y_2[:train_size - len(data1)]

    X_train = np.concatenate([X_1, X_2], axis=0)
    y_train = np.concatenate([y_1, y_2], axis=0)  

    return (X_train, X_val, X_test), (y_train, y_val, y_test), (X_train.shape, X_val.shape)


def preprocess(X, y, bias):
    """
    X - Tuple of training, validation test feature data.
    y - Tuple of training, validation test targe data.
    """
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y
    
    # Normalizing values
    X_train_min, y_train_min = X_train.min(axis=0), y_train.min(axis=0)
    X_train_max, y_train_max = X_train.max(axis=0), y_train.max(axis=0)

    X_train = (X_train - X_train_min)/(X_train_max - X_train_min + bias)
    X_val = (X_val - X_train_min)/(X_train_max - X_train_min + bias)
    X_test = (X_test - X_train_min)/(X_train_max - X_train_min + bias)

    y_train = (y_train - y_train_min)/(y_train_max - y_train_min + bias)
    y_val = (y_val - y_train_min)/(y_train_max - y_train_min + bias)
    y_test = (y_test - y_train_min)/(y_train_max - y_train_min + bias)

    meta_data = [X_train_max, X_train_min, y_train_max, y_train_min]
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), meta_data


def dataset_generator(X, y, batch_size):
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    X_train_t = torch.Tensor(X_train)
    X_val_t = torch.Tensor(X_val)
    X_test_t = torch.Tensor(X_test)
    y_train_t = torch.Tensor(y_train)
    y_val_t = torch.Tensor(y_val.values)
    y_test_t = torch.Tensor(y_test.values)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(pd.__version__)
    print(np.__version__)
    print(torch.__version__)