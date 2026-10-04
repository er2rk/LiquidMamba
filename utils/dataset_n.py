import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_ends(n_rows, window, n_val, n_test):
    """
    Row indices of each window's last day, split chronologically.
    The last row has no target, so no window ends there. 
    The last window of train and of val is dropped: its target is the next split's first day, which that split also sees as an input.
    """
    ends = np.arange(window - 1, n_rows - 1)
    train = ends[: -(n_val + n_test)]
    val = ends[-(n_val + n_test): -n_test]
    test = ends[-n_test:]
    return train[:-1], val[:-1], test


class Window(Dataset):
    """Serves (x, timespans, y) for the windows ending at the given row indices."""

    def __init__(self, features, timespans, targets, ends, window):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.timespans = torch.as_tensor(timespans, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.ends = ends
        self.window = window
        self.n_features = self.features.shape[1]

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        end = int(self.ends[i])
        rows = slice(end - self.window + 1, end + 1)
        return self.features[rows], self.timespans[rows], self.targets[end]


def build_datasets(dataset, window=64, n_val=300, n_test=300):
    """Return train, val and test Window, plus the DataFrame for evaluation."""

    #Read csv into df.
    df = pd.read_csv(f"../data/{dataset}.csv")
    #Turn col names into lowercase just incase.
    df.columns = df.columns.str.lower()
    #Turn date column into datetime format.
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    #All columns in dataset except date and close are used as features. Should be numerical.
    feature_columns = [c for c in df.columns if c not in ("date", "close")]

    #Target for day t is the return from day t to day t+1. 
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    #Timespan is calculated to be fed into our model. As i mentioned in the conventions, subject to change depending on order books.
    df["timespan"] = df["date"].diff().dt.days.fillna(1.0)

    train_ends, val_ends, test_ends = split_ends(len(df), window, n_val, n_test)

    #Feature standardization
    train_rows = df.loc[: train_ends[-1], feature_columns]
    mean, std = train_rows.mean(), train_rows.std().replace(0, 1)
    features = (df[feature_columns] - mean) / std

    arrays = (features.values, df["timespan"].values, df["target"].values)
    train, val, test = (Window(*arrays, ends, window) for ends in (train_ends, val_ends, test_ends))
    return train, val, test, df

