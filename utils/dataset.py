from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import numpy as np


#Changes made to use with the crude oil dataset. Will try to generalize this later.
class StockPrice(Dataset):
    def __init__(self, root, data_file, n_test):
        self.root = root
        self.data_file = data_file
        self.n_test = n_test

        self.data = pd.read_csv(os.path.join(self.root, self.data_file) +'.csv').dropna().reset_index(drop=True)
        self.data['Date'] = pd.to_datetime(self.data['Date']) 
        prev_close = self.data['Close'].shift(1)

        feats = pd.DataFrame({
            'open': self.data['Open'] / prev_close - 1,
            'high': self.data['High'] / prev_close - 1,
            'low': self.data['Low'] / prev_close - 1,
            'close': self.data['Close'] / prev_close - 1,
            'volume': np.log1p(self.data['Volume']).diff(),
        })

        self.ratechg = (self.data['Close'].shift(-1) / self.data['Close'] - 1).values[1:]
        self.close = self.data['Close'].values[1:]
        self.dat = feats.values[1:]
        self.data = self.data.iloc[1:].reset_index(drop=True)

    def get_data(self):
        x_train, x_test = self.dat[:-self.n_test, :], self.dat[-self.n_test:, :]
        y_train = self.ratechg[:-self.n_test]

        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_train = torch.from_numpy(y_train).float()
                
        return x_train, y_train, x_test, self.close, self.data
        