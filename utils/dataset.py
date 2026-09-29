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
        self.n_val = n_test
        self.n_test = n_test

        #Dropping rows with missing values for now. Easier development.
        self.data = pd.read_csv(os.path.join(self.root, self.data_file) +'.csv').dropna().reset_index(drop=True)
        self.data['Date'] = pd.to_datetime(self.data['Date']) 
        prev_close = self.data['Close'].shift(1)

        #Features, the ratio of OHLC to the previous day's closing price. V, change from day to day in log.
        feats = pd.DataFrame({
            'open': self.data['Open'] / prev_close - 1,
            'high': self.data['High'] / prev_close - 1,
            'low': self.data['Low'] / prev_close - 1,
            'close': self.data['Close'] / prev_close - 1,
            'volume': np.log1p(self.data['Volume']).diff(),
        })

        #Label, rate of change predicted for the next day.
        self.ratechg = (self.data['Close'].shift(-1) / self.data['Close'] - 1).values[1:]

        #Dropping the first rows as there's no previous close.
        self.close = self.data['Close'].values[1:]
        self.dat = feats.values[1:]
        self.data = self.data.iloc[1:].reset_index(drop=True)

    def get_data(self):
        #The row count reserved for validation and testing.
        n = self.n_val + self.n_test 
        #Split features and labels by time.
        x_train, x_val, x_test = self.dat[:-n], self.dat[-n:-self.n_test], self.dat[-self.n_test:]
        y_train, y_val = self.ratechg[:-n], self.ratechg[-n:-self.n_test]

        #Standardization on the training set.
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train, x_val, x_test = [(x - mean) / std for x in (x_train, x_val, x_test)]

        x_train, x_val, x_test = [torch.from_numpy(x).float() for x in (x_train, x_val, x_test)]
        y_train, y_val = torch.from_numpy(y_train).float(), torch.from_numpy(y_val).float()

        return x_train, y_train, x_val, y_val, x_test, self.close, self.data
        