import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

METRIC_NAMES = ["MSE", "RMSE", "MAE", "R2", "MAPE", "DIR"]

def evaluation_metric(y_test, y_hat):
    """Metrics for next-day return forecasts. y_test, y_hat: arrays of returns."""
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = 1 - MSE / np.mean(y_test ** 2)                           # vs. always predicting 0
    MAPE = np.mean(np.abs(y_hat - y_test) / (1 + y_test)) * 100   # on reconstructed prices
    #Flat days have no direction, so they are left out.
    moved = y_test != 0
    DIR = np.mean(np.sign(y_hat[moved]) == np.sign(y_test[moved])) # directional accuracy

    return MSE, RMSE, MAE, R2, MAPE, DIR