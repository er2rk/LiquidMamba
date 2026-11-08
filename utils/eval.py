import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score


def evaluate(y_true, y_hat):
    """Metrics for next-day return forecasts. y_true, y_hat: arrays of returns."""
    mse = mean_squared_error(y_true, y_hat)
    #Flat days have no direction, so they are left out of the direction metrics.
    moved = y_true != 0
    up_true, up_hat = y_true[moved] > 0, y_hat[moved] > 0

    return {"mse": mse,
            "rmse": mse ** 0.5,
            "mae": mean_absolute_error(y_true, y_hat),
            "r2": 1 - mse / np.mean(y_true ** 2),   #vs. always predicting 0
            "mape": np.mean(np.abs(y_hat - y_true) / (1 + y_true)) * 100,  #on reconstructed prices
            "dir": np.mean(up_hat == up_true),
            "f1": f1_score(up_true, up_hat, average="macro", zero_division=0), #average of up and down f1.
    }