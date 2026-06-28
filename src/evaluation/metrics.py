import numpy as np

def calculate_wmae(y_true, y_pred, is_holiday):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    is_holiday = np.array(is_holiday)
    
    weights = np.where(is_holiday == 1, 5, 1)
    
    absolute_errors = np.abs(y_true - y_pred)
    return np.sum(weights * absolute_errors) / np.sum(weights)
