import re
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    confusion_matrix, accuracy_score, f1_score
)

def sanitize_numeric_string(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    # keep first number + first decimal point + digits after it, drop rest
    match = re.match(r'^-?\d+(\.\d+)?', s)
    return match.group() if match else None

gender_map = {
    'male': 'Male', 'm': 'Male', 'male': 'Male',
    'female': 'Female', 'f': 'Female', 'female': 'Female'
}

def sanitize_gender(val):
    if pd.isna(val):
        return val
    return gender_map.get(str(val).strip().lower(), val)

def evaluate_reg(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred))
    }

def critical_error_rate(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    hr_idx = labels.index('High_Risk')
    elig_idx = labels.index('Eligible')
    hr_total = cm[hr_idx].sum()
    hr_as_eligible = cm[hr_idx][elig_idx]
    return hr_as_eligible / hr_total if hr_total > 0 else 0, cm

def evaluate_clf(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    crit, _ = critical_error_rate(y_true, y_pred, labels)
    return {'accuracy': float(acc), 'f1_macro': float(f1), 'critical_error_rate': float(crit)}