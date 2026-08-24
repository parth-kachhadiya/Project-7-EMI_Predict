import pandas as pd
import re

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