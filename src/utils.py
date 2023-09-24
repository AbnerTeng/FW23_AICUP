"""
useful utils
"""
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    load .csv files
    """
    data = pd.read_csv(path, encoding = 'utf-8')
    return data


def one_hot_encoding(data: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    One-hot-encoding function
    """
    data_encoded = pd.get_dummies(data, columns = cat_cols)
    return data_encoded
    