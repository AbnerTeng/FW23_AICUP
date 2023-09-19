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