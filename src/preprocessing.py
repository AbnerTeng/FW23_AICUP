"""
File for preprocessing medical istitution, postoffice, and financial institution data

TODO: Convert coordinate system to TWD97
"""

import os
import pandas as pd
from .utils import load_data

def preprocess_medical() -> pd.DataFrame:
    """
    Preprocess medical institution data
    """
    medical_data_path = os.path.join('data', 'external_data')
    medical_data_path = os.path.join(medical_data_path, '醫療機構基本資料.csv')

    data = load_data(medical_data_path)

    # Conserve only useful columns
    data = data[['權屬別', '型態別', '縣市鄉鎮', 'lat', 'lng']]

    return data

def preprocessing_post() -> pd.DataFrame:
    """
    Preprocess post office data
    """
    post_data_path = os.path.join('data', 'external_data')
    post_data_path = os.path.join(post_data_path, '郵局據點資料.csv')

    data = load_data(post_data_path)

    # Conserve only useful columns
    data = data[['局址', 'lat', 'lng']]

    return data

def preprocessing_financial() -> pd.DataFrame:
    """
    Preprocess financial institution data
    """
    financial_data_path = os.path.join('data', 'external_data')
    financial_data_path = os.path.join(financial_data_path, '金融機構基本資料.csv')

    data = load_data(financial_data_path)

    # Conserve only useful columns
    data = data[['金融機構名稱','地址', 'lat', 'lng']]

    return data

if __name__ == "__main__":
    print(preprocessing_financial().head())
