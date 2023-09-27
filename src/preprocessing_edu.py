"""
File for preprocessing educational institution data
"""

import os
import re
import pandas as pd
from .utils import load_data
from .add_coordinates import add_twd97_coordinates_to_dataframe

PATH = f'{os.getcwd()}/data/external_data'

def preprocess_univ() -> pd.DataFrame:
    """
    Preprocess university data
    """
    univ_data_path = os.path.join(PATH, '大學基本資料.csv')
    data = load_data(univ_data_path)

    # sum the number of students  and conserve only useful columns
    data = data.groupby('學校名稱', as_index=False).agg({ '縣市名稱': 'first', '縣市名稱': 'first', '總計': 'sum', 'lat': 'first', 'lng': 'first'})
    data = data.rename(columns={'總計': '學生總數'})
    data['縣市名稱'] = data['縣市名稱'].str.replace(r'\d| ', '', regex=True)
    data = add_twd97_coordinates_to_dataframe(data)

    return data


def preprocessing_shs() -> pd.DataFrame:
    """
    Preprocess senior high school data
    """
    shs_data_path = os.path.join(PATH, '高中基本資料.csv')
    data = load_data(shs_data_path)   
    data = add_twd97_coordinates_to_dataframe(data)
    
    # sum the number of teachers and students sepreately
    data['教師總數'] = data['專任教師數男'] + data['專任教師數女'] + data['兼任教師數男'] + data['兼任教師數女']
    data['學生總數'] = data['學生數男'] + data['學生數女']

    # Conserve only useful columns
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]
    
    return data


def preprocessing_hjs() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    jhs_data_path = os.path.join(PATH, '國中基本資料.csv')
    data = load_data(jhs_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    data['學生總數'] = data['學生數7年級男'] + data['學生數7年級女'] + data['學生數8年級男'] +\
                     data['學生數8年級女'] + data['學生數8年級男'] + data['學生數8年級女']
    
    # Conserve only useful columns
    # data = data[['金融機構名稱','地址', 'lat', 'lng']]
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data


def preprocessing_es() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    es_data_path = os.path.join(PATH, '國小基本資料.csv')
    data = load_data(es_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    
    data['學生總數'] = 0
    for i in range(1, 7, 1):
        data['學生總數'] += data[f'{i}年級男學生數']
        data['學生總數'] += data[f'{i}年級女學生數']

    # Conserve only useful columns
    # data = data[['金融機構名稱','地址', 'lat', 'lng']]
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data
 

if __name__ == "__main__":
    es_processing_data = preprocessing_es()
    
    # print info of data
    print(es_processing_data.info())
    
    #print no. of unique categories
    print(es_processing_data.nunique())

    print(es_processing_data.head(5))

