"""
File for preprocessing educational institution data

TODO: Convert coordinate system to TWD97
"""

import os
import re
import pandas as pd
from .utils import load_data
from .add_coordinates import add_twd97_coordinates_to_dataframe

def preprocess_univ() -> pd.DataFrame:
    """
    Preprocess university data
    """
    univ_data_path = os.path.join('data', 'external_data')
    univ_data_path = os.path.join(univ_data_path, '大學基本資料.csv')

    data = load_data(univ_data_path)

    # sum the number of students  and conserve only useful columns
    data = data.groupby('學校名稱', as_index=False).agg({ '縣市名稱': 'first', '總計': 'sum', 'lat': 'first', 'lng': 'first'})
    data = data.rename(columns={'總計': '學生總數'})
    data['縣市名稱'] = data['縣市名稱'].str.replace(r'\d| ', '', regex=True)
    data = add_twd97_coordinates_to_dataframe(data)

    return data


def preprocessing_SHS() -> pd.DataFrame:
    """
    Preprocess senior high school data
    """
    SHS_data_path = os.path.join('data', 'external_data')
    SHS_data_path = os.path.join(SHS_data_path, '高中基本資料.csv')
    
    data = load_data(SHS_data_path)   
    data = add_twd97_coordinates_to_dataframe(data)
    
    # sum the number of teachers and students sepreately
    data['教師總數'] = data['專任教師數男'] + data['專任教師數女'] + data['兼任教師數男'] + data['兼任教師數女']
    data['學生總數'] = data['學生數男'] + data['學生數女']

    # Conserve only useful columns
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]
    
    return data


def preprocessing_JHS() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    JHS_data_path = os.path.join('data', 'external_data')
    JHS_data_path = os.path.join(JHS_data_path, '國中基本資料.csv')

    data = load_data(JHS_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    data['學生總數'] = data['學生數7年級男'] + data['學生數7年級女'] + data['學生數8年級男'] +\
                     data['學生數8年級女'] + data['學生數8年級男'] + data['學生數8年級女']
    
    # Conserve only useful columns
    # data = data[['金融機構名稱','地址', 'lat', 'lng']]
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data


def preprocessing_ES() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    ES_data_path = os.path.join('data', 'external_data')
    ES_data_path = os.path.join(ES_data_path, '國小基本資料.csv')

    data = load_data(ES_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    data['學生總數'] = data['1年級男學生數'] + data['2年級男學生數'] + data['3年級男學生數'] +\
                        data['4年級男學生數'] + data['5年級男學生數'] + data['6年級男學生數'] +\
                        data['1年級女學生數'] + data['2年級女學生數'] + data['3年級女學生數'] +\
                        data['4年級女學生數'] + data['5年級女學生數'] + data['6年級女學生數']   

    # Conserve only useful columns
    # data = data[['金融機構名稱','地址', 'lat', 'lng']]
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data
 

if __name__ == "__main__":
    univ_processing_data = preprocess_univ()
    
    # print info of data
    print(univ_processing_data.info())
    
    #print no. of unique categories
    print(univ_processing_data.nunique())

    print(univ_processing_data.head(5))

