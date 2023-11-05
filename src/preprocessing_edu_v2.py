"""
File for preprocessing educational institution data
"""

import os
from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point
from src.utils import (
    load_data, 
    add_twd97_coordinates_to_dataframe
)

PATH = f'{os.getcwd()}/data/external_data'
PATH_traindata = f'{os.getcwd()}/data/training_data.csv'

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


def preprocessing_jhs() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    jhs_data_path = os.path.join(PATH, '國中基本資料_v2.csv')
    data = load_data(jhs_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    data['學生總數'] = data['學生數7年級男'] + data['學生數7年級女'] + data['學生數8年級男'] +\
                     data['學生數8年級女'] + data['學生數8年級男'] + data['學生數8年級女']
    
    # Conserve only useful columns
    # data = data[['金融機構名稱','地址', 'lat', 'lng']]
    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數','Is_Combined', 'Is_Popular', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data


def preprocessing_es() -> pd.DataFrame:
    """
    Preprocess junior high school data
    """
    es_data_path = os.path.join(PATH, '國小基本資料_v2.csv')
    data = load_data(es_data_path)
    data = add_twd97_coordinates_to_dataframe(data)

    # sum the number of teachers and students sepreately
    data['教師總數'] = data['男專任教師'] + data['女專任教師']
    
    data['學生總數'] = 0
    for i in range(1, 7, 1):
        data['學生總數'] += data[f'{i}年級男學生數']
        data['學生總數'] += data[f'{i}年級女學生數']

    data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'Is_Popular', 'lat', 'lng', '橫坐標', '縱坐標']]

    return data


def merge_es_info(training_data:pd.DataFrame, es_data:pd.DataFrame):

    column_names = training_data.columns
    
    for row in tqdm(training_data.itertuples()):     
        unit_x = row[column_names.get_loc('橫坐標') + 1]
        unit_y = row[column_names.get_loc('縱坐標') + 1]
        
        nearest_index = find_nearest_facility(unit_x, unit_y, es_data)
        training_data['鄰近熱門國小'] = es_data['Is_Popular'].iloc[nearest_index]
    return training_data


def merge_jhs_info(training_data:pd.DataFrame, jhs_data:pd.DataFrame):

    column_names = training_data.columns
    
    for row in tqdm(training_data.itertuples()):
        unit_x = row[column_names.get_loc('橫坐標') + 1]
        unit_y = row[column_names.get_loc('縱坐標') + 1]
        unit_ID = row[column_names.get_loc('ID') + 1]
        
        nearest_index = find_nearest_facility(unit_x, unit_y, jhs_data)
        training_data['鄰近熱門國中'] = jhs_data['Is_Popular'].iloc[nearest_index]
        training_data['鄰近完全中學'] = jhs_data['Is_Combined'].iloc[nearest_index]
        
        # 確認是否成功對照
        # if(jhs_data['Is_Popular'].iloc[nearest_index]==1):
        #     print( unit_ID , unit_x, unit_y, jhs_data['學校名稱'].iloc[nearest_index])
    
    return training_data


def find_nearest_facility(unit_x: float, unit_y: float, facilities_with_dist) -> float:
    """
    Get the index of the nearest neighbor
    """
    ref_point = Point(unit_x, unit_y)

    #Get the distance between each facility and the target
    facilities_with_dist['distance'] = facilities_with_dist.apply(
        lambda row: ref_point.distance(
            Point(row['橫坐標'], row['縱坐標'])
        ),
        axis = 1
    )

    #Sort the dataframe by distance
    facilities_with_dist = facilities_with_dist.sort_values(
        by = 'distance',
        ascending=True
    )

    # #return index of the nearest neighbor
    nearest_index = facilities_with_dist.head(1).index.tolist()[0]
    return nearest_index


def create_edu_feature():

    trainingdata = load_data(PATH_traindata)
    es_processing_data = preprocessing_es()
    jhs_processing_data = preprocessing_jhs()

    training_edited = merge_es_info(trainingdata, es_processing_data)
    training_edited = merge_jhs_info(trainingdata, jhs_processing_data)

    column_to_move = training_edited['單價']
    training_edited = training_edited.drop('單價', axis=1)
    training_edited['單價'] = column_to_move

    print(training_edited.head())

    training_edited.to_csv('training_data_edited.csv', index=False)

if __name__ == "__main__":

    create_edu_feature()