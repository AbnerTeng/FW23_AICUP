"""
Data preprocessing module for the housing price prediction
Authors: 
- Yu-Chen, Den 
- Guan-Yu, Lin
- Chin-Jui, Chen
- Meng-Chen, Yu
- Tzu-Hao, Liu  
"""
import os
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .utils.data_utils import (
    load_data,
    one_hot_encoding
)

warnings.filterwarnings('ignore')

class PreProc:
    """
    Data preprocessing class
    """
    def __init__(self, data_path: str, dims=0) -> None:
        self.data = load_data(data_path)
        self.model =  RandomForestRegressor(random_state = 0)
        self.dims = dims


    def drop_columns(self) -> None:
        """
        drop useless columns
        """
        self.data.drop(['備註'], axis = 1, inplace = True)


    def categorical_transformation(self) -> pd.DataFrame:
        """
        categorical data transformation
        
        ---------------
        Columns need to be transformed:
        1. 縣市
        2. 使用分區
        3. 主要用途
        4. 主要建材
        5. 建物型態
        """
        self.data = one_hot_encoding(
            self.data,
            ['縣市', '使用分區', '主要用途', '主要建材', '建物型態']
        )


    def main(self) -> tuple:
        """
        Main execution function
        """
        self.drop_columns()
        self.categorical_transformation()
        self.data.drop(
            columns = ['鄉鎮市區', '路名', 'ID'],
            inplace = True
        )


if __name__ == "__main__":
    preproc = PreProc(
        f'{os.getcwd()}/data/training_data.csv'
    )
    preproc.main()
