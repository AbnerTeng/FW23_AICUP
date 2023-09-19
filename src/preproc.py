"""
Data preprocessing module for the housing price prediction
Authors: 
- Yu-Chen, Den 
- Guan-Yu, Lin
- Jin-Jui, Chen
- Meng-Chun, Yu
- Tzu-Hao, Liu  
"""
import os
import warnings
import numpy as np
import pandas as pd
from .utils import load_data

warnings.filterwarnings('ignore')

class PreProc:
    """
    Data preprocessing class
    """
    def __init__(self, data_path: str) -> None:
        self.data = load_data(data_path)


    def drop_columns(self) -> None:
        """
        drop useless columns
        """
        self.data.drop(['備註'], axis = 1, inplace = True)


    def main(self) -> pd.DataFrame:
        """
        Main execution function
        """
        self.drop_columns()
        print(self.data.shape)
        return self.data


if __name__ == "__main__":
    preproc = PreProc(
        f'{os.getcwd()}/data/training_data.csv'
    )
    preproc.main()
