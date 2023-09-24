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
import numpy as np
import pandas as pd
from .utils import load_data, one_hot_encoding

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


    def categorical_transformation(self) -> None:
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
    ## TODO: jaccard coefficient
    """
    set_A = {1, 2, 3, 4}
    set_B = {3, 4, 5, 6}
    set_C = {5, 6, 7, 8}
    
    # Calculate pairwise Jaccard coefficients
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
    
    jaccard_AB = jaccard_similarity(set_A, set_B)
    jaccard_AC = jaccard_similarity(set_A, set_C)
    jaccard_BC = jaccard_similarity(set_B, set_C)
    
    # Calculate overall similarity (average)
    overall_similarity = (jaccard_AB + jaccard_AC + jaccard_BC) / 3
    
    print("Jaccard(A, B):", jaccard_AB)
    print("Jaccard(A, C):", jaccard_AC)
    print("Jaccard(B, C):", jaccard_BC)
    print("Overall Similarity:", overall_similarity)
    """


    def main(self) -> pd.DataFrame:
        """
        Main execution function
        """
        self.drop_columns()
        self.categorical_transformation()
        print(self.data.info())
        print(self.data.head())
        return self.data


if __name__ == "__main__":
    preproc = PreProc(
        f'{os.getcwd()}/data/training_data.csv'
    )
    preproc.main()
