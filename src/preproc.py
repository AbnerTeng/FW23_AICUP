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

    ## TODO: one-hot encoding / Label Encoding
    """
    import pandas as pd
    df = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue']
    })
    df_encoded = pd.get_dummies(df, columns=['Color'])
    
    -------------------------------------------------
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Category_Label'] = le.fit_transform(df['Category'])
    """
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
        print(self.data.shape)
        return self.data


if __name__ == "__main__":
    preproc = PreProc(
        f'{os.getcwd()}/data/training_data.csv'
    )
    preproc.main()
