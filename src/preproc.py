"""
Data preprocessing module for the housing price prediction
Authors: 
- Yu-Chen, Den 
- Guan-Yu, Lin
- Chin-Jui, Chen
- Meng-Chen, Yu
- Tzu-Hao, Liu  
"""
import warnings
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .utils.data_utils import load_data
from .encoder import BetaEncoder

warnings.filterwarnings('ignore')

class PreProc:
    """
    Data preprocessing class
    """
    def __init__(
            self,
            raw_data_path: str,
            feat_data_path: str,
            target_path: str,
            _type: str
        ) -> None:
        self.raw_data = load_data(raw_data_path)
        self.feat_data = load_data(feat_data_path)
        self.target_data = load_data(target_path)
        self.raw_data['縣市_鄉鎮市區'] = self.raw_data['縣市'] + '_' + self.raw_data['鄉鎮市區']
        self.type = _type


    def select_features(
            self, feat_cols: List[str], cat_cols: List[str]
        ) -> Union[Tuple, pd.DataFrame]:
        """
        Select features based on the importance
        """
        selected_features = self.feat_data[feat_cols]

        if "單價" in self.raw_data.columns:
            cat_x = self.raw_data[cat_cols + ['單價']]
        else:
            cat_x = self.raw_data[cat_cols]

        x_data = pd.concat([selected_features, cat_x], axis=1)

        if self.type == 'train':
            y_data = np.log(self.target_data)
            return x_data, y_data
        else:
            return x_data


    def encode_cat_features(
        self,
        cat_cols: List[str],
        train_x: pd.DataFrame,
        test_x: pd.DataFrame,
        private_x: pd.DataFrame,
        train_y: pd.Series
    ) -> Tuple:
        """
        Label encoding + Beta encoding
        """
        for col in tqdm(cat_cols):
            label_encoder = LabelEncoder()
            tmp = np.concatenate([train_x[col], test_x[col]])
            label_encoder.fit(
                np.concatenate([tmp, private_x[col]])
            )
            train_x[col] = label_encoder.transform(train_x[col])
            test_x[col] = label_encoder.transform(test_x[col])
            private_x[col] = label_encoder.transform(private_x[col])

        x_tr, x_vl, y_tr, y_vl = train_test_split(
            train_x, train_y, test_size=0.2, random_state=42
        )
        x_tr.reset_index(drop=True, inplace=True)
        x_vl.reset_index(drop=True, inplace=True)
        y_tr.reset_index(drop=True, inplace=True)
        y_vl.reset_index(drop=True, inplace=True)
        y_vl = np.exp(y_vl)

        for col in tqdm(cat_cols):
            beta_encoder = BetaEncoder(col)
            beta_encoder.fit(x_tr, '單價')
            featue_name = f"{col}_mean"
            x_tr[featue_name] = beta_encoder.transform(x_tr, 'mean')
            x_vl[featue_name] = beta_encoder.transform(x_vl, 'mean')
            test_x[featue_name] = beta_encoder.transform(test_x, 'mean')
            private_x[featue_name] = beta_encoder.transform(private_x, 'mean')

        x_tr = x_tr.drop(['單價'] + cat_cols, axis=1)
        x_vl = x_vl.drop(['單價'] + cat_cols, axis=1)
        test_x = test_x.drop(cat_cols, axis=1)
        private_x = private_x.drop(cat_cols, axis=1)

        return x_tr, x_vl, y_tr, y_vl, test_x, private_x

        