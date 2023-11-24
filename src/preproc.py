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
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from .utils import load_data
from .add_social_economic_feature import add_social_economic_feature
from .n_facilities_v2 import NFacilities
from .knn_test import GetNearestNeighbors

warnings.filterwarnings('ignore')


class BetaTargetEncoder:
    """
    Beta Target Encoder for categorical data
    with more precise mean value
    """
    def __init__(self, group) -> None:
        self.group = group
        self.stats = None


    def fit(self, df: pd.DataFrame, target_col: str) -> float:
        """
        Fit the data
        """
        prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(
            columns={
                'sum': 'n',
                'count': 'N'
            },
            inplace=True
        )
        stats.reset_index(level=0, inplace=True)
        self.stats = stats
        return prior_mean


    @staticmethod
    def calculate_statistics(stat_type: str, alpha: float, beta: float) -> float:
        """
        Set up the parameters for the beta distribution
        
        Parameters
        ----------
        stat_type: str
            The type of the statistics as the key of the dictionary

        alpha, beta: float
            The parameters of the beta distribution, it's the value of the dictionary
            where store as the tuple (nom, denom)
        """
        parameters = {
            'mean': (alpha, alpha + beta),
            'mode': (alpha - 1, alpha + beta - 2),
            'median': (alpha - 1/3, alpha + beta - 2/3),
            'var': (alpha * beta, (alpha + beta)**2 * (alpha + beta + 1)),
            'skewness': (
                2 * (beta - alpha) * np.sqrt(alpha + beta + 1),
                (alpha + beta + 2) * np.sqrt(alpha * beta)
            )
        }
        return parameters.get(stat_type, (None, None))


    def transform(
            self, df: pd.DataFrame, prior_mean: float, stat_type: str, n_min: float=1
        ) -> list:
        """
        Transform the data
        """
        assert stat_type in ['mean', 'median', 'mode', 'var', 'skewness', 'kurtosis']
        assert self.stats is not None
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n, N = df_stats['n'].copy(), df_stats['N'].copy()
        nan_indexs = np.isnan(n)
        n[nan_indexs], N[nan_indexs] = prior_mean, 1.0

        ## prior parameters
        n_prior = np.maximum(n_min - N, 0)
        alpha_prior = prior_mean * n_prior
        beta_prior = (1.0 - prior_mean) * n_prior

        ## posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + (N - n)

        num, denom = self.calculate_statistics(stat_type, alpha, beta)
        value = num / denom
        value[np.isnan(value)] = np.nanmedian(value)
        return value


class PreProc(NFacilities):
    """
    Data preprocessing class
    """
    def __init__(self, facility_path, target_path, args, dims=0) -> None:
        super(NFacilities, self).__init__(
            facility_path, target_path, args.rad
        )
        self.model =  RandomForestRegressor(random_state=0)
        self.dims = dims
        self.feat_cols = [
            '土地面積','移轉層次','總樓層數','屋齡','建物面積','車位面積','車位個數','橫坐標','縱坐標',
            '主建物面積','陽台面積','附屬建物面積','N_lib_2000','avg_distances_高中','avg_distances_國小',
            'avg_distances_火車','avg_distances_醫療', 'avg_distances_公車','avg_distances_國中',
            'avg_distances_大學','avg_distances_便利','avg_distances_AT', 'avg_distances_金融',
            'avg_distances_捷運','avg_distances_郵局', 'avg_tax','density','edu_p'
        ]
        self.cat_cols = ['使用分區','主要用途','主要建材','建物型態','縣市']


    def get_data(self, data_path: str) -> pd.DataFrame:
        """
        Get training / testing data
        """
        data = load_data(data_path)
        return data


    def drop_columns(self, data: pd.DataFrame) -> None:
        """
        drop useless columns
        """
        data.drop(['備註'], axis = 1, inplace = True)


    def get_mean_dist(self, target_path: str):
        """
        get the mean value of k nearest neighbors
        """
        for external_datas in tqdm(
            os.listdir(
                f"{os.getcwd()}/data/external_data"
            )
        ):
            file_name = external_datas.split('.')[0].replace('資料', '')
            get_nn = GetNearestNeighbors(
                f"{os.getcwd()}/data/external_data/{external_datas}",
                target_path,
                3, f"mean_distance_to_{file_name}"
            )
            avg_dist = get_nn.main()
            data = get_nn.update_columns(avg_dist)
        return data


    def get_nfac(self, facility_path: str, target_path: str, rad: int):
        """
        get the number of facilities within the radius
        """
        for external_datas in tqdm(
            os.listdir(
                facility_path
            )
        ):
            nfac = NFacilities(
                f"{os.getcwd()}/data/external_data/{external_datas}",
                target_path, rad
            )
            nfac.main()


    def get_social_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        get the social economic information
        """
        data = add_social_economic_feature(data)
        return data


    def categorical_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
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
        for col in self.cat_cols:
            bte = BetaTargetEncoder(col)
            bte.fit(data, '單價')
            feature_name = f"{col}_mean"
            ## TODO: finish
            


    def main(self, data_path: str) -> tuple:
        """
        Main execution function
        """
        data = self.get_data(data_path)
        self.drop_columns(data)
        data = self.categorical_transformation(data)
        data.drop(
            columns = ['鄉鎮市區', '路名', 'ID'],
            inplace = True
        )
        # data = add_social_economic_feature(data)
        print(data.columns)
        return data
