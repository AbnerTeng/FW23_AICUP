"""
Housing Price prediction module

Author: Yu-Chen, Den
"""
import os
import warnings
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .preproc import PreProc

warnings.filterwarnings('ignore')

class Predict(PreProc):
    """
    Predict class
    
    In the predict class, we inherit the PreProc class to do the data preprocessing first.
    Then, we use the training data to train the model and predict the testing data.
    
    Attributes:
        models: dict
            The dictionary of models
        data: DataFrame
            The training data
        feature: DataFrame
            The features of training data
        output: DataFrame
            The output of training data
    """
    def __init__(self, dims: int) -> None:
        super().__init__(
            f'{os.getcwd()}/data/training_data.csv',
            dims
        )
        self.models = {
            'RF': RandomForestRegressor(
                random_state = 0,
                n_estimators = 300
            ),
            'XGB': XGBRegressor(random_state = 0),
            'LGBM': LGBMRegressor(random_state = 0)
        }
        # self.data = super().main().drop(columns = ['鄉鎮市區', '路名'])
        self.feature, self.output = super().main()


    def train_test_split(self) -> tuple:
        """
        Split data into training and testing sets
        """
        # features, output = self.data.drop(columns = ['ID', '單價']), self.data['單價']
        x_train, x_test, y_train, y_test = train_test_split(
            self.feature, self.output, test_size = 0.2, random_state = 0
        )
        return x_train, x_test, y_train, y_test


    def train(self) -> tuple:
        """
        Train model
        """
        x_train, x_test, y_train, y_test = self.train_test_split()
        pred, mae = pd.DataFrame(), {}
        print("Start Training...")
        for name, model in tqdm(self.models.items()):
            model.fit(x_train, y_train)
            pred[name] = model.predict(x_test)
            mae[name] = mean_absolute_error(y_test, pred[name])
        return pred, mae
