"""
Housing price prediction module
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
from.preproc import PreProc

warnings.filterwarnings('ignore')

class Predict(PreProc):
    """
    Predict class
    """
    def __init__(self) -> None:
        super().__init__(
            f'{os.getcwd()}/data/training_data.csv'
        )
        self.model = {
            'RF': RandomForestRegressor(random_state = 0),
            'XGB': XGBRegressor(random_state = 0),
            'LGBM': LGBMRegressor(random_state = 0)
        }
        self.data = super().main().drop(columns = ['鄉鎮市區', '路名'])


    def train_test_split(self) -> tuple:
        """
        Split data into training and testing sets
        """
        features, output = self.data.drop(columns = ['ID', '單價']), self.data['單價']
        x_train, x_test, y_train, y_test = train_test_split(
            features, output, test_size = 0.2, random_state = 0
        )
        return x_train, x_test, y_train, y_test


    def train(self) -> tuple:
        """
        Train model
        """
        x_train, x_test, y_train, y_test = self.train_test_split()
        pred, mae = pd.DataFrame(), {}
        print("Start Training...")
        for name, model in tqdm(self.model.items()):
            model.fit(x_train, y_train)
            pred[name] = model.predict(x_test)
            mae[name] = mean_absolute_error(y_test, pred[name])
        return pred, mae


    def main(self):
        x_train, x_test, y_train, y_test = self.train_test_split()
        print(f"Size of every sets: \
            {x_train.shape},\
            {x_test.shape},\
            {y_train.shape},\
            {y_test.shape}"
        )
        _, mae = self.train()
        print(mae)
        ## pred.to_csv('data/pred.csv', index  = False)


if __name__ == "__main__":
    predict = Predict()
    predict.main()
        