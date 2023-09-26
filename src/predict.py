"""
Housing price prediction module
"""
import os
import warnings
import pandas as pd
from argparse import ArgumentParser
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
    parser = ArgumentParser()
    parser.add_argument('--dims', type = int, default = 10)
    args = parser.parse_args()
    predict = Predict(args.dims)
    predict.main()
        