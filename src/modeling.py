"""
Modeling
"""
import optuna
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from model_config.mdl_config import (
    xgbr_config,
    lgbmr_config,
    catbr_config
)
from .utils import load_data, train_test_split


class Modeling:
    """
    Training session
    """
    def __init__(self, train_path: str, test_path: str) -> None:
        self.train_data = load_data(train_path)
        self.test_data = load_data(test_path)
        self.feature = self.train_data.drop(columns = ['ID', '單價'])
        self.output = self.train_data['單價']
        self.test_feature = self.test_data.drop(columns = ['ID'])
        self.models = {
            'RFR': RandomForestRegressor(
                random_state=42,
            ),
            'XGBR': XGBRegressor(
                random_state=42,
            ),
            'LGBMR': LGBMRegressor(
                random_state=42,
                verbose=-1
            ),
            'CATR': CatBoostRegressor(
                random_state=42,
                verbose=False
            ),
            'STACKR': self.stacking()
        }

    def stacking(self):
        """
        Stacking model
        """
        level_0 = list()
        level_0.append(
            (
                'xgb', XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    reg_alpha= 0.05,
                    random_state=42
                )
            )
        )
        level_0.append(
            (
                'cat', CatBoostRegressor(
                    iterations=1000,
                    depth=10,
                    learning_rate=0.087,
                    l2_leaf_reg=0.0715564,
                    subsample=0.7963,
                    colsample_bylevel=0.94634,
                    bagging_temperature=0.0709,
                    border_count=232,
                    random_strength=0.63275,
                    verbose=False,
                    random_state=42
                )
            )
        )
        level_0.append(
            (
                'lgbm', LGBMRegressor(
                    num_iterations = 588,
                    learning_rate = 0.018049943310703906,
                    num_leaves = 829,
                    subsample = 0.8920214447324074,
                    colsample_bytree = 0.5330930972309851,
                    min_data_in_leaf = 25,
                    max_bin = 505,
                    random_state=42
                )
            )
        )
        level_1 = Ridge(alpha=0.5)
        stack_model = StackingRegressor(estimators=level_0, final_estimator=level_1, cv=5)
        return stack_model


    def split_data(self, ratio) -> tuple:
        """
        Split data into training and validation set
        """
        x_train, x_valid, y_train, y_valid = train_test_split(self.feature, self.output, ratio)
        return x_train, x_valid, y_train, y_valid


    def train(self, method, x_train, y_train, x_valid, y_valid) -> tuple:
        """
        Train model
        """
        print("Start Training...")
        model = self.models[method]
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mape = mean_absolute_percentage_error(y_valid, pred)
        return model, mape


    def fine_tune(self, method, x_train, x_valid, y_train, y_valid):
        """
        Fine tune the model
        """
        def objective(trial, x_train, y_train, x_valid, y_valid):
            """
            Objective functions
            """
            mapping = {
                'XGBR': (xgbr_config(trial), XGBRegressor()),
                'LGBMR': (lgbmr_config(trial), LGBMRegressor()),
                'CATBR': (catbr_config(trial), CatBoostRegressor())
            }
            config, model = mapping[method]
            model.set_params(**config)
            model.fit(x_train, y_train)
            pred = model.predict(x_valid)
            mape = mean_absolute_percentage_error(y_valid, pred)
            return mape

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, x_train, y_train, x_valid, y_valid),
            n_trials=100
        )
        print(f"Model: {method}")
        print(f"Best params: {study.best_params}")
        print(f"Best score: {study.best_value}")
        return study.best_params
