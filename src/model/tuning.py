"""
Tune model parameters using Optuna
"""
import os
import warnings
import yaml
import numpy as np
import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings("ignore")


class ParamTuner:
    """
    tune model parameters using Optuna
    """
    def __init__(self, xt, yt, xv, yv) -> None:
        self.xt = xt
        self.yt = yt
        self.xv = xv
        self.yv = yv


    def fit_and_evaluate_model(self, model) -> float:
        """Evaluation

        Args:
            model: model used

        Returns:
            float: Mean Absolute Percentage Error
        """
        model.fit(self.xt, self.yt)
        y_pred = np.exp(model.predict(self.xv))
        mape = mean_absolute_percentage_error(self.yv, y_pred)
        return mape * 100


    def objective_xgb(self, trial):
        """
        Objective function for XGBoost

        Args:
            trial

        Returns:
            float: Mean Absolute Percentage Error
        """
        config = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'random_state': 42
        }
        model = XGBRegressor(**config)
        return self.fit_and_evaluate_model(model)


    def objective_cat(self, trial):
        """
        Objective function for CatBoost

        Args:
            trial

        Returns:
            float: Mean Absolute Percentage Error
        """
        config = {
            'iterations': trial.suggest_int('iterations', 500, 1300),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'verbose': False,
            'random_state': 42
        }
        model = CatBoostRegressor(**config)
        return self.fit_and_evaluate_model(model)


    def objective_lgbm(self, trial):
        """
        Objective function for LightGBM

        Args:
            trial

        Returns:
            float: Mean Absolute Percentage Error
        """
        config = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1e-2),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1e-2),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'random_state': 42,
            "metric": "mape",
            "num_iterations": trial.suggest_int("num_iterations", 400, 1000), 
            "verbosity": -1,
            "bagging_freq": 1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 64, 2**9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
        }
        model = LGBMRegressor(**config)
        return self.fit_and_evaluate_model(model)


    def optimize(
            self, objective: callable,
            n_trials: int = 100
        ) -> dict:
        """Optimie model parameters using Optuna

        Args:
            objective (callable): model objective function
            x_train (pd.DataFrame)
            y_train (pd.Series)
            x_valid (pd.DataFrame)
            y_valid (pd.Series)
            n_trials (int, optional): Tries. Defaults to 100.

        Returns:
            dict: best parameters
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(
                trial, self.xt, self.yt, self.xv, self.yv
            ), n_trials=n_trials
        )
        return study.best_params


    def save_yml(self, file_path: str, params: dict) -> None:
        """Save parameters to yaml file

        Args:
            file_path (str): yaml file
            params (dict): best parameters
        """
        if not file_path not in os.listdir(f'{os.getcwd()}/configs'):
            with open(file_path, 'w', encoding="utf-8") as yaml_file:
                yaml.dump(params, yaml_file, default_flow_style=False)
        else:
            print(f'{file_path} already exists')