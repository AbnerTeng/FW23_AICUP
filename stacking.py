import os
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def load_data(path):
    return pd.read_csv(path)


def train_test_split(feat, label):
    x_train, x_valid = feat[:int(len(feat) * 0.8)], feat[int(len(feat) * 0.8):]
    y_train, y_valid = label[:int(len(label) * 0.8)], label[int(len(label) * 0.8):]
    return x_train, x_valid, y_train, y_valid


def logarithm(data):
    return data.apply(lambda x: np.log(x))


def stacking():
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


if __name__ == "__main__":
    x_data = load_data(f"{os.getcwd()}/data/train_feat.csv")
    y_data = load_data(f"{os.getcwd()}/data/train_output.csv")
    y_train = load_data(f"{os.getcwd()}/data/y_train.csv")
    y_valid = load_data(f"{os.getcwd()}/data/y_valid.csv")
    public_train = load_data(f"{os.getcwd()}/data/test_feat.csv")
    sub_data = load_data(f"{os.getcwd()}/data/public_submission_template.csv")
    # private_train = load_data(f"{os.getcwd()}/data/test_feat_v2.csv")
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data)
    print(
        f"Train shape: {x_train.shape}, {y_train.shape}, \
            Valid shape: {x_valid.shape}, {y_valid.shape}"
    )
    y_train, y_valid = logarithm(y_train), logarithm(y_valid)
    stack_model = stacking()
    stack_model.fit(x_train, y_train)
    y_pred = stack_model.predict(x_valid)
    y_pred, y_valid = np.exp(y_pred), np.exp(y_valid)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    print(f"MAPE: {mape * 100}")
    public_pred = np.exp(stack_model.predict(public_train))
    sub_data['predicted_price'] = public_pred
    sub_data.to_csv(f"{os.getcwd()}/data/public_submission_stack_v4.csv", index=False)
