"""
Stacking script for the model
"""
import os
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ..utils.data_utils import (load_data)
warnings.filterwarnings("ignore")


xgb_config = load_data(f"{os.getcwd()}/configs/xgbr.yaml")
cat_config = load_data(f"{os.getcwd()}/configs/catbr.yaml")
lgbm_config = load_data(f"{os.getcwd()}/configs/lgbmr.yaml")


def stacking() -> StackingRegressor:
    """
    stacking regressor

    Returns:
        StackingRegressor: self-defined stacking model
    """
    level_0 = list()
    level_0.append(('xgb', XGBRegressor(**xgb_config)))
    level_0.append(('cat', CatBoostRegressor(**cat_config)))
    level_0.append(('lgbm', LGBMRegressor(**lgbm_config)))
    level_1 = Ridge(alpha=0.5)
    stackmodel = StackingRegressor(estimators=level_0, final_estimator=level_1, cv=5)
    return stackmodel